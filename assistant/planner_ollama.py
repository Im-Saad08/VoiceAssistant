"""Ollama-based intent planner (local, free, no API quota).

Uses the Ollama HTTP API at localhost:11434 with qwen2.5:1.5b by default.
Returns the same {spoken_response, actions, response_language} contract as
the Gemini planner so the router, safety layer, and orchestrator need no
changes.
"""

from __future__ import annotations

import json
import time

import requests

from assistant.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_RETRY_BACKOFF_SECONDS,
    OLLAMA_TEMPERATURE,
)
from assistant.logging_setup import get_logger
from assistant.planner_utils import (
    _coerce_actions,
    _compact_catalog,
    _fallback,
    _normalize_website_actions,
    _website_url,
)

logger = get_logger("planner_ollama")

_client: requests.Session | None = None


def _get_client() -> requests.Session:
    """Lazily create a requests Session for the Ollama HTTP API."""
    global _client
    if _client is None:
        _client = requests.Session()
    return _client


# ---- Public API (called by planner.py dispatcher) --------------------------

MAX_RETRIES = OLLAMA_MAX_RETRIES
RETRY_BACKOFF = OLLAMA_RETRY_BACKOFF_SECONDS


# ---- Website routing (planner-side) -----------------------------------------
# qwen2.5:1.5b reliably routes "open youtube / google / go to <site>" to open_url
# from the prompt alone, but it still over-applies open_application to some
# service names (gmail, github, instagram). Rather than weaken validation or add
# fuzzy guessing in the router, we apply a deterministic, allowlisted mapping
# here in the planner: an open_application whose target is a known website is
# rewritten to open_url with the mapped URL. Only exact string matches against
# this table count — no heuristics — and the router still validates the result.
# Desktop apps (notepad, chrome, calc, ...) are never in this table.
#
# The canonical implementation lives in assistant.planner_utils (_route_actions)
# so BOTH backends share it; these names are re-exported here for compatibility.

def plan(user_text: str, conversation_context: str = "") -> dict:
    """Send the user's utterance to the local Ollama planner and return a plan.

    Returns {spoken_response, actions, response_language} on success.
    On failure, returns a safe fallback with no actions.
    """
    system = _build_system_prompt_ollama()
    prompt_parts: list[str] = []
    if conversation_context:
        prompt_parts.append(f"[Conversation context]\n{conversation_context}")
    prompt_parts.append(f"[User says]\n{user_text}")
    prompt = "\n".join(prompt_parts)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
        },
        "format": "json",
    }

    client = _get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.post(url, json=payload, timeout=OLLAMA_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            parsed = _parse_ollama_response(content)
            parsed["actions"] = _normalize_website_actions(parsed["actions"])
            return parsed

        except requests.ConnectionError as exc:
            logger.warning(
                "ollama planner: connection failed attempt %d (%s)",
                attempt,
                exc,
            )
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                logger.info("ollama planner: waiting %.0fs before retry", wait)
                time.sleep(wait)
                continue
            return _fallback(
                "I can't reach the local AI server. "
                "Is Ollama running? Start it with: ollama serve"
            )

        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                logger.error(
                    "ollama planner: model '%s' not found locally", OLLAMA_MODEL
                )
                return _fallback(
                    f"Model '{OLLAMA_MODEL}' is not installed. "
                    f"Run: ollama pull {OLLAMA_MODEL}"
                )
            logger.warning(
                "ollama planner: HTTP %s attempt %d", status, attempt
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF)
                continue
            return _fallback("The local AI server returned an error.")

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "ollama planner: malformed output attempt %d: %s",
                attempt,
                exc,
            )
            return _fallback(
                "I had trouble understanding the AI response. Could you say it again?"
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ollama planner: attempt %d failed (%s: %s)",
                attempt,
                type(exc).__name__,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF)
                continue
            return _fallback("I'm having trouble with the local AI service.")

    return _fallback("Something went wrong with the planner.")


# ---- Ollama-specific system prompt -----------------------------------------

def _build_system_prompt_ollama() -> str:
    """System prompt tuned for small local models (qwen2.5:1.5b).

    The shared Gemini prompt dumps the full JSON-schema tool catalog (~1400
    tokens) and only says "Return ONLY valid JSON".  A 1.5B model gets lost in
    that and echoes the catalog's shape back as a flat {"action": ...} object —
    it never wraps output in the required {spoken_response, actions,
    response_language} contract.  This variant:
      - uses a one-line-per-tool compact catalog (much shorter),
      - shows a concrete filled-in example of the EXACT output shape,
      - states the three required keys by name,
    which small instruction models follow reliably.
    """
    return (
        "You are a desktop voice assistant (like JARVIS). You understand "
        "English, Urdu, and Hinglish.\n\n"
        "TOOLS you may use (ONLY these):\n"
        f"{_compact_catalog()}\n\n"
        "LANGUAGE RULE:\n"
        '- If the user writes English, write spoken_response in English and set '
        'response_language "en".\n'
        '- If the user writes Urdu script, write spoken_response in Urdu and '
        'set response_language "ur".\n'
        '- If the user mixes Urdu/Hindi with English (Hinglish), reply in '
        'Hinglish and set response_language "hinglish".\n\n'
        "OUTPUT FORMAT — respond with ONLY ONE JSON object, exactly these "
        "three keys:\n"
        '  "spoken_response":   short natural sentence (1-2 sentences) in the '
        "user's language.\n"
        '  "actions":           a list of action objects '
        '{"type": "<tool name>", "params": {"<required param>": "<value>"}}. '
        "Use [] when no action is needed.\n"
        '  "response_language": "en" | "ur" | "hinglish".\n\n'
        'Example for "Open Notepad for me":\n'
        '{"spoken_response":"Opening Notepad.","actions":['
        '{"type":"open_application","params":{"target":"notepad"}}],'
        '"response_language":"en"}\n\n'
        "RULES:\n"
        "- Websites (YouTube, Gmail, Google, GitHub, Instagram, ...) and any "
        "online service -> open_url with the site's url. NEVER open_application "
        "for a website. open_application is ONLY for desktop programs: "
        "notepad, calculator, paint, file explorer, cmd, powershell, task "
        "manager, vscode, chrome.\n"
        '- "the first result"/"that result" -> browser_open_result with index 1 '
        "(an integer).\n"
        '- "the second result" -> browser_open_result with index 2 (an integer).\n'
        '- "go back"/"back" -> browser_go_back.\n'
        '- "go to <site>" or "open <site>" -> open_url (a website), not '
        "open_application.\n"
        "- If [Browser state] says a browser is already open, REUSE it for "
        "navigation — do NOT emit 'Open Chrome' again.\n"
        "- Math: 'what is 25 times 13' -> calculate with the expression "
        "(25*13). Only when the user wants it ON the calculator app "
        "(calculator pe / on the calculator) -> calculator_perform.\n"
        "- If you emit actions, spoken_response should describe what you are "
        "DOING about to happen, never claim it already succeeded.\n"
        "- Ambiguous request -> make a reasonable assumption and proceed.\n"
        "- Never use a tool name that is not listed above.\n"
        "- ALWAYS include every REQUIRED param for the tool you choose.\n"
        "- Output the JSON object only. No markdown, no preamble, no commentary.\n"
    )


# ---- Response parsing -------------------------------------------------------

def _parse_ollama_response(content: str) -> dict:
    """Parse the JSON content from the Ollama chat response.

    Handles common small-model wobble: extra text around JSON, missing keys,
    non-list actions.
    """
    # Strip markdown code fences if the model wraps its output.
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop opening fence (```json or ```) and closing fence.
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    parsed = json.loads(text)

    # Ensure required keys exist.
    if "spoken_response" not in parsed:
        raise ValueError("missing 'spoken_response' in planner output")
    parsed.setdefault("response_language", "en")
    parsed["actions"] = _coerce_actions(parsed.get("actions"))
    return parsed
