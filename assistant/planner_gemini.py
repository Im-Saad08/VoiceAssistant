"""Gemini-based intent planner.

Takes recognized speech + conversation context and returns a structured plan
(action list + spoken response + language) via Gemini structured output.

The LLM never executes anything directly — it only decides WHAT should happen.
Python validates the plan against the tool registry before anything runs.
"""

from __future__ import annotations

import json
import time

from google import genai
from google.genai import types

from assistant.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RETRY_BACKOFF_SECONDS,
    GEMINI_TEMPERATURE,
)
from assistant.logging_setup import get_logger
from assistant.planner_utils import (
    _build_response_schema,
    _build_system_prompt,
    _coerce_actions,
    _fallback,
)

logger = get_logger("planner_gemini")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


_RESPONSE_SCHEMA = _build_response_schema()


# ---- Public API (called by planner.py dispatcher) --------------------------

def plan(user_text: str, conversation_context: str = "") -> dict:
    """Send the user's utterance to Gemini and return the parsed plan dict.

    Returns {spoken_response, actions, response_language} on success.
    On failure, returns a safe fallback (no actions, error message spoken).
    """
    system = _build_system_prompt()
    contents_parts: list[str] = []
    if conversation_context:
        contents_parts.append(f"[Conversation context]\n{conversation_context}\n")
    contents_parts.append(f"[User says]\n{user_text}")
    contents = "\n".join(contents_parts)

    client = _get_client()
    start = time.perf_counter()

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=_RESPONSE_SCHEMA,
                    temperature=GEMINI_TEMPERATURE,
                ),
            )
            elapsed = time.perf_counter() - start
            logger.info("planner: LLM responded in %.2fs (attempt %d)", elapsed, attempt)
            parsed = json.loads(resp.text)
            if "spoken_response" not in parsed:
                raise ValueError("missing 'spoken_response' in planner output")
            parsed.setdefault("response_language", "en")
            parsed["actions"] = _coerce_actions(parsed.get("actions"))
            return parsed

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("planner: malformed output attempt %d: %s", attempt, exc)
            return _fallback("I had trouble understanding that. Could you say it again?")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "planner: attempt %d failed (%s: %s)", attempt, type(exc).__name__, exc
            )
            if _is_daily_quota_exhausted(exc):
                logger.warning("planner: daily/project quota exhausted — not retrying")
                return _fallback("I'm having trouble reaching the AI service right now.")
            if attempt < GEMINI_MAX_RETRIES:
                wait = _rate_limit_wait(exc) or GEMINI_RETRY_BACKOFF_SECONDS
                logger.info("planner: waiting %.0fs before retry", wait)
                time.sleep(wait)
                continue
            return _fallback("I'm having trouble reaching the AI service right now.")

    return _fallback("Something went wrong with the planner.")


# ---- Error classification ---------------------------------------------------

def _is_daily_quota_exhausted(exc: Exception) -> bool:
    """True when Gemini reports the DAILY/PROJECT free-tier quota is exhausted.

    Looks like e.g. 429 RESOURCE_EXHAUSTED with
      "You exceeded your current quota ... generate_content_free_tier_requests"
      quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier

    Retrying seconds later can never help this condition — it resets on a
    timescale of hours/day — so the planner fails fast instead of sleeping
    through the whole retry budget.
    """
    try:
        low = str(exc).lower()
        if "you exceeded your current quota" in low:
            return True
        if "perday" in low and "quota" in low:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _rate_limit_wait(exc: Exception) -> float | None:
    """Extract a suggested retry delay (seconds) from a 429/503 error if present."""
    try:
        msg = str(exc)
        import re as _re

        m = _re.search(r"retry in ([\d.]+)s", msg, _re.I)
        if m:
            return min(float(m.group(1)), 75.0)
        if "RATE" in msg.upper() or "RESOURCE_EXHAUSTED" in msg.upper() or "503" in msg:
            return GEMINI_RETRY_BACKOFF_SECONDS + 3.0
    except Exception:  # noqa: BLE001
        pass
    return None
