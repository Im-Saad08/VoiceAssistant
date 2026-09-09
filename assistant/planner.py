"""Backend-agnostic planner dispatcher.

Selects the active LLM planner (Ollama or Gemini) based on the
``PLANNER_BACKEND`` config setting and delegates to it.

Public API
--------
plan(user_text, conversation_context="") -> dict
    Returns {spoken_response, actions, response_language}.

Backend implementations live in assistant.planner_ollama and
assistant.planner_gemini.  This module only routes to the configured backend
and applies shared contract normalization.
"""

from __future__ import annotations

from assistant import config
from assistant.logging_setup import get_logger
from assistant.planner_utils import (
    _coerce_actions,
    _normalize_app_targets,
    _resolve_intents,
    _route_actions,
    _website_url,
)

logger = get_logger("planner")

_KNOWN_LANGUAGES = {"en", "ur", "hinglish"}

# Tiny, explicit allowlist of bare browser-relative commands that must ALWAYS
# act on the open browser. Only fires when the planner returned NO actions and
# the browser is actually open — never overrides a real plan.
_EXACT_BACK_PHRASES = {
    "go back",
    "back",
    "go back please",
    "back please",
    "peeche",
    "peeche jao",
    "wapis jao",
    "wapis chalo",
}

# Small, explicit search-intent prefixes. Used ONLY to repair a plan that is
# empty or pure navigation when the 1.5B model echoes open_url to the current
# page instead of emitting browser_search — never as a general utterance
# matcher, and never when the plan already contains a browser_search (or any
# other non-navigation tool such as file_search).
_SEARCH_PREFIXES = (
    "search for ",
    "search ",
    "look up ",
    "look for ",
    "find ",
    "google ",
    "dhundho ",
    "dhund ",
    "khojo ",
)

# Deterministic rescue for a completely failed plan (e.g. qwen emits malformed
# JSON and the fallback has zero actions): still get the REQUIRED scenarios
# right. Exact-prefix + known-website only; desktop apps never match.
_WEBSITE_PREFIXES = (
    "go to the ",
    "go to ",
    "take me to ",
    "navigate to ",
    "open ",
)


# ---- Backend dispatch -------------------------------------------------------

def plan(
    user_text: str,
    conversation_context: str = "",
    browser_state: dict | None = None,
    calculator_state: dict | None = None,
) -> dict:
    """Route the planning request to the configured LLM backend.

    Returns {spoken_response, actions, response_language} on success.
    On any failure, returns a safe fallback with no actions.

    ``browser_state`` and ``calculator_state`` are live snapshots (see
    tools.browser.browser_state_snapshot / tools.calculator.calculator_snapshot)
    used for deterministic routing guarantees; each is read lazily when None.
    """
    # Read the backend setting at call time so tests (and .env changes) can
    # switch backends without an import-order dependency.
    backend = config.PLANNER_BACKEND
    if backend == "ollama":
        from assistant.planner_ollama import plan as _plan
    else:
        from assistant.planner_gemini import plan as _plan

    result = _plan(user_text, conversation_context)
    return _finalize_plan(result, user_text, browser_state, calculator_state)


def validate_plan(result: dict) -> dict:
    """Public alias for _validate_plan (see below)."""
    return _validate_plan(result)


def _validate_plan(result: dict) -> dict:
    """Ensure the plan dict has the required contract shape.

    Applied to EVERY backend's output so the orchestrator always receives a
    consistent {spoken_response, actions, response_language} dict.
    """
    if "spoken_response" not in result:
        raise ValueError("missing 'spoken_response' in planner output")
    result.setdefault("response_language", "en")
    result["actions"] = _coerce_actions(result.get("actions"))
    return result


def _finalize_plan(
    result: dict,
    user_text: str,
    browser_state: dict | None = None,
    calculator_state: dict | None = None,
) -> dict:
    """Deterministic invariants that no LLM size should violate.

    Contract shape -> valid response_language -> canonical app targets ->
    routing -> intent resolution (open-target/calculator/result-index) ->
    routing again -> bare browser-relative command fallback. Applied to BOTH
    backends after their own parsing.
    """
    plan = _validate_plan(result)
    _normalize_response_language(plan)
    _normalize_language_for_script(plan, user_text)
    if browser_state is None:
        browser_state = _lazy_browser_state()
    if calculator_state is None:
        calculator_state = _lazy_calculator_state()

    actions = _normalize_app_targets(plan.get("actions") or [])
    actions = _route_actions(actions, browser_state)
    plan["actions"] = actions
    _resolve_intents(plan, user_text, browser_state, calculator_state)
    # Re-route after _resolve_intents so a deterministic action (bare open_url,
    # website-as-open_application) is still mapped by the same allowlist.
    plan["actions"] = _route_actions(plan.get("actions") or [], browser_state)

    _apply_search_override(plan, user_text)
    if not plan.get("actions"):
        # The LLM produced nothing usable (empty or failed plan): rescue the
        # exact browser-relative / website-relative commands this system is
        # REQUIRED to get right, off explicit allowlists only.
        _apply_browser_back_fallback(plan, user_text, browser_state)
        _apply_website_fallback(plan, user_text)
    return plan


def _normalize_response_language(plan: dict) -> None:
    """Constrain response_language to the supported set, always canonicalized.

    Valids are stored lowercase ("EN"/"  Ur  " -> "en"/"ur"); anything outside
    the supported set defaults to "en". The orchestrator and TTS rely on this
    being exactly one of {en, ur, hinglish}.
    """
    lang = str(plan.get("response_language") or "en").strip().lower()
    plan["response_language"] = lang if lang in _KNOWN_LANGUAGES else "en"


# Letters of the Urdu/Perso-Arabic script. A request actually written in this
# script is unambiguously Urdu input, yet the LLM sometimes tags the response
# "en". Script detection is deterministic and has zero false positives: no
# English or Roman-Urdu utterance contains these code points.
_URDU_SCRIPT_RANGES = (
    (0x0600, 0x06FF),  # Arabic block (Urdu/Perso-Arabic letters & marks)
    (0x0750, 0x077F),  # Arabic Supplement (e.g. ٹ ڈ ڑ)
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def _has_urdu_script(text: str) -> bool:
    return any(
        lo <= ord(ch) <= hi for lo, hi in _URDU_SCRIPT_RANGES for ch in (text or "")
    )


def _normalize_language_for_script(plan: dict, user_text: str) -> None:
    """Force response_language "ur" for Urdu-script input (deterministic).

    The LLM (any size, any backend) is the language authority for ambiguous
    input, but Arabic/Urdu script is not ambiguous — the user wrote Urdu, so the
    response must be Urdu. Applied like every other finalizer normalization,
    independent of which backend produced the plan.
    """
    if _has_urdu_script(user_text or ""):
        plan["response_language"] = "ur"


def _lazy_browser_state() -> dict:
    try:
        from tools.browser import browser_state_snapshot

        return browser_state_snapshot()
    except Exception:  # noqa: BLE001
        logger.debug("browser state unavailable at plan time", exc_info=True)
        return {}


def _lazy_calculator_state() -> dict:
    try:
        from tools.calculator import calculator_snapshot

        return calculator_snapshot()
    except Exception:  # noqa: BLE001
        logger.debug("calculator state unavailable at plan time", exc_info=True)
        return {}


def _apply_browser_back_fallback(
    plan: dict,
    user_text: str,
    browser_state: dict | None,
) -> None:
    """Turn exactly-entered back commands into browser_go_back when the browser
    is open and the planner produced no actions.

    Narrow by design: exact phrases only, and only when the browser is already
    open — this never invents actions for arbitrary chatter.
    """
    if plan.get("actions"):
        return
    if not (browser_state and browser_state.get("open")):
        return
    utterance = " ".join((user_text or "").split()).strip().lower().rstrip(".,!?")
    if utterance in _EXACT_BACK_PHRASES:
        plan["actions"] = [{"type": "browser_go_back", "params": {}}]


def _apply_search_override(plan: dict, user_text: str) -> None:
    """Repair a misrouted search command: plan empty or only navigation.

    qwen2.5:1.5b occasionally echoes ``open_url`` to the browser's current page
    for "search for <topic>" instead of emitting ``browser_search``. This rule
    corrects ONLY that failure shape and only from an explicit allowlist:

      - the utterance starts with a known search prefix, AND
      - the plan contains no browser_search already, AND
      - every planned action is navigation-ish (open_url / open_application) or
        nothing — a plan that already has file_search, browser_open_result, or
        any other tool is left untouched.
    """
    actions = plan.get("actions") or []
    if any(
        isinstance(a, dict) and a.get("type") == "browser_search" for a in actions
    ):
        return
    query = _search_query(user_text or "")
    if query is None:
        return
    if not actions or all(
        isinstance(a, dict) and a.get("type") in ("open_url", "open_application")
        for a in actions
    ):
        plan["actions"] = [{"type": "browser_search", "params": {"query": query}}]


def _search_query(user_text: str) -> str | None:
    """Extract the search query if the utterance emits search intent."""
    original = " ".join(user_text.strip().split())
    low = original.lower().rstrip(".,!?")
    if not low:
        return None
    for prefix in _SEARCH_PREFIXES:
        if not low.startswith(prefix):
            continue
        words = original[len(prefix):].split()
        if not words:
            return None
        # "search karo <topic>" (Hinglish intensifier after the verb).
        while words and words[0].lower() in ("karo", "kijiye", "kar do", "karoo", "ker"):
            words = words[1:]
        if words and words[0].lower() == "for":
            words = words[1:]
        query = " ".join(words).strip(".,!? ").strip()
        if not query:
            return None
        # "google chrome kholo" means open the browser, not a web search.
        if query.lower().startswith("chrome"):
            return None
        return query
    return None


def _apply_website_fallback(plan: dict, user_text: str) -> None:
    """Rescue a completely failed plan into a known-website open_url.

    Fires only when the plan has NO actions (the LLM returned a safe fallback
    for a valid request) and the utterance is an exact known-site command
    ("go to YouTube", "open Gmail"). Desktop-app names are never websites, so
    "open notepad" / "open chrome" are left untouched.
    """
    if plan.get("actions"):
        return
    original = " ".join((user_text or "").strip().split())
    if not original:
        return
    low = original.lower()
    for prefix in _WEBSITE_PREFIXES:
        if low.startswith(prefix):
            site = original[len(prefix):].strip(".,!? ").strip()
            url = _website_url(site)
            if url is not None:
                plan["actions"] = [{"type": "open_url", "params": {"url": url}}]
                # The safe fallback text ("I had trouble...") described a
                # failure; replace it with an honest description of the action.
                plan["spoken_response"] = f"Opening {site[0:1].upper() + site[1:]}."
            return
