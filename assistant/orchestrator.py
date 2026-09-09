"""Orchestrator: ties the full turn cycle together.

    User text → Plan → Route/Execute → Speak + log (after results are known)

Called once per turn from main.py's input loop.

The response is spoken ONLY after execution results are known, so the assistant
can never claim success for an action that failed: if anything in the plan
fails, the spoken line is rebuilt from the actual outcomes instead of playing
the planner's intent-narration verbatim.
"""

from __future__ import annotations

import time

from assistant.context import ConversationContext
from assistant.logging_setup import get_logger
from assistant.planner import plan
from assistant.router import format_results, route

logger = get_logger("orchestrator")

_context = ConversationContext()

# Spoken failure templates per supported response language. The {detail} part
# carries the real error ("browser_open_result: no search results available").
_FAIL_LANG = {
    "en": {
        "none_ok": "I couldn't do that. {detail}",
        "some_ok": (
            "I completed {ok} steps, but {failed} could not be completed: {detail}"
        ),
    },
    "ur": {
        "none_ok": "میں یہ نہیں کر سکا۔ {detail}",
        "some_ok": "{ok} کام مکمل ہوئے، لیکن {failed} مکمل نہیں ہو سکے: {detail}",
    },
    "hinglish": {
        "none_ok": "Mujh se ye nahi ho saka. {detail}",
        "some_ok": "{ok} steps complete hue, lekin {failed} complete nahi hue: {detail}",
    },
}


def reset() -> None:
    """Clear conversation history (start of a new session)."""
    global _context
    _context = ConversationContext()
    try:
        from tools.calculator import reset_calculator

        reset_calculator()
    except Exception:  # noqa: BLE001
        logger.debug("calculator reset unavailable", exc_info=True)


def run_turn(user_text: str) -> str | None:
    """Execute a single conversation turn. Returns the spoken response text (or None)."""
    start = time.perf_counter()

    # ---- 1. Record user turn.
    _context.add_user(user_text)
    logger.info("user: %s", user_text)

    # ---- 2. Ask the planner, feeding the live browser and calculator state so
    #         relative commands ("the first result", "go back", "there",
    #         "do it there", "that") and the deterministic routing layer agree
    #         with what the router will see.
    browser_snapshot = _browser_snapshot()
    calc_snapshot = _calculator_snapshot()
    planner_ctx = _context.summary()
    browser_block = _browser_context_block()
    if browser_block:
        planner_ctx = f"{planner_ctx}\n\n[Browser state]\n{browser_block}"
    calc_block = _calculator_context_block()
    if calc_block:
        planner_ctx = f"{planner_ctx}\n\n[Calculator state]\n{calc_block}"

    plan_start = time.perf_counter()
    result = plan(
        user_text,
        planner_ctx,
        browser_state=browser_snapshot,
        calculator_state=calc_snapshot,
    )
    plan_ms = (time.perf_counter() - plan_start) * 1000

    spoken_response = result.get("spoken_response", "")
    actions_plan = result.get("actions", [])
    lang = result.get("response_language", "en")
    _context.set_language(lang)
    logger.info(
        "plan: lang=%s actions=%d (%.0fms LLM)",
        lang,
        len(actions_plan),
        plan_ms,
    )

    # ---- 3. Execute actions. Nothing is spoken yet.
    results: list[dict] = []
    if actions_plan:
        route_start = time.perf_counter()
        results = route(actions_plan)
        route_ms = (time.perf_counter() - route_start) * 1000
        result_summary = format_results(results)
        _context.last_action_summary = result_summary
        logger.info("route: %s (%.0fms)", result_summary, route_ms)
    else:
        _context.last_action_summary = None

    # ---- 4. Build the response from actual outcomes, speak it once, then
    #         record it in conversation context AFTER execution settled.
    final_text = _truthful_response(spoken_response, results, lang)
    if final_text:
        _speak(final_text)
        _context.add_assistant(final_text)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info("turn total: %.0fms", elapsed)

    return final_text or None


def _truthful_response(llm_text: str, results: list[dict], lang: str) -> str:
    """Return a spoken line that matches what actually happened.

    All actions succeeded -> the planner's narration (it describes the actions
    in present/future terms, which is accurate once they ran). The moment any
    action failed, the planner's text is NOT played verbatim — instead a line
    composed from the real outcome is used, so "Opening YouTube." is never
    spoken when navigation actually failed.
    """
    failures = [r for r in results if not r.get("ok")]
    if not failures:
        return llm_text or ""

    detail = "; ".join(
        f"{r.get('type', 'action')}: {r.get('error', 'failed')}" for r in failures
    )
    ok_count = len(results) - len(failures)
    templates = _FAIL_LANG.get(lang, _FAIL_LANG["en"])

    if ok_count == 0:
        return templates["none_ok"].format(detail=detail)
    return templates["some_ok"].format(
        ok=ok_count, failed=len(failures), detail=detail
    )


def _browser_snapshot() -> dict:
    """Live browser-state dict for planner routing ({} if unavailable/closed)."""
    try:
        from tools.browser import browser_state_snapshot

        return browser_state_snapshot()
    except Exception:  # noqa: BLE001
        logger.debug("browser state unavailable at turn start", exc_info=True)
        return {}


def _browser_context_block() -> str:
    """Short [Browser state] text block for the planner prompt ("" if closed)."""
    try:
        from tools.browser import browser_state_text

        return browser_state_text()
    except Exception:  # noqa: BLE001
        logger.debug("browser context block unavailable", exc_info=True)
        return ""


def _calculator_snapshot() -> dict:
    """Live calculator-state dict for planner routing ({} if unavailable)."""
    try:
        from tools.calculator import calculator_snapshot

        return calculator_snapshot()
    except Exception:  # noqa: BLE001
        logger.debug("calculator state unavailable at turn start", exc_info=True)
        return {}


def _calculator_context_block() -> str:
    """Short [Calculator state] text block for the planner prompt ("" if idle)."""
    try:
        from tools.calculator import calculator_state_text

        return calculator_state_text()
    except Exception:  # noqa: BLE001
        logger.debug("calculator context block unavailable", exc_info=True)
        return ""


def _speak(text: str) -> None:
    """Speak with error isolation — TTS failure never crashes a turn."""
    try:
        from audio.tts import speak

        speak(text)
    except Exception:  # noqa: BLE001
        logger.exception("TTS speak failed (non-fatal)")
