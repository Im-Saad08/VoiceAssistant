"""Action router: validates planned actions against the tool registry and
executes them safely.

- Unknown action types are rejected (LLM hallucination guard).
- Confirm-level actions require explicit user consent before running.
- Each action result is logged and returned for the orchestrator to use.
"""

from __future__ import annotations

import json

from assistant.logging_setup import get_logger
from safety import permissions
from safety.confirmations import confirm
from tools.registry import execute, get_tool, resolve_name, validate_params

logger = get_logger("router")


def route(actions_plan: list[dict] | None) -> list[dict]:
    """Execute each planned action; returns a list of result dicts.

    Two-phase, so a malformed action is rejected with ZERO side effects before
    ANY action in the plan runs — no narration spoken, no confirmation prompt,
    no tool execution:

      1. Validate every action against the registry (type resolves + params).
      2. Execute only the actions that passed validation (the executor
         re-validates as it goes, for defense in depth).

    This is what catches the planner bug where `open_application` was emitted
    without its required `target`: the action is rejected up-front with a clear
    "invalid plan" error instead of failing mid-execution, and a well-formed
    action later in the same plan is never triggered by a bad earlier one.

    Each result: {type, ok, value/error}.
    """
    if not actions_plan:
        return []

    # ---- Phase 1: validate all actions; no side effects. ------------------
    # Outcomes are kept keyed by input index so the returned results mirror
    # the plan's order exactly (a valid action after a malformed one keeps its
    # position instead of being shifted after the rejections).
    rejected: dict[int, dict] = {}
    valid_slots: dict[int, dict] = {}
    for idx, action in enumerate(actions_plan):
        if not isinstance(action, dict):
            logger.warning("router: malformed action (not an object) — rejecting")
            rejected[idx] = {"type": "<malformed>", "ok": False, "error": "malformed action"}
            continue

        action_type = action.get("type", "")
        params = action.get("params") or {}
        resolved = resolve_name(action_type)
        if resolved is None:
            logger.warning("router: unknown action type '%s' — rejecting", action_type)
            rejected[idx] = {"type": action_type, "ok": False, "error": "unknown action"}
            continue

        tool_def = get_tool(resolved)
        assert tool_def is not None  # resolved implies exists.

        try:
            validate_params(tool_def, params)
        except ValueError as exc:
            # Underspecified/malformed params emitted by the planner. Reject
            # loudly and deterministically — never guess missing values here.
            logger.warning("router: invalid plan for %s: %s", resolved, exc)
            rejected[idx] = {"type": resolved, "ok": False, "error": f"invalid plan: {exc}"}
            continue

        valid_slots[idx] = action

    # ---- Phase 2: execute only valid actions. ------------------------------
    executed: dict[int, dict] = {}
    for idx in sorted(valid_slots):
        action = valid_slots[idx]
        action_type = action.get("type", "")
        params = action.get("params") or {}
        narration = action.get("narration")
        resolved = resolve_name(action_type)
        assert resolved is not None  # already validated in phase 1.
        tool_def = get_tool(resolved)
        assert tool_def is not None

        # Narration is the assistant telling the user what's about to happen.
        if narration:
            _speak_safe(narration)

        # Confirmation gate.
        if permissions.requires_confirmation(resolved):
            desc = _describe_action(resolved, params)
            if not confirm(f"This requires confirmation: {desc}. Should I proceed?"):
                executed[idx] = {"type": resolved, "ok": False, "error": "not confirmed"}
                continue

        # Execute.
        logger.info("router: executing %s(%s)", resolved, _safe_repr(params))
        result = execute(tool_def, params)
        logger.info(
            "router: %s -> %s",
            resolved,
            "OK: " + str(result.get("value", ""))[:100] if result.get("ok") else "FAIL: " + str(result.get("error", ""))[:100],
        )
        executed[idx] = {"type": resolved, **result}

    # ---- Merge back into original plan order. ------------------------------
    return [rejected[i] if i in rejected else executed[i] for i in range(len(actions_plan))]


def format_results(results: list[dict]) -> str:
    """One-line summary of action results for context tracking."""
    if not results:
        return ""
    parts = []
    for r in results:
        tag = r["type"]
        if r.get("ok"):
            parts.append(f"{tag}: done")
        else:
            parts.append(f"{tag}: {r.get('error', 'failed')}")
    return "; ".join(parts)


def _describe_action(action_type: str, params: dict) -> str:
    if action_type == "open_application":
        return f"open {params.get('target', 'application')}"
    if action_type == "delete_path":
        return f"permanently delete {params.get('path', 'a file')}"
    if action_type == "browser_search":
        return f"search the web for '{params.get('query', '')}'"
    return f"run {action_type}({', '.join(f'{k}={v}' for k,v in params.items())})"


def _speak_safe(text: str) -> None:
    try:
        from audio.tts import speak
        speak(text)
    except Exception:  # noqa: BLE001
        logger.debug("narration speak failed (non-fatal)", exc_info=True)


def _safe_repr(params: dict) -> str:
    return json.dumps(params, ensure_ascii=False)[:200]
