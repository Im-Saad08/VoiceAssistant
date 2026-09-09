"""Safety levels and permissions for tool execution."""

from __future__ import annotations

from tools.registry import get_tool

# Safety levels.
AUTO = "auto"      # execute immediately
CONFIRM = "confirm"  # require explicit user confirmation first


def safety_level(action_type: str) -> str:
    """Return the safety level for a resolved action type."""
    resolved = get_tool(action_type)
    if resolved is None:
        raise ValueError(f"unknown action type: {action_type}")
    return resolved.safety


def requires_confirmation(action_type: str) -> bool:
    try:
        return safety_level(action_type) == CONFIRM
    except ValueError:
        # Unknown actions always require scrutiny.
        return True
