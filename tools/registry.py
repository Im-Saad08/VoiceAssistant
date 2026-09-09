"""Central tool/action registry.

Every capability the assistant may execute is an allowlisted Tool with:
  - name            (machine name, used by the LLM planner)
  - description     (what it does, for the planner prompt)
  - params          (JSON-schema dict describing valid arguments)
  - safety          ("auto" = run immediately, "confirm" = ask first)
  - handler         (the Python implementation)

The LLM may only reference tools registered here. The router validates every
planned action against the registry before anything executes, so the LLM can
never cause arbitrary code to run.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from assistant.logging_setup import get_logger

logger = get_logger("registry")


@dataclass
class Tool:
    name: str
    description: str
    params: dict  # JSON schema for arguments
    safety: str  # "auto" | "confirm"
    handler: Callable
    aliases: list = field(default_factory=list)


_REGISTRY: dict[str, Tool] = {}
# Map display names / aliases -> canonical tool name.
_ALIASES: dict[str, str] = {}


def tool(name: str, description: str, params: dict, safety: str = "auto", aliases=None):
    """Register a handler as an executable tool."""

    def decorator(fn: Callable) -> Callable:
        t = Tool(
            name=name,
            description=description,
            params=params,
            safety=safety,
            handler=fn,
            aliases=list(aliases or []),
        )
        _REGISTRY[name] = t
        for a in t.aliases:
            _ALIASES[a] = name
        logger.debug("registered tool: %s", name)
        return fn

    return decorator


def get_tool(name: str) -> Tool | None:
    name = _ALIASES.get(name, name)
    return _REGISTRY.get(name)


def resolve_name(name: str) -> str | None:
    name = _ALIASES.get(name, name)
    return name if name in _REGISTRY else None


def all_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def tool_catalog() -> list[dict[str, Any]]:
    """A planner-friendly catalog of every tool (name, description, params)."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "params": t.params,
        }
        for t in _REGISTRY.values()
    ]


def validate_params(tool_def: Tool, params: Any) -> dict:
    """Best-effort validation of a planned action's arguments.

    We do a pragmatic check (types and required keys) rather than pull in a
    full JSON-schema validator. Returns a cleaned dict; raises ValueError if
    the params are clearly unusable.
    """
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"tool '{tool_def.name}' params must be an object, got {type(params).__name__}")

    schema = tool_def.params or {}
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for key in required:
        if key not in params or params[key] in (None, ""):
            raise ValueError(f"tool '{tool_def.name}' missing required param '{key}'")

    cleaned: dict = {}
    for key, value in params.items():
        prop = properties.get(key, {})
        if prop.get("type") == "integer":
            cleaned[key] = int(value)
        elif prop.get("type") == "number":
            cleaned[key] = float(value)
        else:
            cleaned[key] = value
    return cleaned


def execute(tool_def: Tool, params: dict) -> Any:
    """Run a tool, capturing failures into a structured result."""
    try:
        kwargs = validate_params(tool_def, params)
        result = tool_def.handler(**kwargs)
        return {"ok": True, "value": result}
    except Exception as exc:  # noqa: BLE001 - surface as a friendly result
        logger.exception("tool '%s' failed", tool_def.name)
        return {"ok": False, "error": str(exc)}


# Import tool modules so their @tool decorators register.
# (Order matters only for readability; registration is independent.)
from tools import applications, browser, calculator, files, system  # noqa: E402,F401
