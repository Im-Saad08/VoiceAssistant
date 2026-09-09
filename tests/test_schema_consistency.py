"""Verifies the Gemini structured-output schema is derived correctly from the
tool registry: every registered tool appears as a branch, and each branch's
`params.required` matches the registry's declared required fields.

This is the offline guard for the planner bug where `open_application` was
emitted without its required `target` — if the schema stops carrying a tool's
required params, this test fails before any Gemini quota is spent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.genai import types as genai_types  # noqa: E402

from assistant.planner_utils import _build_response_schema  # noqa: E402
from tools.registry import all_tools, get_tool, validate_params  # noqa: E402


def _branches() -> list:
    schema = _build_response_schema()
    items = schema.properties["actions"].items
    return list(items.any_of)


def _branch_for(name: str):
    for b in _branches():
        if b.properties["type"].enum[0] == name:
            return b
    raise AssertionError(f"no schema branch for tool '{name}'")


def test_schema_covers_all_registered_tools():
    covered = {b.properties["type"].enum[0] for b in _branches()}
    assert covered == {t.name for t in all_tools()}


def test_schema_top_level_requires():
    schema = _build_response_schema()
    assert schema.required == ["spoken_response", "actions"]


def test_schema_required_params_match_registry():
    for t in all_tools():
        branch = _branch_for(t.name)
        declared_req = set((t.params or {}).get("required", []))
        assert set(branch.properties["params"].required) == declared_req, (
            f"schema params.required for {t.name} != registry (extra or missing)"
        )
        declared_props = set((t.params or {}).get("properties", {}).keys())
        assert set(branch.properties["params"].properties.keys()) == declared_props, (
            f"schema params.properties for {t.name} has extra/missing keys"
        )


def test_open_application_target_required_in_schema():
    """Regression test for the planner bug: target must be declared required."""
    branch = _branch_for("open_application")
    assert "target" in branch.properties["params"].required


def test_integer_param_maps_to_integer_type():
    branch = _branch_for("browser_open_result")
    assert (
        branch.properties["params"].properties["index"].type == genai_types.Type.INTEGER
    )


def test_validate_params_rejects_empty_for_any_required_tool():
    """The Python guard must still reject every under-specified action."""
    for t in all_tools():
        declared_req = (t.params or {}).get("required", [])
        if not declared_req:
            continue
        try:
            validate_params(t, {})
        except ValueError:
            continue
        raise AssertionError(f"{t.name} with empty params should be rejected")


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("schema consistency: all tests passed")


if __name__ == "__main__":
    _main()