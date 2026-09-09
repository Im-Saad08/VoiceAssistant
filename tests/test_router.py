"""Tests for the action router: it executes allowed actions and rejects
unknown ones, without ever running arbitrary commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.router import format_results, route  # noqa: E402


def test_route_unknown_action_rejected():
    results = route([{"type": "run_arbitrary_command", "params": {"cmd": "rm -rf /"}}])
    assert len(results) == 1
    assert results[0]["ok"] is False
    assert results[0]["error"] == "unknown action"


def test_route_get_time_auto():
    results = route([{"type": "get_time", "params": {}}])
    assert len(results) == 1
    assert results[0]["ok"] is True
    assert isinstance(results[0]["value"], str)


def test_route_malformed_params():
    # open_application requires 'target'
    results = route([{"type": "open_application", "params": {}}])
    assert results[0]["ok"] is False


def test_route_none_plan():
    assert route(None) == []
    assert route([]) == []


def test_route_missing_params_key():
    # Planner emitted the action with no 'params' field at all.
    results = route([{"type": "open_application"}])
    assert results[0]["ok"] is False
    assert results[0]["error"].startswith("invalid plan: ")


def test_route_invalid_plan_reports_clear_error():
    # The planner bug regression: empty params must be rejected up-front with a
    # clear, deterministic error naming the missing field.
    results = route([{"type": "open_application", "params": {}}])
    assert results[0]["ok"] is False
    assert results[0]["error"].startswith("invalid plan: ")


def test_route_malformed_non_dict_item():
    results = route([{"type": "get_time", "params": {}}, "garbage", 42])
    assert len(results) == 3
    assert results[0]["ok"] is True  # valid action still executes
    assert results[1]["ok"] is False and results[1]["error"] == "malformed action"
    assert results[2]["ok"] is False and results[2]["error"] == "malformed action"


def test_route_rejects_malformed_before_confirm_and_executes_valid():
    # delete_path is a confirm-level tool, but a plan missing its required
    # 'path' must be rejected in validation — before any confirmation prompt is
    # shown — while a valid get_time in the same plan still runs. (No
    # interactive prompt is reachable here; otherwise this test would hang.)
    results = route(
        [
            {"type": "delete_path", "params": {}},
            {"type": "get_time", "params": {}},
        ]
    )
    assert results[0]["ok"] is False
    assert results[0]["error"].startswith("invalid plan: ")
    assert "path" in results[0]["error"]
    assert results[1]["ok"] is True


def test_format_results_empty():
    assert format_results([]) == ""


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("router: all tests passed")


if __name__ == "__main__":
    _main()
