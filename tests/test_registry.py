"""Tests for the tool registry: registration, aliases, safety, validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.permissions import requires_confirmation  # noqa: E402
from tools.registry import (  # noqa: E402
    all_tools,
    get_tool,
    resolve_name,
    validate_params,
)


def test_known_tools_registered():
    names = {t.name for t in all_tools()}
    for expected in (
        "open_application",
        "close_application",
        "open_url",
        "browser_search",
        "browser_open_result",
        "get_time",
        "get_date",
        "file_search",
        "file_open",
        "delete_path",
    ):
        assert expected in names, f"missing tool {expected}"


def test_alias_resolution():
    assert resolve_name("browser_navigate") == "open_url"
    assert resolve_name("open_search_result") == "browser_open_result"
    assert resolve_name("read_page") == "read_browser_page"
    # Unknown -> None
    assert resolve_name("run_arbitrary_command") is None


def test_safety_levels():
    assert requires_confirmation("delete_path") is True
    assert requires_confirmation("get_time") is False
    assert requires_confirmation("open_application") is False


def test_validate_required_param():
    t = get_tool("open_application")
    # Missing required 'target' -> error
    try:
        validate_params(t, {})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_validate_integer_param():
    t = get_tool("browser_open_result")
    cleaned = validate_params(t, {"index": "2"})
    assert cleaned["index"] == 2
    assert isinstance(cleaned["index"], int)


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("registry: all tests passed")


if __name__ == "__main__":
    _main()
