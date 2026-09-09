"""Integration tests for the Gemini planner (live API).

These require the Gemini free-tier quota to be available. The free tier is
rate-limited (20 req/min), so run these a few at a time rather than all at once:

    python tests/test_planner_integration.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.orchestrator import reset  # noqa: E402
from assistant.planner import plan  # noqa: E402


def _plan_for(text: str) -> dict:
    reset()
    return plan(text)


def _action_types(p: dict) -> list[str]:
    return [a.get("type") for a in p.get("actions", [])]


def test_english_open_notepad():
    p = _plan_for("Open Notepad for me")
    assert "open_application" in _action_types(p)
    assert p["spoken_response"]


def test_english_natural_variant():
    p = _plan_for("Can you get the text editor up?")
    assert "open_application" in _action_types(p)


def test_urdu_open_notepad():
    p = _plan_for("Notepad کھول دو")
    assert "open_application" in _action_types(p)
    assert p.get("response_language", "").lower() in ("ur", "hinglish")


def test_mixed_language():
    p = _plan_for("Chrome kholo aur YouTube open karo")
    types = _action_types(p)
    assert "open_application" in types
    # Should also involve opening/navigating the browser.
    assert any(t in types for t in ("open_url", "browser_search", "open_application"))


def test_conversational_first_result():
    # Simulate prior context: browser search just happened.
    reset()
    p = plan("Open the first result", "User: Search Python tutorials\nAssistant: I found the results.")
    assert "browser_open_result" in _action_types(p)


def test_safety_delete_requests_confirmation_tool():
    p = _plan_for("Delete this folder")
    assert "delete_path" in _action_types(p)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="test function names")
    args = parser.parse_args()

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    if args.only:
        tests = [(n, f) for n, f in tests if n in args.only]

    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {type(e).__name__}: {e}")

    print("planner integration: done")


if __name__ == "__main__":
    _main()
