"""Tests for the orchestrator's turn lifecycle.

Locks in the stabilization-pass guarantees:

  - the spoken response is built from ACTUAL execution results, never from the
    planner's intent-narration when an action failed (no false success claims);
  - the response is spoken exactly once, AFTER execution;
  - a TTS failure never corrupts the turn or the stored context;
  - the live [Browser state] block is fed to the planner so relative commands
    resolve against the real open browser;
  - conversation context is updated after execution settles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assistant.orchestrator as orch  # noqa: E402

_ORIG = {}


def _patch(attr, value):
    _ORIG[attr] = getattr(orch, attr, None)
    setattr(orch, attr, value)


def _patch_browser(snapshot, block):
    _patch("_browser_snapshot", lambda: snapshot)
    _patch("_browser_context_block", lambda: block)


def _restore():
    for attr, value in _ORIG.items():
        setattr(orch, attr, value)
    _ORIG.clear()


_EMPTY_PLAN = lambda text, ctx="", browser_state=None, calculator_state=None: {  # noqa: E731
    "spoken_response": "ok",
    "actions": [],
    "response_language": "en",
}


# ---- _truthful_response -----------------------------------------------------

def test_truthful_response_all_ok_keeps_llm_text():
    out = orch._truthful_response(
        "Opening YouTube.",
        [{"type": "open_url", "ok": True, "value": "browser open"}],
        "en",
    )
    assert out == "Opening YouTube."


def test_truthful_response_all_failed_replaces_false_claim():
    out = orch._truthful_response(
        "Opening YouTube.",
        [{"type": "open_url", "ok": False, "error": "navigation timed out"}],
        "en",
    )
    assert "Opening YouTube." not in out
    assert "navigation timed out" in out
    assert "couldn't" in out


def test_truthful_response_partial_failure_reports_counts_and_detail():
    out = orch._truthful_response(
        "Opening things.",
        [
            {"type": "open_url", "ok": True, "value": "ok"},
            {"type": "browser_open_result", "ok": False, "error": "no search results available yet"},
        ],
        "en",
    )
    assert "1 steps" in out or "1" in out
    assert "could not be completed" in out
    assert "no search results available yet" in out


def test_truthful_response_uses_language_templates():
    en = orch._truthful_response("x", [{"type": "a", "ok": False, "error": "boom"}], "en")
    ur = orch._truthful_response("x", [{"type": "a", "ok": False, "error": "boom"}], "ur")
    hi = orch._truthful_response("x", [{"type": "a", "ok": False, "error": "boom"}], "hinglish")
    assert "couldn't" in en
    assert "نہیں کر سکا" in ur
    assert "nahi ho saka" in hi
    assert "boom" in en and "boom" in ur and "boom" in hi


def test_truthful_response_unknown_language_falls_back_to_english():
    out = orch._truthful_response("x", [{"type": "a", "ok": False, "error": "boom"}], "fr")
    assert "couldn't" in out


# ---- run_turn: no false-success speech --------------------------------------

def test_run_turn_never_plays_false_success_claim():
    _patch_browser({}, "")
    spoken = []

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        return {
            "spoken_response": "Opening YouTube.",
            "actions": [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}],
            "response_language": "en",
        }

    _patch("plan", _plan)
    _patch("route", lambda actions: [
        {"type": "open_url", "ok": False, "error": "navigation timed out"}
    ])
    _patch("_speak", lambda text: spoken.append(text))
    try:
        orch.reset()
        text = orch.run_turn("open youtube")
        assert text is not None
        assert "Opening YouTube." not in text
        assert "navigation timed out" in text
        assert len(spoken) == 1
        assert spoken[0] == text
    finally:
        _restore()


def test_run_turn_all_ok_speaks_llm_text_once():
    _patch_browser({}, "")
    spoken = []

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        return {
            "spoken_response": "Opening Chromebook thing.",
            "actions": [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}],
            "response_language": "en",
        }

    _patch("plan", _plan)
    _patch("route", lambda actions: [
        {"type": "open_url", "ok": True, "value": "browser open"}
    ])
    _patch("_speak", lambda text: spoken.append(text))
    try:
        orch.reset()
        text = orch.run_turn("open youtube")
        assert text == "Opening Chromebook thing."
        assert len(spoken) == 1
        assert spoken[0] == text
    finally:
        _restore()


# ---- run_turn: TTS failure is isolated --------------------------------------

def test_tts_failure_does_not_corrupt_turn():
    import audio.tts as tts_mod

    orig_tts_speak = tts_mod.speak
    tts_mod.speak = lambda text: (_ for _ in ()).throw(RuntimeError("audio device gone"))
    _patch("_browser_snapshot", lambda: {})
    _patch("_browser_context_block", lambda: "")
    _patch("plan", lambda text, ctx="", browser_state=None, calculator_state=None: {
        "spoken_response": "Opening Notepad.",
        "actions": [],
        "response_language": "en",
    })
    try:
        orch.reset()
        text = orch.run_turn("open notepad")
        assert text == "Opening Notepad."
        summary = orch._context.summary()
        assert "User: open notepad" in summary
        assert "Assistant: Opening Notepad." in summary
    finally:
        tts_mod.speak = orig_tts_speak
        _restore()


# ---- run_turn: browser state feeding the planner ----------------------------

def test_browser_state_block_appended_to_planner_context():
    snap = {
        "open": True,
        "url": "https://www.youtube.com",
        "title": "YouTube",
        "has_results": False,
        "result_count": 0,
        "last_search": None,
    }
    block = "browser is already open (reuse this session); current page: YouTube | https://www.youtube.com"
    _patch_browser(snap, block)
    captured = {}

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        captured["ctx"] = ctx
        captured["state"] = browser_state
        return {"spoken_response": "ok", "actions": [], "response_language": "en"}

    _patch("plan", _plan)
    _patch("_speak", lambda text: None)
    try:
        orch.reset()
        orch.run_turn("go back")
        assert "[Browser state]" in captured["ctx"]
        assert "browser is already open" in captured["ctx"]
        assert captured["state"] is snap
    finally:
        _restore()


def test_browser_state_block_absent_when_browser_closed():
    _patch_browser({}, "")
    captured = {}

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        captured["ctx"] = ctx
        return _EMPTY_PLAN(text, ctx, browser_state, calculator_state)

    _patch("plan", _plan)
    _patch("_speak", lambda text: None)
    try:
        orch.reset()
        orch.run_turn("open notepad")
        assert "[Browser state]" not in captured["ctx"]
    finally:
        _restore()


# ---- run_turn: context updated after execution ------------------------------

def test_context_updated_after_execution():
    _patch_browser({}, "")

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        return {
            "spoken_response": "Getting the time.",
            "actions": [{"type": "get_time", "params": {}}],
            "response_language": "en",
        }

    _patch("plan", _plan)
    _patch("_speak", lambda text: None)
    try:
        orch.reset()
        orch.run_turn("what time is it")
        summary = orch._context.summary()
        assert "Last action result:" in summary
        assert "get_time: done" in summary
    finally:
        _restore()


def test_multi_turn_conversation_context_stored():
    _patch_browser({}, "")
    _patch("plan", _EMPTY_PLAN)
    _patch("_speak", lambda text: None)
    try:
        orch.reset()
        orch.run_turn("hello")
        orch.run_turn("what time is it")
        orch.run_turn("thanks")
        s = orch._context.summary()
        assert "User: hello" in s
        assert "Assistant: ok" in s
        assert "User: what time is it" in s
        assert "User: thanks" in s
    finally:
        _restore()


# ---- run_turn: calculator state feeding the planner -------------------------

def test_calculator_state_block_appended_to_planner_context():
    snap = {"expression": "25*13", "value": 325, "opened": True}
    block = "last calculation 25*13 = 325; calculator is open"
    _patch("_calculator_snapshot", lambda: snap)
    _patch("_calculator_context_block", lambda: block)
    captured = {}

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        captured["ctx"] = ctx
        captured["state"] = calculator_state
        return {"spoken_response": "ok", "actions": [], "response_language": "en"}

    _patch("plan", _plan)
    _patch("_speak", lambda text: None)
    try:
        orch.reset()
        orch.run_turn("do it there")
        assert "[Calculator state]" in captured["ctx"]
        assert "last calculation 25*13 = 325" in captured["ctx"]
        assert captured["state"] is snap
    finally:
        _restore()


def test_calculator_state_block_absent_when_calculator_idle():
    _patch_browser({}, "")
    _patch("_calculator_snapshot", lambda: {"expression": None, "value": None, "opened": False})
    _patch("_calculator_context_block", lambda: "")
    captured = {}

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        captured["ctx"] = ctx
        return _EMPTY_PLAN(text, ctx, browser_state, calculator_state)

    _patch("plan", _plan)
    _patch("_speak", lambda text: None)
    try:
        orch.reset()
        orch.run_turn("open notepad")
        assert "[Calculator state]" not in captured["ctx"]
        assert "[Browser state]" not in captured["ctx"]
    finally:
        _restore()


# ---- regression: a previous turn's failure never contaminates the next ----

def test_previous_failure_never_leaks_into_empty_plan_turn():
    """A failed calculator step in turn 1 must NOT show up again in turn 2."""
    _patch_browser({}, "")
    spoken = []
    turn = [0]

    def _plan(text, ctx="", browser_state=None, calculator_state=None):
        if turn[0] == 0:
            turn[0] += 1
            return {
                "spoken_response": "Done - 25*13 is 325.",
                "actions": [{"type": "calculator_perform", "params": {"expression": "25*13"}}],
                "response_language": "en",
            }
        return {"spoken_response": "ok", "actions": [], "response_language": "en"}

    def _route(actions):
        if actions:
            return [{"type": "calculator_perform", "ok": False, "error": "calculator not available"}]
        return []

    _patch("plan", _plan)
    _patch("route", _route)
    _patch("_speak", lambda text: spoken.append(text))
    try:
        orch.reset()
        first = orch.run_turn("25 into 13 on the calculator")
        second = orch.run_turn("calculator open karo")
        assert "calculator not available" in first
        assert "calculator not available" not in second
        assert second == "ok"
    finally:
        _restore()


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("orchestrator: all tests passed")


if __name__ == "__main__":
    _main()