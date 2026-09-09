"""Offline tests for the safe calculator tools.

No real Calculator window is touched: ``calculator_perform`` tests stub out
``_ensure_calc_window`` / ``_send_keys`` so we verify the tool contract, the
keystroke payload, and the context snapshot purely in memory.

Locks in:
  - a whitelist-only safe evaluator (never eval), incl. division-by-zero and
    malicious-input rejection;
  - speech-to-expression extraction for English, verb-first, symbol, and
    Urdu/Hinglish phrasing;
  - calculator-frame vs plain-math distinction and deictic back-reference;
  - the router accepting ``calculate`` and ``calculator_perform`` with validated
    required params.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.calculator as calc  # noqa: E402
from assistant.router import route  # noqa: E402


# ---- _safe_eval -----------------------------------------------------------------

def test_safe_eval_basic_ops():
    assert calc._safe_eval("25*13") == 325
    assert calc._safe_eval("7+8") == 15
    assert calc._safe_eval("20-5") == 15
    assert calc._safe_eval("100/4") == 25
    assert calc._safe_eval("2.5*4") == 10


def test_safe_eval_precedence_and_parens():
    assert calc._safe_eval("2+3*4") == 14
    assert calc._safe_eval("(2+3)*4") == 20
    assert calc._safe_eval("-5+3") == -2
    assert calc._safe_eval("2*(3+4)-1") == 13


def test_safe_eval_rejects_division_by_zero():
    try:
        calc._safe_eval("5/0")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for division by zero")


def test_safe_eval_rejects_non_arithmetic_input():
    for bad in ("__import__('os')", "os.system('x')", "25;1", "[1,2]", "lambda: 1", "9**9**9", "0b1", "1e3*2"):
        try:
            calc._safe_eval(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected rejection of {bad!r}")


def test_safe_eval_rejects_empty_and_whitespace():
    for bad in ("", "   ", None):
        try:
            calc._safe_eval(bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("expected rejection")


def test_safe_eval_integral_converts_to_int():
    assert isinstance(calc._safe_eval("100/4"), int)
    assert isinstance(calc._safe_eval("2.5+2.5"), int)
    assert isinstance(calc._safe_eval("2.5*4"), int)


# ---- _arithmetic_from_text -------------------------------------------------------

def test_arithmetic_english_infix():
    assert calc._arithmetic_from_text("what is 25 times 13") == "25*13"
    assert calc._arithmetic_from_text("what is 25 plus 13") == "25+13"
    assert calc._arithmetic_from_text("25 minus 5") == "25-5"
    assert calc._arithmetic_from_text("25 divided by 5") == "25/5"
    assert calc._arithmetic_from_text("2 into 3") == "2*3"


def test_arithmetic_verb_first():
    assert calc._arithmetic_from_text("multiply 25 by 13") == "25*13"
    assert calc._arithmetic_from_text("divide 100 by 4") == "100/4"
    assert calc._arithmetic_from_text("add 5 and 7") == "5+7"
    assert calc._arithmetic_from_text("subtract 3 from 10") == "10-3"


def test_arithmetic_symbols():
    assert calc._arithmetic_from_text("25*13") == "25*13"
    assert calc._arithmetic_from_text("25 x 13") == "25*13"
    assert calc._arithmetic_from_text("25 × 13") == "25*13"
    assert calc._arithmetic_from_text("100 ÷ 4") == "100/4"


def test_arithmetic_urdu_hinglish():
    assert calc._arithmetic_from_text("25 ko 13 se multiply karo") == "25*13"
    assert calc._arithmetic_from_text("100 ko 4 se divide karo") == "100/4"
    assert calc._arithmetic_from_text("7 ko 6 se guna karo") == "7*6"


def test_arithmetic_returns_none_when_absent():
    assert calc._arithmetic_from_text("hello there") is None
    assert calc._arithmetic_from_text("what time is it") is None
    assert calc._arithmetic_from_text("") is None
    assert calc._arithmetic_from_text("open notepad") is None


# ---- frame vs deictic -------------------------------------------------------------

def test_calc_frame_detection():
    assert calc._calc_frame("calculator kholo")
    assert calc._calc_frame("open the calculator")
    assert calc._calc_frame("karke dikhao")  # "show it on the calc" construction
    assert calc._calc_frame("25 multiplied by 13") is False  # plain math


def test_calc_deictic_uses_previous_expression_only_with_ctx():
    state = {"expression": "25*13", "value": 325, "opened": True}
    assert calc._calc_deictic_expr("do it there", state) == "25*13"
    assert calc._calc_deictic_expr("show that on the calculator", state) == "25*13"
    assert calc._calc_deictic_expr("do it there", None) is None
    assert calc._calc_deictic_expr("open notepad", state) is None


# ---- snapshot / context -------------------------------------------------------------

def test_snapshot_and_state_text_roundtrip():
    calc.reset_calculator()
    assert calc.calculator_snapshot() == {
        "expression": None, "value": None, "opened": False,
    }
    assert calc.calculator_state_text() == ""
    calc._STATE.update(expression="25*13", value=325, opened=True)
    snap = calc.calculator_snapshot()
    assert snap["expression"] == "25*13"
    assert snap["value"] == 325
    text = calc.calculator_state_text()
    assert "25*13 = 325" in text
    assert "calculator is open" in text
    calc.reset_calculator()
    assert calc.calculator_snapshot()["expression"] is None


# ---- route-level: calculate -----------------------------------------------------------

def test_route_calculate_returns_answer():
    out = route([{"type": "calculate", "params": {"expression": "6*7"}}])
    assert out == [{"type": "calculate", "ok": True, "value": "42"}]


def test_route_calculate_missing_param_rejected_before_execution():
    out = route([{"type": "calculate", "params": {}}])
    assert out[0]["ok"] is False
    assert "missing required param" in out[0]["error"]


def test_route_calculate_invalid_expression_reports_failure_truthfully():
    out = route([{"type": "calculate", "params": {"expression": "5/0"}}])
    assert out[0]["ok"] is False
    assert "division by zero" in out[0]["error"]


# ---- route-level: calculator_perform (window + keystrokes stubbed) --------------------

def test_route_calculator_perform_sends_expression_and_equal():
    calc.reset_calculator()
    sent = []
    calc._ensure_calc_window = lambda: None
    calc._send_keys = lambda s: sent.append(s)
    try:
        out = route([{"type": "calculator_perform", "params": {"expression": "25*13"}}])
        assert out[0]["ok"] is True
        assert out[0]["value"] == "325"
        assert sent == ["25*13="]
        snap = calc.calculator_snapshot()
        assert snap["expression"] == "25*13"
        assert snap["value"] == 325
        assert snap["opened"] is True
    finally:
        calc.reset_calculator()


def test_route_calculator_perform_missing_param_rejected():
    calc._ensure_calc_window = lambda: None
    calc._send_keys = lambda s: None
    out = route([{"type": "calculator_perform", "params": {}}])
    assert out[0]["ok"] is False
    assert "missing required param" in out[0]["error"]


def test_route_calculator_perform_validates_before_touching_app():
    # A malicious expression must be rejected BEFORE the window interaction.
    touched = []
    calc._send_keys = lambda s: touched.append(s)
    try:
        out = route([{"type": "calculator_perform", "params": {"expression": "os.system('x')"}}])
        assert out[0]["ok"] is False
        assert touched == []  # nothing was ever sent to the app
        assert calc.calculator_snapshot()["expression"] is None
    finally:
        calc.reset_calculator()


def test_route_calculator_perform_surfaces_app_failure_truthfully():
    calc._ensure_calc_window = lambda: (_ for _ in ()).throw(ValueError("could not open the Calculator app"))
    out = route([{"type": "calculator_perform", "params": {"expression": "25*13"}}])
    assert out[0]["ok"] is False
    assert "could not open the Calculator app" in out[0]["error"]


def test_open_application_calculator_marks_opened_without_error():
    # Regression: open_application referenced an undefined `key` when marking the
    # calculator as opened (NameError on the REAL app-open path — the window
    # interaction tests never execute it). Exercise the real handler with the
    # launch bits patched so it stays hermetic.
    import tools.applications as apps

    orig = (apps.subprocess.Popen, apps.shutil.which, apps.os.path.isfile)
    apps.subprocess.Popen = lambda *a, **k: None
    apps.shutil.which = lambda _n: r"C:\Windows\System32\calc.exe"
    apps.os.path.isfile = lambda _p: True
    try:
        calc.reset_calculator()
        out = route([{"type": "open_application", "params": {"target": "calculator"}}])
        assert out[0]["ok"] is True, out
        assert calc.calculator_snapshot()["opened"] is True
    finally:
        apps.subprocess.Popen, apps.shutil.which, apps.os.path.isfile = orig
        calc.reset_calculator()


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("calculator: all tests passed")


if __name__ == "__main__":
    _main()