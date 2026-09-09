"""Safe calculator interaction tools (no arbitrary execution).

Two registered tools share one tiny, allowlisted arithmetic core:

  - ``calculate``            -> compute the answer and return it
  - ``calculator_perform``   -> type the expression into the real Windows
                                Calculator app and press "="

The LLM can never execute anything through these tools:
  * expressions are validated against a character whitelist (digits and
    ``+ - * / ( ) .``) and evaluated by an AST-walking evaluator, never eval();
  * the Windows Calculator keystrokes are sent with ctypes ``SendInput``
    (UNICODE keys) to a real, allowlisted app launched exactly like the
    applications tool launches calc.exe - no shell, no generated commands.

The module also keeps a small snapshot (last expression + value + whether the
app was opened) that the orchestrator feeds back to the planner so deictic
follow-ups ("do it there", "show that on the calculator") stay in context.
"""

from __future__ import annotations

import ast
import ctypes
import operator
import re
import subprocess
import time

from ctypes import wintypes

from tools.registry import tool

# ---------------------------------------------------------------------------
# Safe arithmetic core
# ---------------------------------------------------------------------------

_CHARS_RE = re.compile(r"^[0-9+\-*/().\s]+$")
_NUM_RE = r"\d+(?:\.\d+)?"

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_OP_WORD_RE = re.compile(
    r"(multiplied by|divided by|times|into|divide|multiply|plus|minus|subtract|add|guna)",
    re.I,
)
_URDU_NUM_OP_RE = re.compile(
    rf"({_NUM_RE})\s+ko\s+({_NUM_RE})\s+se\s+(multiply|guna|divide|plus|minus|add|subtract|times)",
    re.I,
)
_VERB_FIRST_RE = re.compile(
    rf"\b(multiply|divide|add|subtract)\s+({_NUM_RE})\s+(?:by\s+)?({_NUM_RE})\b",
    re.I,
)
_ADD_AND_RE = re.compile(rf"\b(add|plus)\s+({_NUM_RE})\s+and\s+({_NUM_RE})", re.I)
_SUB_FROM_RE = re.compile(
    rf"\b(subtract|minus|take\s+away)\s+({_NUM_RE})\s+from\s+({_NUM_RE})",
    re.I,
)
_SYMBOL_RE = re.compile(rf"({_NUM_RE})\s*([*/x×÷+\-−])\s*({_NUM_RE})")

_OP_SYMBOLS = {
    "*": "*", "/": "/", "+": "+", "-": "-", "−": "-",
    "x": "*", "X": "*", "×": "*", "÷": "/",
}
_WORD_OPS = {
    "multiplied by": "*", "times": "*", "into": "*", "multiply": "*", "guna": "*",
    "divided by": "/", "divide": "/",
    "plus": "+", "add": "+",
    "minus": "-", "subtract": "-",
}


def _safe_eval(expr: str) -> int | float:
    """Evaluate a *whitelisted* arithmetic expression. Never eval()."""
    e = (expr or "").strip()
    if not e or not _CHARS_RE.match(e):
        raise ValueError("invalid expression")
    try:
        tree = ast.parse(e, mode="eval")
    except (SyntaxError, ValueError):
        raise ValueError("invalid expression")

    def node_value(n):
        if isinstance(n, ast.Expression):
            return node_value(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("invalid expression")
        if isinstance(n, ast.BinOp):
            left = node_value(n.left)
            right = node_value(n.right)
            op = _BINOPS.get(type(n.op))
            if op is None:
                raise ValueError("invalid operator")
            if isinstance(n.op, ast.Div) and right == 0:
                raise ValueError("division by zero")
            return op(left, right)
        if isinstance(n, ast.UnaryOp):
            operand = node_value(n.operand)
            if isinstance(n.op, ast.UAdd):
                return operand
            if isinstance(n.op, ast.USub):
                return -operand
            raise ValueError("invalid operator")
        raise ValueError("invalid expression")

    val = node_value(tree)
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def _fmt(value) -> str:
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def _arithmetic_from_text(text: str) -> str | None:
    """Extract one canonical binary expression ("25*13") from spoken text."""
    t = (text or "").strip()
    if not t:
        return None
    m = _SYMBOL_RE.search(t)
    if m:
        return f"{m.group(1)}{_OP_SYMBOLS[m.group(2)]}{m.group(3)}"
    m = _URDU_NUM_OP_RE.search(t)
    if m:
        op = _WORD_OPS.get(m.group(3).lower())
        if op is None:
            return None
        return f"{m.group(1)}{op}{m.group(2)}"
    m = _ADD_AND_RE.search(t)
    if m:
        return f"{m.group(2)}+{m.group(3)}"
    m = _SUB_FROM_RE.search(t)
    if m:
        return f"{m.group(3)}-{m.group(2)}"
    m = _VERB_FIRST_RE.search(t)
    if m:
        op = _WORD_OPS.get(m.group(1).lower())
        if op is None:
            return None
        return f"{m.group(2)}{op}{m.group(3)}"
    m = _OP_WORD_RE.search(t)
    if m:
        word = m.group(1).lower()
        op = _WORD_OPS.get(word)
        if op is None:
            return None
        before = re.search(rf"({_NUM_RE})\s*$", t[: m.start()])
        after = re.search(rf"({_NUM_RE})", t[m.end():])
        if before and after:
            return f"{before.group(1)}{op}{after.group(1)}"
    return None


# ---------------------------------------------------------------------------
# "is this a calculator-frame / a deictic reference to the last calculation?"
# ---------------------------------------------------------------------------

_DEICTIC_RE = re.compile(
    r"\b(that|this|it|there|same|ye|yeh|yhi|isse|usko|uss|woh|wahan|wahin|pehla wala|pehle wala|pehli wali)\b",
    re.I,
)
_DEICTIC_CALC_SHORT = frozenset({
    "do it there", "do that there", "do it", "do that",
    "it there", "that there",
    "calculate it there", "carry it there",
    "karo wahan", "karo wahin", "wahan karo", "wahin karo",
    "isse karo", "usko karo", "ye kar do",
})


def _calc_frame(text: str) -> bool:
    """True when the utterance names the calculator app or a "show this CALC"
    construction ("karke dikhao") - vs. a plain math question to answer."""
    low = (text or "").lower()
    if re.search(r"\b(calculator|calc)\b", low):
        return True
    if re.search(r"\b(?:hisab|hisaab)\b", low):
        return True
    if re.search(r"kar\s*(?:ke|kar)\s*dikh(?:ao|a|e)?", low):
        return True
    return False


def _calc_deictic_expr(text: str, state: dict | None) -> str | None:
    """Return the previous expression when the utterance points back at it."""
    if not state or not state.get("expression"):
        return None
    low = " ".join((text or "").strip().lower().split()).rstrip(".,!?;")
    if not low:
        return None
    if low in _DEICTIC_CALC_SHORT:
        return state["expression"]
    if _DEICTIC_RE.search(low) and _calc_frame(low):
        return state["expression"]
    return None


# ---------------------------------------------------------------------------
# Context snapshot (fed to the planner like the browser snapshot)
# ---------------------------------------------------------------------------

_STATE = {"expression": None, "value": None, "opened": False}


def calculator_snapshot() -> dict:
    """Fresh copy of the calculator context for the planner (pure read)."""
    return dict(_STATE)


def calculator_state_text() -> str:
    """Human-readable one-liner appended to the planner context block."""
    s = _STATE
    if s["expression"] is None:
        return ""
    parts = [f"last calculation {s['expression']} = {s['value']}"]
    if s["opened"]:
        parts.append("calculator is open")
    return "; ".join(parts)


def reset_calculator() -> None:
    _STATE.update(expression=None, value=None, opened=False)


def mark_calculator_opened() -> None:
    _STATE["opened"] = True


# ---------------------------------------------------------------------------
# Windows Calculator interaction (ctypes SendInput, no dependencies)
# ---------------------------------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002

HWND = ctypes.c_void_p
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.FindWindowW.restype = HWND
_user32.SetForegroundWindow.argtypes = [HWND]
_user32.SetForegroundWindow.restype = wintypes.BOOL


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _send_keys(text: str) -> None:
    """Send Unicode keystrokes to the foreground window, chunked."""
    CHUNK = 25
    if not text:
        return
    for i in range(0, len(text), CHUNK):
        chunk = text[i: i + CHUNK]
        n = len(chunk)
        arr = (_INPUT * n)()
        for j, ch in enumerate(chunk):
            inp = _INPUT()
            inp.type = INPUT_KEYBOARD
            inp.ki = _KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE, 0, 0)
            arr[j] = inp
        _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        for j, ch in enumerate(chunk):
            inp = _INPUT()
            inp.type = INPUT_KEYBOARD
            inp.ki = _KEYBDINPUT(0, ord(ch), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0)
            arr[j] = inp
        _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))
        time.sleep(0.03)


def _ensure_calc_window() -> None:
    """Attach to the running Calculator or start the allowlisted calc.exe."""
    hwnd = _user32.FindWindowW(None, "Calculator")
    if not hwnd:
        subprocess.Popen(["calc.exe"], close_fds=True)
        for _ in range(60):
            time.sleep(0.1)
            hwnd = _user32.FindWindowW(None, "Calculator")
            if hwnd:
                break
    if not hwnd:
        raise ValueError("could not open the Calculator app")
    _user32.SetForegroundWindow(hwnd)
    time.sleep(0.25)


# ---------------------------------------------------------------------------
# Registered tools
# ---------------------------------------------------------------------------

@tool(
    name="calculate",
    description=(
        "Compute a simple arithmetic expression and return the numeric answer "
        "(e.g. expression '25*13' -> '325'). Used for plain math questions such "
        "as 'what is 25 times 13'."
    ),
    params={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '25*13'.",
            },
        },
        "required": ["expression"],
    },
    safety="auto",
)
def calculate(expression: str) -> str:
    """Compute the answer (answer-mode, no calculator UI involved)."""
    expr = _arithmetic_from_text(expression) or expression
    value = _safe_eval(expr)
    _STATE.update(expression=expr, value=value)
    return _fmt(value)


@tool(
    name="calculator_perform",
    description=(
        "Type a simple arithmetic expression into the real Windows Calculator "
        "app and press 'equal' (e.g. '25*13'). Use ONLY when the user asks to "
        "do / show the calculation ON the calculator app (calculator pe karo)."
    ),
    params={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression to enter, e.g. '25*13'.",
            },
        },
        "required": ["expression"],
    },
    safety="auto",
)
def calculator_perform(expression: str) -> str:
    """Execute the expression in the Windows Calculator app (interaction-mode)."""
    expr = _arithmetic_from_text(expression) or expression
    _safe_eval(expr)  # validate before touching the app
    _ensure_calc_window()
    _send_keys(expr + "=")
    value = _safe_eval(expr)
    _STATE.update(expression=expr, value=value, opened=True)
    return _fmt(value)