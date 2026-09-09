"""Windows application launch/close tools.

Uses lightweight subprocess / os.startfile to open installed applications.
The registry maps friendly names to executables; Chrome is special-cased to
drive the assistant's Playwright browser so that "Open Chrome" is followed by
search/navigate actions in the SAME session.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from tools.registry import tool

# Friendly name -> executable (or None to signal a special-case handler).
# Keys are lowercase aliases the planner may send in `target`.
_APP_MAP = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "this pc": "explorer.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
}

# Extensions installed on this machine (detected at runtime).
_EXTRA_APPS: dict[str, str] = {}


def _detect_known_extensions() -> None:
    """Find common apps not in System32 (VS Code, etc.)."""
    candidates = {
        "vscode": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            r"C:\Program Files\Microsoft VS Code\Code.exe",
        ],
    }
    for name, paths in candidates.items():
        for p in paths:
            if os.path.isfile(p):
                _EXTRA_APPS[name] = p
                break


_detected = False


def _resolve_target(target: str) -> tuple[str, str | None]:
    """Return (kind, payload): ('exe', path) or ('special', handler)."""
    global _detected
    if not _detected:
        _detect_known_extensions()
        _detected = True

    key = (target or "").strip().lower()
    if key in ("chrome", "google chrome", "google-chrome", "chrome browser", "browser"):
        return ("special", "chrome")
    if key in _APP_MAP:
        return ("exe", _APP_MAP[key])
    if key in _EXTRA_APPS:
        return ("exe", _EXTRA_APPS[key])
    # Last resort: treat the target as a bare executable name.
    if key and not any(c in key for c in " /\\"):
        exe = shutil.which(key + ".exe") or shutil.which(key)
        if exe:
            return ("exe", exe)
    return ("exe", None)


@tool(
    name="open_application",
    description=(
        "Open a DESKTOP application (a program installed on this PC). targets "
        "include: notepad, calculator, paint, file explorer, cmd, powershell, "
        "task manager, vscode, chrome. Websites are NOT applications — for "
        "YouTube/Gmail/etc. use open_url instead."
    ),
    params={
        "type": "object",
        "properties": {"target": {"type": "string", "description": "application name"}},
        "required": ["target"],
    },
)
def open_application(target: str) -> str:
    kind, payload = _resolve_target(target)
    key = (target or "").strip().lower()
    if kind == "special" and payload == "chrome":
        # Start (or focus) the assistant's Playwright browser session.
        from tools import browser  # lazy import avoids circular dependency

        browser.ensure_browser()
        return "opened browser"
    if payload is None:
        raise ValueError(f"unknown application: {target}")

    exe_path = shutil.which(payload)
    if exe_path is None:
        # Fall back to direct path for System32 apps not on PATH.
        win_dir = os.environ.get("SystemRoot", r"C:\Windows")
        exe_path = os.path.join(win_dir, "System32", payload)
    if not os.path.isfile(exe_path):
        raise ValueError(f"could not find application: {target}")

    subprocess.Popen([exe_path], close_fds=True)
    if _APP_MAP.get(key) == "calc.exe":
        try:
            from tools import calculator  # lazy import avoids circular dependency
            calculator.mark_calculator_opened()
        except Exception:
            pass
    return f"opened {target}"


@tool(
    name="close_application",
    description="Close a desktop application by name (notepad, calculator, paint, etc.).",
    params={
        "type": "object",
        "properties": {"target": {"type": "string", "description": "application name"}},
        "required": ["target"],
    },
)
def close_application(target: str) -> str:
    key = (target or "").strip().lower()
    if key in ("chrome", "google chrome", "google-chrome", "chrome browser", "browser"):
        from tools import browser  # lazy import avoids circular dependency

        browser.close_browser()
        return "closed browser"
    if key in ("calculator", "calc"):
        from tools import calculator as calc_tools
        calc_tools.reset_calculator()

    exe = _APP_MAP.get(key)

    exe = _APP_MAP.get(key)
    if exe is None:
        # Try to resolve a bare name to an exe.
        resolved = shutil.which(key + ".exe")
        exe = resolved if resolved else None
    if exe is None:
        raise ValueError(f"unknown application: {target}")

    image = os.path.basename(exe)
    # Graceful close via taskkill (no /F) so the app can save state.
    subprocess.run(["taskkill", "/IM", image], capture_output=True, timeout=10)
    return f"closed {target}"
