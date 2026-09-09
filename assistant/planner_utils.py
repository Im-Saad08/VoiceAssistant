"""Shared planner utilities: schema, coercion, fallback, system prompt.

Extracted from planner.py so both the Gemini and Ollama backends can share
the same response schema, action normalization, and fallback logic without
circular imports.
"""

from __future__ import annotations

import json
import re

from google.genai import types

from assistant.logging_setup import get_logger
from tools.applications import _APP_MAP
from tools.calculator import (
    _arithmetic_from_text,
    _calc_deictic_expr,
    _calc_frame,
    _fmt,
    _safe_eval,
)
from tools.registry import all_tools, tool_catalog

logger = get_logger("planner_utils")


# ---------------------------------------------------------------------------
# Response schema (Gemini structured output)
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
    "object": types.Type.OBJECT,
    "array": types.Type.ARRAY,
}


def _json_schema_to_types(js: dict) -> types.Schema:
    """Convert a registry JSON-schema dict into a genai types.Schema."""
    stype = _TYPE_MAP.get(js.get("type") or "object", types.Type.STRING)
    schema = types.Schema(
        type=stype,
        description=js.get("description"),
        enum=js.get("enum"),
    )
    if stype == types.Type.OBJECT and "properties" in js:
        schema.properties = {
            k: _json_schema_to_types(v) for k, v in js["properties"].items()
        }
        schema.required = list(js.get("required", []))
    return schema


def _build_response_schema() -> types.Schema:
    """Build the any_of union response schema from the tool registry."""
    branches = []
    for t in all_tools():
        branches.append(
            types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "type": types.Schema(type=types.Type.STRING, enum=[t.name]),
                    "params": _json_schema_to_types(t.params or {}),
                    "narration": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Optional short progress update to speak BEFORE "
                            "executing this action. Keep it brief and natural."
                        ),
                    ),
                },
                required=["type", "params"],
            )
        )

    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "response_language": types.Schema(
                type=types.Type.STRING,
                description="ISO code of the language for the spoken response (en, ur, hinglish, etc.).",
            ),
            "spoken_response": types.Schema(
                type=types.Type.STRING,
                description=(
                    "A short, natural sentence the assistant should speak aloud. "
                    "Sound like a capable computer assistant — not robotic, not overly verbose."
                ),
            ),
            "actions": types.Schema(
                type=types.Type.ARRAY,
                description="Ordered list of actions to perform (may be empty).",
                items=types.Schema(type=types.Type.OBJECT, any_of=branches),
            ),
        },
        required=["spoken_response", "actions"],
    )


# ---------------------------------------------------------------------------
# System prompt (shared by both backends)
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    catalog = json.dumps(tool_catalog(), indent=1)
    return (
        "You are a desktop voice assistant (like JARVIS). "
        "You understand natural language in English, Urdu, and Hinglish. "
        "When the user speaks Urdu script, respond in Urdu (ur). "
        "When the user speaks Hinglish/Urdu-English mix, respond in hinglish. "
        "When the user speaks English, respond in English (en).\n\n"
        "You have the following tools available. You may ONLY use these tools:\n\n"
        f"{catalog}\n\n"
        "RULES:\n"
        "- Only emit actions using tool names from the list above.\n"
        "- EVERY action MUST include a complete 'params' object with ALL "
        "required fields shown for that tool (e.g. open_application requires "
        "'target'; browser_search requires 'query'; browser_open_result "
        "requires 'index'). Do NOT leave params empty.\n"
        "- Match the user's conversational language in spoken_response.\n"
        "- Keep spoken_response short and natural (1-2 sentences max).\n"
        "- For multi-step tasks, add a brief narration to each action "
        "describing what you're about to do.\n"
        "- If the user's request is ambiguous, make a reasonable assumption "
        "and proceed.\n"
        "- If you truly cannot determine what to do, leave actions empty and "
        "explain why.\n"
        "- For destructive actions (delete, shutdown), still include the "
        "action; the safety layer handles confirmation.\n"
        "- Websites vs desktop apps: YouTube, Gmail, Google, GitHub, and any "
        "other website/online service are opened with open_url (their URL). "
        "open_application is ONLY for desktop programs (notepad, calculator, "
        "paint, file explorer, cmd, powershell, task manager, vscode, chrome).\n"
        "- Math: 'what is 25 times 13' -> calculate with the expression "
        "(25*13). Only when the user wants it ON the calculator app "
        "(calculator pe / on the calculator) -> calculator_perform.\n"
        "- Browser-relative commands:\n"
        "    'the first result'/'that result' -> browser_open_result index 1\n"
        "    'the second result' -> browser_open_result index 2\n"
        "    'go back'/'back' -> browser_go_back\n"
        "    'go to <site>' / 'open <site>' -> open_url\n"
        "- If [Browser state] says a browser is already open, REUSE it — do not "
        "emit 'Open Chrome'; navigate in the open session.\n"
        "- Be truthful in spoken_response: if you emitted actions, describe "
        "what the actions are DOING (\"Opening YouTube...\"), never claim an "
        "action already succeeded before it has run.\n"
        "- Return ONLY valid JSON. No markdown, no explanations outside JSON.\n"
    )


# ---------------------------------------------------------------------------
# Compact catalog (for small local models)
# ---------------------------------------------------------------------------

def _compact_catalog() -> str:
    """One-line-per-tool tool summary for small local LLMs (e.g. qwen2.5:1.5b).

    Small models get lost in a full JSON-schema dump (1000+ tokens) and start
    echoing its shape back instead of the requested plan structure.  This flat
    list keeps the prompt short and makes the tool choice + required params
    unmistakable.
    """
    lines = []
    for t in sorted(all_tools(), key=lambda t: t.name):
        params = t.params or {}
        required = params.get("required", [])
        if not required:
            lines.append(f"- {t.name}: {t.description} (no params)")
        else:
            lines.append(
                f"- {t.name}: {t.description}  REQUIRED params: {', '.join(required)}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action normalization
# ---------------------------------------------------------------------------

def _coerce_actions(raw: object) -> list[dict]:
    """Defensively normalize the LLM's ``actions`` field.

    The model may return a dict, a JSON string, or a list whose items are
    missing a ``params`` object.  Coerce everything to a list of well-formed
    action dicts so the router always validates a consistent shape.  Missing
    required params are deliberately NOT guessed — the router rejects them.
    """
    if not isinstance(raw, list):
        if raw is not None:
            logger.warning(
                "planner: 'actions' was %s, resetting to []",
                type(raw).__name__,
            )
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            logger.warning(
                "planner: dropping malformed action item (%s)",
                type(item).__name__,
            )
            continue
        item = dict(item)
        if not isinstance(item.get("params"), dict):
            item["params"] = {}
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Deterministic website / browser routing (shared by BOTH planner backends)
# ---------------------------------------------------------------------------
# qwen2.5:1.5b (and, occasionally, Gemini) blurs websites with desktop apps and
# leaves browser-relative commands under-specified. Rather than weaken the
# router's validation, we apply a small, exact, allowlisted correction HERE —
# the planner layer — so the router still validates every resulting action:

_WEBSITE_URLS: dict[str, str] = {
    "amazon": "https://www.amazon.com",
    "facebook": "https://www.facebook.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "instagram": "https://www.instagram.com",
    "linkedin": "https://www.linkedin.com",
    "netflix": "https://www.netflix.com",
    "reddit": "https://www.reddit.com",
    "twitter": "https://x.com",
    "whatsapp": "https://web.whatsapp.com",
    "wikipedia": "https://en.wikipedia.org/wiki/Main_Page",
    "yahoo": "https://www.yahoo.com",
    "youtube": "https://www.youtube.com",
}


def _website_url(value: str) -> str | None:
    """Known URL for a website name/domain, else None (not a website we know).

    Matches exact site names ("gmail"), the mapped domain ("mail.google.com"),
    and bare <name>.<tld> variants ("gmail.com"). Returns None for anything
    else, so desktop-app targets like "notepad" or "chrome" never map.
    """
    s = (value or "").strip().lower().rstrip("/")
    if not s:
        return None
    if "://" in s:
        s = s.split("://", 1)[1]
    if s.startswith("www."):
        s = s[4:]
    host = s.split("/", 1)[0].split(":", 1)[0]

    for name, url in _WEBSITE_URLS.items():
        domain = url.split("://", 1)[1]
        if domain.startswith("www."):
            domain = domain[4:]
        domain = domain.split("/", 1)[0]
        if host == name or host == domain:
            return url
        for tld in (".com", ".org", ".net", ".io", ".co"):
            if host == name + tld:
                return url
    return None


def _route_actions(actions: list[dict], browser_state: dict | None = None) -> list[dict]:
    """Deterministic, allowlisted routing corrections for any backend's plan.

    - open_application on a known website target  -> open_url (mapped URL)
    - open_url with a bare site name              -> open_url (full URL)
    - browser_open_result missing an index, when the browser holds results
        -> index 1 (the unambiguous default behind "that result")

    Exact-match only. Unknown targets and explicit-but-wrong values are left
    for the router to reject — never guessed here. Desktop apps (notepad,
    chrome, calc, ...) are never in this table and pass through untouched.
    """
    has_results = bool(browser_state and browser_state.get("has_results"))
    out: list[dict] = []
    for action in actions:
        action = dict(action)
        params = action.get("params") or {}
        if action.get("type") == "open_application":
            target = params.get("target")
            if isinstance(target, str):
                url = _website_url(target)
                if url:
                    out.append({"type": "open_url", "params": {"url": url}})
                    continue
        elif action.get("type") == "open_url":
            url = params.get("url")
            if isinstance(url, str) and "." not in url and "://" not in url:
                mapped = _website_url(url)
                if mapped:
                    action["params"] = {"url": mapped}
        elif action.get("type") == "browser_open_result":
            if has_results and params.get("index") in (None, ""):
                action["params"] = dict(params)
                action["params"]["index"] = 1
        out.append(action)
    return out


# Backwards-compatible alias used by tests and planner_ollama.plan().
def _normalize_website_actions(actions: list[dict], browser_state: dict | None = None) -> list[dict]:
    return _route_actions(actions, browser_state)


# ---------------------------------------------------------------------------
# Deterministic intent resolution (open-target -> calculator -> result index)
# ---------------------------------------------------------------------------
# Wherever the intent is *obvious* from the words, decide it deterministically
# instead of trusting a small model to pick the tool. The LLM stays the fallback
# for genuinely ambiguous phrasing. All corrections are exact-match; unknown
# targets are left for the router to reject, never guessed here.

_APP_TARGET_ALIASES = {
    "google-chrome": "chrome",
    "google chrome": "chrome",
    "chrome browser": "chrome",
    "windows chrome": "chrome",
    "browser": "chrome",
    "web browser": "chrome",
    "notepad.exe": "notepad",
    "calc.exe": "calculator",
    "calc": "calculator",
    "windows calculator": "calculator",
    "calculator.exe": "calculator",
    "ms-paint": "paint",
}
_DESKTOP_APPS = set(_APP_MAP) | {"chrome"}


def _canonical_app_target(target: str) -> str:
    low = (target or "").strip().lower()
    return _APP_TARGET_ALIASES.get(low, low)


def _normalize_app_targets(actions: list[dict]) -> list[dict]:
    """Canonicalize open_application targets BEFORE execution (no prompt-only)."""
    out: list[dict] = []
    for action in actions:
        if isinstance(action, dict) and action.get("type") == "open_application":
            p = action.get("params")
            if isinstance(p, dict) and isinstance(p.get("target"), str):
                p = dict(p)
                p["target"] = _canonical_app_target(p["target"])
                action = dict(action)
                action["params"] = p
        out.append(action)
    return out


# ---- open-target extraction --------------------------------------------------

_WEB_PREFIXES = (
    "take me to ", "go to the ", "go to ", "navigate to ", "open up ",
    "open ", "launch ", "start ", "get ", "put ",
)
_WEB_SUFFIXES = (" khol do", " khol de", " kholo", " open kar do", " open karo", " khol")
_BROWSER_SITE_RE = re.compile(
    r"\b(?:browser|chrome)\s+(?:mein|men|me|main|par|pe)\s+(.+?)\s+(?:kholo|khol do|khol de|open karo|open kar do)\b",
    re.I,
)
_TRAILERS = (
    " in the browser", " on the browser", " in browser", " on browser",
    " up", " there", " here", " now", " please", " online",
)
_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?:[/:][^\s]*)?$",
    re.I,
)


def _strip_politeness(text: str) -> str:
    low = (text or "").strip()
    for pre in ("can you please", "could you please", "can you", "could you", "please", "jarvis", "hey", "ok"):
        while True:
            if low.lower().startswith(pre + " "):
                low = low[len(pre) + 1:].strip().lstrip(", ")
                continue
            if low.lower() == pre:
                low = ""
                break
            break
    return low


def _clean_candidate(s: str) -> str:
    s = (s or "").strip()
    low = s.lower()
    changed = True
    while changed:
        changed = False
        for tr in _TRAILERS:
            if low.endswith(tr):
                low = low[: -len(tr)].strip()
                s = s[: -len(tr)].strip()
                changed = True
    s = s.strip().rstrip(".,!?;: ")
    if s.lower().startswith("the "):
        s = s[4:].strip()
    return s.strip()


def _classify_open(cand: str):
    """-> ("site", url, display) | ("url", url, display) | ("app", canon, display) | None"""
    if not cand or len(cand) > 40:
        return None
    low = cand.lower()
    url = _website_url(cand)
    if url is not None:
        return ("site", url, _site_display(cand))
    if _DOMAIN_RE.match(cand):
        return ("url", _normalize_url(cand), cand.rstrip("/"))
    canon = _canonical_app_target(low)
    if canon in _DESKTOP_APPS or low in _APP_TARGET_ALIASES:
        return ("app", canon, _canonical_app_target(low).capitalize())
    if low.endswith(".exe"):
        canon_exe = _canonical_app_target(low)
        if canon_exe in _DESKTOP_APPS:
            return ("app", canon_exe, canon_exe.capitalize())
    return None


_SITE_DISPLAY = {
    "amazon": "Amazon", "facebook": "Facebook", "github": "GitHub",
    "gmail": "Gmail", "google": "Google", "instagram": "Instagram",
    "linkedin": "LinkedIn", "netflix": "Netflix", "reddit": "Reddit",
    "twitter": "X", "whatsapp": "WhatsApp", "wikipedia": "Wikipedia",
    "yahoo": "Yahoo", "youtube": "YouTube",
}


def _site_display(cand: str) -> str:
    c = (cand or "").strip()
    for name, url in _WEBSITE_URLS.items():
        if _website_url(c) == url:
            return _SITE_DISPLAY.get(name, name.capitalize())
    c = c.split("://", 1)[-1].lstrip("www.").split("/")[0].split(".")[0]
    return c.capitalize()


def _normalize_url(u: str) -> str:
    """Canonical URL for same-page comparisons (known site -> full URL)."""
    u = (u or "").strip()
    known = _website_url(u)
    if known:
        return known
    if not u:
        return u
    if u.startswith("//"):
        u = "https:" + u
    if "://" not in u:
        u = "https://" + u
    return u.rstrip("/").lower()


def _same_url(a: str, b: str) -> bool:
    return bool(a) and _normalize_url(a) == _normalize_url(b)


def _extract_open_target(text: str):
    """Extract an obvious open-target (site | explicit url | desktop app)."""
    s = _strip_politeness(text).strip().rstrip(".?!, ")
    low = s.lower()
    if not low:
        return None
    # Hinglish suffix: "<target> kholo / open karo ..."
    for suffix in _WEB_SUFFIXES:
        if low.endswith(suffix):
            cand = s[: len(s) - len(suffix)].strip().rstrip("., ")
            r = _classify_open(cand)
            if r:
                return r
    # "browser mein <site> kholo"
    m = _BROWSER_SITE_RE.search(s)
    if m:
        r = _classify_open(_clean_candidate(m.group(1)))
        if r:
            return r
    # English prefixes: "open <target>", "go to <target>", ...
    for prefix in _WEB_PREFIXES:
        if low.startswith(prefix):
            cand = s[len(prefix):].strip()
            if cand:
                r = _classify_open(_clean_candidate(cand))
                if r:
                    return r
            return None
    return None


# ---- calculator intent ------------------------------------------------------

def _resolve_calculator(text: str, calc_state: dict | None):
    """-> ("perform", expr) | ("calculate", expr) | ("unresolved", None) | None

    "perform" means drive the real Calculator app; "calculate" means answer the
    math question. A bare calculator-frame with nothing to enter is intentionally
    NOT routed to the browser (requirement: calculator tasks must NEVER open the
    browser as a fallback).
    """
    state = calc_state or {}
    expr = _arithmetic_from_text(text)
    ctx_expr = state.get("expression")
    frame = _calc_frame(text)
    deictic = _calc_deictic_expr(text, state)
    if deictic:
        return ("perform", ctx_expr)
    if frame and expr:
        return ("perform", expr)
    if expr:
        return ("calculate", expr)
    if frame:
        return ("unresolved", None)
    return None


# ---- browser result selection ------------------------------------------------

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "pehla": 1, "pehli": 1, "dusra": 2, "dusri": 2, "teesra": 3, "teesri": 3,
}


def _ordinal_result_index(text: str) -> int | None:
    """1-based result index for browser-result phrases, else None."""
    low = " ".join((text or "").strip().lower().split())
    if not low:
        return None
    m = re.search(r"\b(first|second|third|fourth|fifth)\b[^.,!?]{0,14}\bresult\b", low)
    if not m:
        m = re.search(r"\b(pehla|pehli|dusra|dusri|teesra|teesri)\b[^.,!?]{0,14}\bresult\b", low)
    if m:
        return _ORDINALS[m.group(1)]
    if re.search(r"\b(?:that|the top|the best)\s+result\b", low):
        return 1
    if low in ("open that", "click that", "that result", "pehla result", "the first one"):
        return 1
    return None


# ---- reconcile model actions against the resolved intent ---------------------

def _ensure_website_open(actions: list[dict], url: str, display: str):
    """Never route a website-open to a search or to another page."""
    out: list[dict] = []
    have = False
    changed = False
    for action in actions:
        tp = str(action.get("type"))
        p = action.get("params") or {}
        if tp == "open_url":
            u = p.get("url")
            if isinstance(u, str) and _same_url(u, url):
                have = True
                out.append(action)
            else:
                changed = True  # misrouted to a different page -> drop
        elif tp == "browser_search":
            q = p.get("query")
            qk = _website_url(str(q)) if isinstance(q, (str,)) and q else None
            if qk is not None and _same_url(qk, url):
                changed = True  # model did a search for a site we were told to open
            else:
                out.append(action)
        elif tp == "open_application":
            tg = p.get("target")
            if isinstance(tg, str) and (
                _website_url(tg) or _canonical_app_target(tg) == "chrome"
            ):
                changed = True  # a website-open never needs a separate Chrome
                # launch — open_url starts/reuses the persistent browser.
            else:
                out.append(action)
        else:
            out.append(action)
    if not have:
        out.insert(0, {"type": "open_url", "params": {"url": url}})
        changed = True
    return out, changed


def _ensure_app_open(actions: list[dict], app: str):
    """Desktop-app intent never rides the browser."""
    out: list[dict] = []
    have = False
    changed = False
    for action in actions:
        tp = str(action.get("type"))
        p = action.get("params") or {}
        if tp == "open_application":
            tg = _canonical_app_target(p.get("target")) if isinstance(p.get("target"), str) else None
            if tg == app:
                have = True
                out.append(action)
            else:
                out.append(action)
        elif tp in ("open_url", "browser_search") or str(tp).startswith("browser_"):
            changed = True  # a browser action for a desktop-app request -> drop
        else:
            out.append(action)
    if not have:
        out.insert(0, {"type": "open_application", "params": {"target": app}})
        changed = True
    return out, changed


def _resolve_intents(plan: dict, user_text: str, browser_state: dict | None, calculator_state: dict | None) -> None:
    """Deterministic resolution pass applied INSIDE _finalize_plan.

    Order: obvious open-target -> calculator -> browser-result index.
    Each branch can rewrite plan["actions"] and, when the model's narration is
    now wrong (or missing), plan["spoken_response"].
    """
    text = (user_text or "").strip()
    if not text:
        return

    target = _extract_open_target(text)
    if target is not None:
        kind, value, display = target
        if kind in ("site", "url"):
            new_actions, changed = _ensure_website_open(plan.get("actions") or [], value, display)
            plan["actions"] = new_actions
            if changed:
                plan["spoken_response"] = f"Opening {display}."
        else:
            new_actions, changed = _ensure_app_open(plan.get("actions") or [], value)
            plan["actions"] = new_actions
            if changed:
                plan["spoken_response"] = f"Opening {display}."
        return

    calc = _resolve_calculator(text, calculator_state)
    if calc is not None:
        kind, expr = calc
        if kind == "unresolved":
            plan["actions"] = []
            plan["spoken_response"] = (
                "I couldn't find a calculation to enter into the Calculator. "
                "Tell me what you want to calculate."
            )
            return
        if kind == "calculate":
            try:
                val = _safe_eval(expr)
            except ValueError:
                plan["actions"] = []
                plan["spoken_response"] = "That calculation didn't work. Please check the numbers."
                return
            plan["actions"] = [{"type": "calculate", "params": {"expression": expr}}]
            plan["spoken_response"] = f"{expr} is {_fmt(val)}."
            return
        # kind == "perform"
        try:
            val = _safe_eval(expr)
        except ValueError:
            val = None
        plan["actions"] = [{"type": "calculator_perform", "params": {"expression": expr}}]
        plan["spoken_response"] = (
            f"Done - {expr} is {_fmt(val)}." if val is not None else
            f"Entering {expr} into Calculator."
        )
        return

    idx = _ordinal_result_index(text)
    if idx is not None and (browser_state or {}).get("has_results"):
        acts = plan.get("actions") or []
        if not acts:
            plan["actions"] = [{"type": "browser_open_result", "params": {"index": idx}}]
            plan["spoken_response"] = "Opening that result."
        elif all(str(a.get("type")) in ("open_url", "open_application") for a in acts):
            plan["actions"] = [{"type": "browser_open_result", "params": {"index": idx}}]
    # Otherwise leave the model's plan (or the empty plan) for the search and
    # website/back fallbacks in planner.py.


# ---------------------------------------------------------------------------
# Fallback response
# ---------------------------------------------------------------------------

def _fallback(message: str) -> dict:
    """Return a safe no-action plan with the given spoken message."""
    return {
        "response_language": "en",
        "spoken_response": message,
        "actions": [],
    }
