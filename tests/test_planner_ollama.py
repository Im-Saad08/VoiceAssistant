"""Offline tests for the Ollama planner backend.

No Ollama server is required — these verify:
  - Response parsing handles well-formed, malformed, and fenced JSON.
  - Connection/server errors produce safe fallback responses.
  - Retry logic applies correctly for transient vs fatal errors.
  - The return contract is identical to the Gemini planner.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import planner_ollama  # noqa: E402
from assistant.planner_ollama import (  # noqa: E402
    _build_system_prompt_ollama,
    _normalize_website_actions,
    _parse_ollama_response,
    _website_url,
)
from assistant.planner_utils import _coerce_actions, _compact_catalog  # noqa: E402


# ---- _parse_ollama_response tests -------------------------------------------

def test_parse_well_formed_json():
    content = '{"spoken_response":"Hello!","actions":[{"type":"get_time","params":{}}],"response_language":"en"}'
    result = planner_ollama._parse_ollama_response(content)
    assert result["spoken_response"] == "Hello!"
    assert result["response_language"] == "en"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["type"] == "get_time"


def test_parse_missing_spoken_response_raises():
    content = '{"actions":[],"response_language":"en"}'
    try:
        planner_ollama._parse_ollama_response(content)
    except ValueError as e:
        assert "spoken_response" in str(e)
    else:
        raise AssertionError("expected ValueError for missing spoken_response")


def test_parse_defaults_response_language():
    content = '{"spoken_response":"ok","actions":[]}'
    result = planner_ollama._parse_ollama_response(content)
    assert result["response_language"] == "en"


def test_parse_coerces_actions():
    """Even if the model emits malformed actions, _coerce_actions normalizes."""
    content = '{"spoken_response":"ok","actions":"not a list"}'
    result = planner_ollama._parse_ollama_response(content)
    assert result["actions"] == []


def test_parse_code_fenced_json():
    """Small models sometimes wrap output in ```json fences."""
    content = '```json\n{"spoken_response":"hi","actions":[]}\n```'
    result = planner_ollama._parse_ollama_response(content)
    assert result["spoken_response"] == "hi"


def test_parse_bare_code_fence():
    content = '```\n{"spoken_response":"hi","actions":[]}\n```'
    result = planner_ollama._parse_ollama_response(content)
    assert result["spoken_response"] == "hi"


def test_parse_invalid_json_raises():
    try:
        planner_ollama._parse_ollama_response("not json at all {{{")
    except Exception:
        return
    raise AssertionError("expected an exception for invalid JSON")


def test_parse_empty_actions():
    content = '{"spoken_response":"Sure thing!","actions":[]}'
    result = planner_ollama._parse_ollama_response(content)
    assert result["actions"] == []
    assert result["spoken_response"] == "Sure thing!"


# ---- Connection error handling ----------------------------------------------

class _FakeResponse:
    """Simulates requests.Response for connection/server error tests."""

    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)

    def json(self):
        return self._json


class _FakeSession:
    """Simulates requests.Session for testing error/retry paths."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._calls = 0

    def post(self, url, json=None, timeout=None):
        resp = self._responses.pop(0)
        self._calls += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_connection_error_produces_fallback():
    """When Ollama is not running, plan() must return a helpful fallback."""
    import requests
    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES

    session = _FakeSession([requests.ConnectionError("Connection refused")])
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 1
    try:
        result = planner_ollama.plan("open notepad")
        assert result["actions"] == []
        assert "Ollama" in result["spoken_response"] or "local AI" in result["spoken_response"]
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries


def test_model_not_found_fails_immediately():
    """When the model is not downloaded, plan() must NOT retry."""
    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES

    # HTTP 404 from Ollama means model not found
    session = _FakeSession([_FakeResponse(404)])
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 3
    try:
        result = planner_ollama.plan("hello")
        assert result["actions"] == []
        assert "not installed" in result["spoken_response"].lower() or "not found" in result["spoken_response"].lower()
        assert session._calls == 1  # did NOT retry
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries


def test_connection_error_retries_with_backoff():
    """Transient connection errors must retry with increasing backoff."""
    import requests as req

    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES
    orig_sleep = planner_ollama.time.sleep

    sleeps = []
    session = _FakeSession([
        req.ConnectionError("timeout"),
        req.ConnectionError("timeout"),
        _FakeResponse(200, {"message": {"content": '{"spoken_response":"hi","actions":[]}'}}),
    ])
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 3
    planner_ollama.time.sleep = sleeps.append
    try:
        result = planner_ollama.plan("hi")
        assert result["spoken_response"] == "hi"
        assert session._calls == 3
        assert len(sleeps) == 2  # waited between retries
        assert sleeps[0] < sleeps[1]  # backoff increases
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries
        planner_ollama.time.sleep = orig_sleep


def test_successful_plan():
    """Happy path: Ollama returns a well-formed plan."""
    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES

    llm_output = json.dumps({
        "spoken_response": "Opening Notepad for you.",
        "actions": [{"type": "open_application", "params": {"target": "notepad"}}],
        "response_language": "en",
    })
    session = _FakeSession([_FakeResponse(200, {"message": {"content": llm_output}})])
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 1
    try:
        result = planner_ollama.plan("open notepad")
        assert result["spoken_response"] == "Opening Notepad for you."
        assert len(result["actions"]) == 1
        assert result["actions"][0]["type"] == "open_application"
        assert result["actions"][0]["params"]["target"] == "notepad"
        assert result["response_language"] == "en"
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries


def test_multi_action_plan():
    """Two-action plan from Ollama passes through correctly."""
    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES

    llm_output = json.dumps({
        "spoken_response": "Opening Chrome and YouTube.",
        "actions": [
            {"type": "open_application", "params": {"target": "chrome"}},
            {"type": "open_url", "params": {"url": "youtube.com"}},
        ],
        "response_language": "en",
    })
    session = _FakeSession([_FakeResponse(200, {"message": {"content": llm_output}})])
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 1
    try:
        result = planner_ollama.plan("chrome kholo aur youtube kholo")
        assert len(result["actions"]) == 2
        assert result["actions"][0]["type"] == "open_application"
        assert result["actions"][1]["type"] == "open_url"
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries


def test_plan_returns_correct_contract():
    """The return dict must have exactly the keys the orchestrator expects."""
    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES

    llm_output = json.dumps({
        "spoken_response": "It is 3 PM.",
        "actions": [],
        "response_language": "en",
    })
    session = _FakeSession([_FakeResponse(200, {"message": {"content": llm_output}})])
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 1
    try:
        result = planner_ollama.plan("what time is it")
        assert set(result.keys()) == {"spoken_response", "actions", "response_language"}
        assert isinstance(result["actions"], list)
        assert isinstance(result["spoken_response"], str)
        assert isinstance(result["response_language"], str)
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries


# ---- Regression: the exact live failure case --------------------------------
# Real qwen2.5:1.5b returned a flat object inspired by the tool catalog —
#   {"action": "open_application", "target": "notepad"}
# — i.e. no spoken_response / actions / response_language contract, which made
# plan() fall back with "missing 'spoken_response'".
# The fix is the Ollama-specific prompt (tested live); these tests lock in the
# parsing contract: the old flat shape must STILL be rejected, never repaired
# by guessing params or wrapping.

FLAT_CATALOG_SHAPED_OUTPUT = '{"action": "open_application", "target": "notepad"}'

def test_flat_catalog_shaped_output_rejected_not_repaired():
    """The old bad shape must not be silently wrapped into a plan."""
    try:
        _parse_ollama_response(FLAT_CATALOG_SHAPED_OUTPUT)
    except ValueError as e:
        assert "spoken_response" in str(e)
    else:
        raise AssertionError(
            "flat {'action':...} output must raise, not be repaired by guessing params"
        )


def test_flat_output_plan_returns_safe_fallback():
    """plan() against that raw content must produce a fallback, not a plan."""
    session = _FakeSession(
        [_FakeResponse(200, {"message": {"content": FLAT_CATALOG_SHAPED_OUTPUT}})]
    )
    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 1
    try:
        result = planner_ollama.plan("open notepad")
        assert result["actions"] == []
        assert "understanding" in result["spoken_response"].lower() or \
               "trouble" in result["spoken_response"].lower()
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries


def test_ollama_prompt_requests_exact_contract():
    """The Ollama system prompt must name all three required keys + an example."""
    s = _build_system_prompt_ollama()
    for key in ("spoken_response", "actions", "response_language"):
        assert key in s, f"prompt missing required key '{key}'"
    assert '"target":"notepad"' in s  # concrete filled-in example present
    assert "OUTPUT FORMAT" in s


def test_ollama_prompt_compact_catalog_lists_required_params():
    """Compact catalog must call out required params (e.g. open_application)."""
    cat = _compact_catalog()
    assert "open_application" in cat
    assert "REQUIRED params: target" in cat
    assert "browser_open_result" in cat
    assert "REQUIRED params: index" in cat


# ---- Regression: websites must route to open_url, not open_application -------
# Live Ollama testing showed qwen2.5:1.5b over-applies open_application to
# service names (gmail, github, instagram) no matter how the prompt is worded.
# The fix is a deterministic, allowlisted planner-side map: an exact website
# match rewrites the action to open_url with the mapped URL. These tests lock
# in the mapping without touching the router or weakening validation.

def test_website_url_maps_known_sites():
    assert _website_url("youtube") == "https://www.youtube.com"
    assert _website_url("gmail") == "https://mail.google.com"
    assert _website_url("google") == "https://www.google.com"
    assert _website_url("github") == "https://github.com"
    assert _website_url("instagram") == "https://www.instagram.com"


def test_website_url_matches_domains_and_tld_variants():
    assert _website_url("gmail.com") == "https://mail.google.com"
    assert _website_url("www.youtube.com") == "https://www.youtube.com"
    assert _website_url("https://github.com") == "https://github.com"
    assert _website_url("mail.google.com") == "https://mail.google.com"


def test_website_url_refuses_unknown_and_desktop_apps():
    assert _website_url("notepad") is None
    assert _website_url("chrome") is None
    assert _website_url("calc") is None
    assert _website_url("randomword") is None
    assert _website_url("") is None
    assert _website_url("  ") is None


def test_normalize_rewrites_website_open_application_to_open_url():
    actions = [{"type": "open_application", "params": {"target": "gmail"}}]
    out = _normalize_website_actions(actions)
    assert out == [{"type": "open_url", "params": {"url": "https://mail.google.com"}}]


def test_normalize_handles_target_with_tld():
    actions = [{"type": "open_application", "params": {"target": "github.com"}}]
    out = _normalize_website_actions(actions)
    assert out == [{"type": "open_url", "params": {"url": "https://github.com"}}]


def test_normalize_keeps_desktop_app_actions():
    actions = [{"type": "open_application", "params": {"target": "chrome"}}]
    assert _normalize_website_actions(actions) == actions
    actions = [{"type": "open_application", "params": {"target": "notepad"}}]
    assert _normalize_website_actions(actions) == actions


def test_normalize_expands_bare_open_url_name():
    actions = [{"type": "open_url", "params": {"url": "gmail"}}]
    assert _normalize_website_actions(actions) == [{
        "type": "open_url",
        "params": {"url": "https://mail.google.com"},
    }]


def test_normalize_leaves_full_urls_untouched():
    actions = [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}]
    assert _normalize_website_actions(actions) == actions
    # Chrome-inspired URL with a dot is never rewritten.
    actions = [{"type": "open_url", "params": {"url": "youtube.com"}}]
    assert _normalize_website_actions(actions) == actions


def test_plan_rewrites_model_open_application_gmail_to_open_url():
    """End-to-end at the planner layer: the exact live failure case."""
    orig_client = planner_ollama._get_client
    orig_retries = planner_ollama.MAX_RETRIES

    llm_output = json.dumps({
        "spoken_response": "Opening Gmail.",
        "actions": [{"type": "open_application", "params": {"target": "gmail"}}],
        "response_language": "en",
    })
    session = _FakeSession([_FakeResponse(200, {"message": {"content": llm_output}})])
    planner_ollama._get_client = lambda: session
    planner_ollama.MAX_RETRIES = 1
    try:
        result = planner_ollama.plan("open gmail")
        assert result["spoken_response"] == "Opening Gmail."
        assert result["actions"][0]["type"] == "open_url"
        assert result["actions"][0]["params"]["url"] == "https://mail.google.com"
    finally:
        planner_ollama._get_client = orig_client
        planner_ollama.MAX_RETRIES = orig_retries


def test_ollama_prompt_forbids_websites_as_open_application():
    """The prompt must teach websites -> open_url.

    Live testing showed that putting per-site examples or a "Open Chrome IS an
    exception" line at the END of the prompt backfires: the 1.5B model latches
    onto the last JSON shape and echoes `open_application target chrome` for
    ANY website. So the committed prompt carries only the rule line, and the
    deterministic _WEBSITE_URLS map (tested separately) is the guarantee.
    """
    s = _build_system_prompt_ollama()
    assert "open_url" in s and "open_application" in s
    assert "NEVER open_application for a website" in s
    assert "open_url with the site's url" in s
    # No confabulation magnets: no inline per-site URL examples and no
    # "...IS an exception" style caveat — the 1.5B model latches onto the last
    # JSON/exception shape and starts emitting open_application for every site.
    assert "open gmail" not in s
    assert "mail.google.com" not in s
    assert "IS an exception" not in s
    # chrome still belongs to open_application (a desktop app), and the
    # browser-reuse rule may name it only in a prohibition ("do NOT emit
    # 'Open Chrome'"), which is not the confabulation shape.
    assert "chrome" in s


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("planner ollama: all tests passed")


if __name__ == "__main__":
    _main()
