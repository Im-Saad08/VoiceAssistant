"""Offline tests for defensive normalization of Gemini planner output.

No Gemini API is used here — these verify that whatever shape the model
returns (dict, string, list of malformed items), the planner coerces it to a
clean list of action dicts that the router can validate deterministically.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import planner  # noqa: E402
from assistant import planner_gemini  # noqa: E402
from assistant.planner import _coerce_actions  # noqa: E402


class _FakeClient:
    """Fake genai client: raises `exc` on every generate_content call.

    lets the retry tests count real API attempts without network access.
    """

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):  # noqa: ANN003
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(text='{"spoken_response":"hi","actions":[]}')


def _plan_with_failure(exc: Exception, max_retries: int = 3):
    """Run planner.plan() against a fake Gemini client that raises `exc`.

    Forces PLANNER_BACKEND to 'gemini' so the call exercises the Gemini retry
    path, then restores config and all patched globals afterwards. Returns
    (result, client, slept_seconds_list).
    """
    import assistant.config as cfg

    orig_backend = cfg.PLANNER_BACKEND
    orig_get_client = planner_gemini._get_client
    orig_retries = planner_gemini.GEMINI_MAX_RETRIES
    orig_sleep = planner_gemini.time.sleep

    client = _FakeClient(exc)
    sleeps: list[float] = []
    try:
        cfg.PLANNER_BACKEND = "gemini"
        planner_gemini._get_client = lambda: client
        planner_gemini.GEMINI_MAX_RETRIES = max_retries
        planner_gemini.time.sleep = sleeps.append
        result = planner.plan("test")
    finally:
        cfg.PLANNER_BACKEND = orig_backend
        planner_gemini._get_client = orig_get_client
        planner_gemini.GEMINI_MAX_RETRIES = orig_retries
        planner_gemini.time.sleep = orig_sleep
    return result, client, sleeps


def test_coerce_actions_non_list():
    assert _coerce_actions(None) == []
    assert _coerce_actions({"type": "get_time"}) == []
    assert _coerce_actions("i am json") == []


def test_coerce_actions_wellformed_untouched():
    plan = [{"type": "get_time", "params": {}}]
    assert _coerce_actions(plan) == plan


def test_coerce_actions_repairs_missing_params():
    plan = [{"type": "open_application"}, {"type": "get_time", "params": "x"}]
    out = _coerce_actions(plan)
    assert out == [
        {"type": "open_application", "params": {}},
        {"type": "get_time", "params": {}},
    ]


def test_coerce_actions_drops_non_dict_items():
    out = _coerce_actions([{"type": "get_time", "params": {}}, 42, "junk"])
    assert out == [{"type": "get_time", "params": {}}]


def test_coerce_actions_never_guesses_params():
    # Missing required values must NOT be filled in here — the router rejects
    # them. Coercion only normalizes the container shape.
    out = _coerce_actions([{"type": "open_application", "params": {}}])
    assert out[0]["params"] == {}


# ---- Retry policy -----------------------------------------------------------

_DAILY_QUOTA_MSG = (
    "429 RESOURCE_EXHAUSTED. {'message': 'You exceeded your current quota, "
    "please check your plan and billing details. ... Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
    "model: gemini-3.6-flash', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}"
)


def test_daily_quota_exhausted_does_not_retry():
    """The exact 429 quoted in the report: must fail immediately, no retry."""
    result, client, sleeps = _plan_with_failure(Exception(_DAILY_QUOTA_MSG))
    assert client.calls == 1          # exactly one API attempt
    assert sleeps == []               # no waiting at all
    assert result["actions"] == []    # safe fallback
    assert "AI service" in result["spoken_response"]


def test_transient_server_error_still_retries():
    """503 (server overload) is transient — the retry path must be preserved."""
    exc = Exception("503 RESOURCE_EXHAUSTED Service Unavailable / high demand")
    result, client, sleeps = _plan_with_failure(exc, max_retries=3)
    assert client.calls == 3          # retried to the max
    assert len(sleeps) == 2           # waited between each attempt


def test_per_minute_throttle_still_retries():
    """A 429 with a retry hint but no daily markers is transient, not fatal."""
    exc = Exception(
        "429 RESOURCE_EXHAUSTED Quota exceeded for metric: "
        "generate_content_free_tier_requests, limit: 20 requests per minute. "
        "Please retry in 30.5s."
    )
    result, client, sleeps = _plan_with_failure(exc, max_retries=3)
    assert client.calls == 3
    assert len(sleeps) == 2
    assert result["actions"] == []


# ---- Dispatcher routing -----------------------------------------------------

def test_backend_ollama_routes_to_ollama_planner():
    """When PLANNER_BACKEND='ollama', plan() must call the Ollama backend."""
    import assistant.config as cfg
    from assistant import planner_ollama

    orig = cfg.PLANNER_BACKEND
    orig_plan = planner_ollama.plan
    calls = []
    planner_ollama.plan = lambda t, c="": (  # noqa: ANN001,ANN002
        calls.append((t, c))
        or {"spoken_response": "test", "actions": [], "response_language": "en"}
    )
    try:
        cfg.PLANNER_BACKEND = "ollama"
        result = planner.plan("hello")
        assert len(calls) == 1
        assert calls[0] == ("hello", "")
        assert result["spoken_response"] == "test"
    finally:
        cfg.PLANNER_BACKEND = orig
        planner_ollama.plan = orig_plan


def test_backend_gemini_routes_to_gemini_planner():
    """When PLANNER_BACKEND='gemini', plan() must call the Gemini backend."""
    import assistant.config as cfg
    from assistant import planner_gemini

    orig = cfg.PLANNER_BACKEND
    orig_plan = planner_gemini.plan
    calls = []
    planner_gemini.plan = lambda t, c="": (  # noqa: ANN001,ANN002
        calls.append((t, c))
        or {"spoken_response": "gemini says hi", "actions": [], "response_language": "en"}
    )
    try:
        cfg.PLANNER_BACKEND = "gemini"
        result = planner.plan("test msg")
        assert len(calls) == 1
        assert calls[0] == ("test msg", "")
        assert result["spoken_response"] == "gemini says hi"
    finally:
        cfg.PLANNER_BACKEND = orig
        planner_gemini.plan = orig_plan


def test_default_backend_is_ollama():
    """PLANNER_BACKEND must default to 'ollama' when unset."""
    import assistant.config as cfg
    assert cfg.PLANNER_BACKEND == "ollama"


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("planner offline: all tests passed")


if __name__ == "__main__":
    _main()
