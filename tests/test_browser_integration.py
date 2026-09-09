"""Fake-browser integration tests (no Playwright, no internet, no live Chrome).

The REAL router + REAL tool handlers run against a scripted fake
BrowserController patched into ``tools.browser._controller``, so the exact
live flow — Open Chrome → Go to YouTube → Search Python tutorials → Open the
first result — is exercised end to end on a SINGLE browser instance:

  - the browser is launched exactly once and reused for every later action;
  - relative commands (search, "first result") resolve against remembered state;
  - failed browser actions surface the accurate error (no false success);
  - the orchestrator drives the same chain across multiple turns and feeds the
    live [Browser state] back into the planner.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assistant.orchestrator as orch  # noqa: E402
import tools.browser as browser_mod  # noqa: E402
from assistant.router import route  # noqa: E402


class _FakePage:
    def __init__(self):
        self.url = "about:blank"
        self.title = ""

    def is_closed(self):
        return False


class _FakeBrowserController:
    """Scripted stand-in for BrowserController with identical public methods."""

    def __init__(self):
        self._context = None
        self._page = None
        self.last_results = []
        self.last_search = None
        self.calls = []  # ordered, e.g. ["launch", "search:python", ...]

    def ensure_browser(self):
        if self._context is None:
            self._context = "fake-persistent-context"
            self._page = _FakePage()
            self.calls.append("launch")
        self.calls.append("ensure")
        return self._page

    @property
    def page(self):
        return self.ensure_browser()

    def close(self):
        self._context = None
        self._page = None

    def navigate(self, url):
        self.ensure_browser()
        if not url.startswith("http"):
            url = "https://" + url
        self._page.url = url
        self._page.title = url
        self.calls.append(f"navigate:{url}")
        return f"navigated {url}"

    def search(self, query):
        self.ensure_browser()
        self.last_search = query
        self.last_results = [
            {"title": f"Result {i}: {query}", "url": f"https://result{i}.example"}
            for i in (1, 2, 3)
        ]
        self._page.url = "https://search.example/?q=" + query
        self._page.title = "Search results"
        self.calls.append(f"search:{query}")
        return "3 results found"

    def open_result(self, index):
        if not self.last_results:
            raise ValueError("no search results available yet")
        if not (1 <= index <= len(self.last_results)):
            raise ValueError(
                f"result index {index} out of range (1-{len(self.last_results)})"
            )
        return self.navigate(self.last_results[index - 1]["url"])

    def go_back(self):
        self.ensure_browser()
        self._page.url = "https://back.example"
        self._page.title = "Back"
        self.calls.append("go_back")
        return "went back"


def _install(fake):
    original = browser_mod._controller
    browser_mod._controller = fake
    return original


# ---- Route-level: the exact live flow on one session ------------------------

def test_route_full_flow_reuses_browser_session():
    fake = _FakeBrowserController()
    orig = _install(fake)
    try:
        results = route(
            [
                {"type": "open_application", "params": {"target": "chrome"}},
                {"type": "open_url", "params": {"url": "https://www.youtube.com"}},
                {"type": "browser_search", "params": {"query": "Python tutorials"}},
                {"type": "browser_open_result", "params": {"index": 1}},
            ]
        )
        assert len(results) == 4
        assert all(r["ok"] for r in results), results

        # One persistent session: exactly one launch, never relaunched.
        assert fake.calls.count("launch") == 1
        # Order: launched before navigating, searched, then opened result 1.
        assert fake.calls.index("launch") < fake.calls.index("navigate:https://www.youtube.com")
        assert "search:Python tutorials" in fake.calls
        assert fake.calls[-1] == "navigate:https://result1.example"
    finally:
        browser_mod._controller = orig
        fake.close()


def test_browser_not_relaunched_across_route_calls():
    fake = _FakeBrowserController()
    orig = _install(fake)
    try:
        r1 = route([{"type": "open_url", "params": {"url": "https://www.youtube.com"}}])
        assert r1[0]["ok"]
        assert fake.calls.count("launch") == 1

        r2 = route([{"type": "browser_search", "params": {"query": "python"}}])
        assert r2[0]["ok"]
        assert fake.calls.count("launch") == 1  # still the same session
        assert fake.calls.count("ensure") >= 2
    finally:
        browser_mod._controller = orig
        fake.close()


def test_out_of_range_result_reports_truthful_error():
    fake = _FakeBrowserController()
    orig = _install(fake)
    try:
        assert route([{"type": "browser_search", "params": {"query": "python"}}])[0]["ok"]
        results = route([{"type": "browser_open_result", "params": {"index": 9}}])
        assert results[0]["ok"] is False
        assert "out of range" in results[0]["error"]
        assert "(1-3)" in results[0]["error"]
    finally:
        browser_mod._controller = orig
        fake.close()


def test_open_first_result_without_prior_search_fails_truthfully():
    fake = _FakeBrowserController()
    orig = _install(fake)
    try:
        results = route([{"type": "browser_open_result", "params": {"index": 1}}])
        assert results[0]["ok"] is False
        assert "no search results available yet" in results[0]["error"]
    finally:
        browser_mod._controller = orig
        fake.close()


# ---- Orchestrator-level: multi-turn conversation, same session --------------

def _plan_for(text, actions):
    return {
        "spoken_response": text,
        "actions": actions,
        "response_language": "en",
    }


def test_orchestrator_multi_turn_drives_browser():
    fake = _FakeBrowserController()
    orig = _install(fake)
    orig_plan = orch.plan
    orig_speak = orch._speak
    planner_log = []

    def fake_plan(utterance, ctx="", browser_state=None, calculator_state=None):
        planner_log.append((utterance, ctx))
        t = utterance.lower()
        if "chrome" in t:
            return _plan_for("Opening Chrome.", [{"type": "open_application", "params": {"target": "chrome"}}])
        if "youtube" in t:
            return _plan_for("Opening YouTube.", [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}])
        if "search" in t:
            return _plan_for("Searching.", [{"type": "browser_search", "params": {"query": "Python tutorials"}}])
        if "result" in t:
            return _plan_for("Opening the first result.", [{"type": "browser_open_result", "params": {"index": 1}}])
        return _plan_for("ok.", [])

    orch.plan = fake_plan
    orch._speak = lambda text: None
    try:
        orch.reset()
        orch.run_turn("Open Chrome")
        orch.run_turn("Go to YouTube")
        orch.run_turn("Search for Python tutorials")
        orch.run_turn("Open the first result")

        # Exactly one browser session across the whole conversation.
        assert fake.calls.count("launch") == 1
        assert fake.calls.index("launch") < fake.calls.index("navigate:https://www.youtube.com")
        assert "search:Python tutorials" in fake.calls
        assert fake.calls[-1] == "navigate:https://result1.example"

        # From the 2nd turn on, the live browser state reached the planner.
        assert "[Browser state]" not in planner_log[0][1]
        assert "[Browser state]" in planner_log[1][1]
        assert "[Browser state]" in planner_log[2][1]
    finally:
        orch.plan = orig_plan
        orch._speak = orig_speak
        orch.reset()
        browser_mod._controller = orig
        fake.close()


def test_orchestrator_failed_browser_step_speaks_truthfully():
    fake = _FakeBrowserController()
    orig = _install(fake)
    orig_plan = orch.plan
    orig_speak = orch._speak
    spoken = []

    def fake_plan(utterance, ctx="", browser_state=None, calculator_state=None):
        return _plan_for(
            "Opening the first result.",
            [{"type": "browser_open_result", "params": {"index": 1}}],
        )

    orch.plan = fake_plan
    orch._speak = lambda text: spoken.append(text)
    try:
        orch.reset()
        text = orch.run_turn("Open the first result")
        assert text is not None
        assert "Opening the first result." not in text  # no false success
        assert "no search results" in text              # accurate error surfaced
        assert len(spoken) == 1
        assert spoken[0] == text
    finally:
        orch.plan = orig_plan
        orch._speak = orig_speak
        orch.reset()
        browser_mod._controller = orig
        fake.close()


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("browser integration: all tests passed")


if __name__ == "__main__":
    _main()