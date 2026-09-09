"""Playwright-driven browser automation.

A single BrowserController manages one persistent, headed Chromium session
so that "Open Chrome", "go to YouTube", "search", and "open the first result"
all operate on the same browser. Search results are stored on the controller
so later turns can reference "the first result" conversationally.

Playwright (sync API) is used rather than PyAutoGUI because it is structured
and reliable for websites.
"""

from __future__ import annotations

import re

from tools.registry import tool
from assistant.config import BROWSER_HEADED, BROWSER_PROFILE_DIR, SEARCH_ENGINE
from assistant.logging_setup import get_logger

logger = get_logger("browser")

# Boilerplate links never surfaced as results (generic fallback only).
_JUNK_DOMAIN_RE = re.compile(
    r"(gstatic\.|googleusercontent\.|youtube\.com#|/search\?|\.google\.[a-z]+/(intl|preference|interstitial))",
    re.I,
)
_DEFAULT_TIMEOUT_MS = 20_000
_MAX_RESULTS = 6


class BrowserController:
    def __init__(self) -> None:
        self._pw = None
        self._context = None
        self._page = None
        self.last_results: list[dict] = []
        self.last_search = None

    # ---- lifecycle -----------------------------------------------------
    def ensure_browser(self):
        if self._context is not None:
            # Reuse an existing page if it's still alive.
            if self._page is not None and not self._page.is_closed():
                return self._page
            self._page = self._context.new_page()
            return self._page
        return self._launch()

    def _launch(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        if BROWSER_PROFILE_DIR:
            self._context = self._pw.chromium.launch_persistent_context(
                user_data_dir=BROWSER_PROFILE_DIR,
                headless=not BROWSER_HEADED,
                viewport={"width": 1280, "height": 800},
            )
        else:
            browser = self._pw.chromium.launch(headless=not BROWSER_HEADED)
            self._context = browser.new_context(
                viewport={"width": 1280, "height": 800}
            )
        self._context.set_default_timeout(_DEFAULT_TIMEOUT_MS)
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        logger.info("browser launched (headless=%s)", not BROWSER_HEADED)
        return self._page

    def close(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        except Exception:  # noqa: BLE001
            logger.debug("browser close swallowed", exc_info=True)
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:  # noqa: BLE001
            pass
        self._context = None
        self._page = None
        self._pw = None
        logger.info("browser closed")

    @property
    def page(self):
        return self.ensure_browser()

    # ---- navigation -----------------------------------------------------
    def navigate(self, url: str) -> str:
        if not re.match(r"^https?://", url):
            url = "https://" + url
        page = self.page
        page.goto(url, wait_until="domcontentloaded")
        return self._status()

    def _status(self) -> str:
        try:
            return f"{self._page.title()} | {self._page.url}"
        except Exception:  # noqa: BLE001
            return "browser open"

    # ---- search ---------------------------------------------------------
    def search(self, query: str) -> str:
        page = self.page
        url = _search_url(query)
        try:
            page.goto(url, wait_until="domcontentloaded")
            # Give server-rendered results a moment to appear.
            try:
                page.wait_for_selector("li.b_algo, h2 a, h3 a", timeout=6000)
            except Exception:  # noqa: BLE001
                logger.debug("search results selector not found", exc_info=True)
        except Exception:  # noqa: BLE001
            logger.warning("search page load issue", exc_info=True)
        self._extract_results()
        self.last_search = query
        if not self.last_results:
            return f"No results extracted for '{query}'"
        return f"{len(self.last_results)} results found"

    def _extract_results(self) -> None:
        page = self.page
        pairs: list[tuple[str, str]] = []
        # Engine-specific result blocks (title heading + anchor inside).
        for block_sel, heading_sel in (("li.b_algo", "h2"), ("div.b_algo", "h2")):
            blocks = page.locator(block_sel)
            n = blocks.count()
            if n:
                for i in range(min(n, _MAX_RESULTS * 2)):
                    try:
                        b = blocks.nth(i)
                        heading = b.locator(heading_sel).first
                        title = re.sub(r"\s+", " ", (heading.inner_text() or "")).strip()
                        anchor = heading.locator("a").first
                        href = _resolve_href(anchor.get_attribute("href") or "")
                    except Exception:  # noqa: BLE001
                        continue
                    if title:
                        pairs.append((title, href))
                if pairs:
                    break
        if not pairs:
            # Generic: heading-wrapped anchors.
            for sel in ("h2 a", "h3 a"):
                loc = page.locator(sel)
                n = loc.count()
                if n:
                    for i in range(min(n, _MAX_RESULTS * 2)):
                        try:
                            title = re.sub(r"\s+", " ", (loc.nth(i).inner_text() or "")).strip()
                            href = _resolve_href(loc.nth(i).get_attribute("href") or "")
                        except Exception:  # noqa: BLE001
                            continue
                        if title:
                            pairs.append((title, href))
                    if pairs:
                        break
        if not pairs:
            # Last resort: any external anchor with meaningful text.
            loc = page.locator("a[href]")
            n = loc.count()
            for i in range(n):
                try:
                    title = re.sub(r"\s+", " ", (loc.nth(i).inner_text() or "")).strip()
                    href = _resolve_href(loc.nth(i).get_attribute("href") or "")
                except Exception:  # noqa: BLE001
                    continue
                if title and len(title) >= 12:
                    pairs.append((title, href))
                if len(pairs) >= _MAX_RESULTS * 2:
                    break

        results: list[dict] = []
        seen: set[str] = set()
        for text, href in pairs:
            if not text or len(text) < 12:
                continue
            if not href.startswith("http"):
                continue
            if _JUNK_DOMAIN_RE.search(href):
                continue
            if href in seen:
                continue
            seen.add(href)
            results.append({"title": text[:120], "url": href})
            if len(results) >= _MAX_RESULTS:
                break
        self.last_results = results

    def open_result(self, index: int) -> str:
        if not self.last_results:
            raise ValueError("no search results available yet")
        if not (1 <= index <= len(self.last_results)):
            raise ValueError(f"result index {index} out of range (1-{len(self.last_results)})")
        url = self.last_results[index - 1]["url"]
        return self.navigate(url)

    def go_back(self) -> str:
        page = self.page
        try:
            page.go_back(wait_until="domcontentloaded")
        except Exception:  # noqa: BLE001
            logger.debug("go_back failed", exc_info=True)
        return self._status()

    # ---- interaction -----------------------------------------------------
    def click_text(self, text: str) -> str:
        page = self.page
        locator = page.get_by_text(text, exact=False).first
        locator.scroll_into_view_if_needed()
        locator.click()
        return self._status()

    def type_text(self, selector: str, text: str) -> str:
        page = self.page
        locator = page.locator(selector).first
        locator.click()
        locator.fill(text)
        return self._status()

    def press_key(self, key: str) -> str:
        page = self.page
        page.keyboard.press(key)
        return self._status()

    def read_page(self, max_chars: int = 1500) -> str:
        page = self.page
        try:
            text = page.inner_text("body")
        except Exception:  # noqa: BLE001
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]


def _quote(s: str) -> str:
    from urllib.parse import quote

    return quote(s)


def _resolve_href(href: str) -> str:
    """Turn redirect hrefs (Bing ck/a, DuckDuckGo uddg) into real target URLs."""
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    if "bing.com/ck/a" in href:
        target = _extract_param(href, "u")
        if target:
            decoded = _base64url_decode(target)
            if decoded:
                return decoded
    if "duckduckgo.com/l/" in href:
        target = _extract_param(href, "uddg")
        if target:
            return target
    return href


def _extract_param(url: str, key: str) -> str | None:
    from urllib.parse import parse_qs, urlparse

    try:
        qs = parse_qs(urlparse(url).query)
        val = qs.get(key, [None])[0]
        return val if val else None
    except Exception:  # noqa: BLE001
        return None


def _base64url_decode(s: str) -> str:
    import base64

    # Bing sometimes prefixes real URLs with "a1".
    if s.startswith("a1"):
        s = s[2:]
    try:
        padded = s + "=" * (-len(s) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8")
    except Exception:  # noqa: BLE001
        return ""


def _search_url(query: str) -> str:
    if SEARCH_ENGINE == "duckduckgo":
        return "https://duckduckgo.com/?q=" + _quote(query)
    if SEARCH_ENGINE == "google":
        return "https://www.google.com/search?q=" + _quote(query)
    return "https://www.bing.com/search?q=" + _quote(query)


# ---- module-level singleton -----------------------------------------------
_controller = BrowserController()


def ensure_browser():
    return _controller.ensure_browser()


def close_browser() -> None:
    _controller.close()


# ---- planner-facing state snapshot (pure read, never launches the browser) --
def browser_state_snapshot() -> dict:
    """Read-only view of the controller's state for the planner prompt.

    Never starts Playwright: fields are only read when the browser is already
    open. Shape: {open, url, title, has_results, result_count, last_search}.
    """
    c = _controller
    state = {
        "open": c._context is not None,
        "url": None,
        "title": None,
        "has_results": bool(c.last_results),
        "result_count": len(c.last_results),
        "last_search": c.last_search,
    }
    if state["open"]:
        try:
            page = c._page
            if page is not None and not page.is_closed():
                state["url"] = page.url
                state["title"] = page.title
        except Exception:  # noqa: BLE001
            logger.debug("browser state read failed", exc_info=True)
    return state


def browser_state_text() -> str:
    """A short human block describing the live browser state ("" if closed).

    Fed to the planner via [Browser state] so relative commands ("the first
    result", "go back", "there") resolve against real, current state.
    """
    s = browser_state_snapshot()
    if not s["open"]:
        return ""
    lines = ["browser is already open (reuse this session)"]
    if s["url"]:
        title = s["title"] or ""
        lines.append(f"current page: {title} | {s['url']}")
    if s["last_search"]:
        lines.append(f"last search: {s['last_search']}")
    if s["has_results"]:
        lines.append(
            f"search results available: {s['result_count']} "
            "(pick by number via browser_open_result — 'the first result' is index 1)"
        )
    return "; ".join(lines)


# ---- registered tools -----------------------------------------------------
@tool(
    name="open_url",
    description=(
        "Open or navigate to a URL/website in the browser (opens the browser "
        "if it is closed, otherwise reuses the open one)."
    ),
    params={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "full URL or domain"}},
        "required": ["url"],
    },
    aliases=["browser_navigate", "open_website"],
)
def _open_url(url: str) -> str:
    return _controller.navigate(url)


@tool(
    name="browser_search",
    description="Search the web for a query and remember the results.",
    params={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search query"}},
        "required": ["query"],
    },
)
def _browser_search(query: str) -> str:
    return _controller.search(query)


@tool(
    name="browser_open_result",
    description=(
        "Open one of the remembered search results by number (1 = first "
        "result, 2 = second). 'the first result' / 'that result' is index 1."
    ),
    params={
        "type": "object",
        "properties": {"index": {"type": "integer", "description": "1-based result number"}},
        "required": ["index"],
    },
    aliases=["open_search_result"],
)
def _browser_open_result(index: int) -> str:
    return _controller.open_result(index)


@tool(
    name="browser_click",
    description="Click an element on the current page by its visible text.",
    params={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "visible text to click"}},
        "required": ["text"],
    },
)
def _browser_click(text: str) -> str:
    return _controller.click_text(text)


@tool(
    name="browser_type",
    description="Type text into a field on the current page using a CSS selector.",
    params={
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "CSS selector for the input"},
            "text": {"type": "string", "description": "text to type"},
        },
        "required": ["selector", "text"],
    },
)
def _browser_type(selector: str, text: str) -> str:
    return _controller.type_text(selector, text)


@tool(
    name="browser_go_back",
    description="Go back to the previous page in the browser ('go back' / 'back').",
    params={"type": "object", "properties": {}},
    aliases=["browser_back"],
)
def _browser_go_back() -> str:
    return _controller.go_back()


@tool(
    name="read_browser_page",
    description="Read the current page's text content (first ~1500 chars).",
    params={"type": "object", "properties": {}},
    aliases=["read_page"],
)
def _read_page() -> str:
    return _controller.read_page()


@tool(
    name="browser_close",
    description="Close the assistant's browser.",
    params={"type": "object", "properties": {}},
)
def _browser_close() -> str:
    _controller.close()
    return "browser closed"
