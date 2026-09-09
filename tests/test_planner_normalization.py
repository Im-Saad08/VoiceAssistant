"""Tests for the deterministic planner-side normalization layer.

Lock in the invariants that NO LLM size may violate:

  - open_application on a known website target  -> open_url (mapped URL)
  - bare open_url site names                     -> open_url (full URL)
  - browser_open_result with a missing index     -> index 1 (when results exist)
  - response_language is constrained to {en, ur, hinglish}
  - bare "go back" phrases on an OPEN browser    -> browser_go_back

All exact-match allowlists: unknown targets are left for the router to reject,
never guessed here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import planner  # noqa: E402
from assistant.planner import (  # noqa: E402
    _apply_browser_back_fallback,
    _finalize_plan,
    _search_query,
)
from assistant.planner_utils import (  # noqa: E402
    _canonical_app_target,
    _extract_open_target,
    _normalize_app_targets,
    _ordinal_result_index,
    _resolve_intents,
    _route_actions,
    _website_url,
)

_OPEN_BROWSER = {
    "open": True,
    "url": "https://www.youtube.com",
    "has_results": False,
    "result_count": 0,
    "last_search": None,
}
_CLOSED_BROWSER = {"open": False, "has_results": False, "result_count": 0}


# ---- _website_url -----------------------------------------------------------

def test_website_url_known_sites():
    assert _website_url("youtube") == "https://www.youtube.com"
    assert _website_url("gmail") == "https://mail.google.com"
    assert _website_url("twitter") == "https://x.com"


def test_website_url_variants():
    assert _website_url("www.youtube.com") == "https://www.youtube.com"
    assert _website_url("https://mail.google.com") == "https://mail.google.com"
    assert _website_url("gmail.com") == "https://mail.google.com"


def test_website_url_refuses_desktop_apps_and_unknowns():
    assert _website_url("chrome") is None
    assert _website_url("notepad") is None
    assert _website_url("calc") is None
    assert _website_url("whatever") is None


# ---- _route_actions ---------------------------------------------------------

def test_route_rewrites_website_open_application():
    out = _route_actions([{"type": "open_application", "params": {"target": "youtube"}}])
    assert out == [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}]


def test_route_keeps_desktop_open_application():
    actions = [{"type": "open_application", "params": {"target": "chrome"}}]
    assert _route_actions(actions) == actions
    actions = [{"type": "open_application", "params": {"target": "notepad"}}]
    assert _route_actions(actions) == actions


def test_route_expands_bare_open_url_name():
    out = _route_actions([{"type": "open_url", "params": {"url": "gmail"}}])
    assert out == [{"type": "open_url", "params": {"url": "https://mail.google.com"}}]


def test_route_leaves_real_urls_untouched():
    actions = [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}]
    assert _route_actions(actions) == actions
    actions = [{"type": "open_url", "params": {"url": "youtube.com"}}]
    assert _route_actions(actions) == actions


def test_route_defaults_missing_result_index_when_results_exist():
    st = dict(_OPEN_BROWSER, has_results=True, result_count=3)
    out = _route_actions(
        [{"type": "browser_open_result", "params": {}}], st
    )
    assert out == [{"type": "browser_open_result", "params": {"index": 1}}]


def test_route_keeps_explicit_result_index():
    st = dict(_OPEN_BROWSER, has_results=True, result_count=3)
    actions = [{"type": "browser_open_result", "params": {"index": 2}}]
    assert _route_actions(actions, st) == actions


def test_route_does_not_guess_index_without_results():
    # No results -> leaving index missing lets the router reject it truthfully.
    out = _route_actions(
        [{"type": "browser_open_result", "params": {}}], _OPEN_BROWSER
    )
    assert out == [{"type": "browser_open_result", "params": {}}]


def test_route_never_rewrites_unknown_targets():
    actions = [{"type": "open_application", "params": {"target": "steam"}}]
    assert _route_actions(actions) == actions


# ---- response_language coercion ---------------------------------------------

def _finalize(raw, utterance="hi", browser_state=None):
    return _finalize_plan(
        {"spoken_response": "ok", "actions": [], "response_language": raw},
        utterance,
        browser_state,
    )


def test_language_keeps_supported_values():
    for lang in ("en", "ur", "hinglish", "EN", "  Ur  "):
        assert _finalize(lang)["response_language"] in ("en", "ur", "hinglish")


def test_language_unknown_defaults_to_en():
    for bad in ("fr", "es", "", None, "deutsch", "xyz"):
        assert _finalize(bad)["response_language"] == "en"


def test_urdu_script_response_language_forced_ur():
    # Even if the LLM tags an Arabic-script request "en", script detection is
    # deterministic: the user wrote Urdu, so the response is Urdu.
    for raw_lang in ("en", "ur", "fr"):
        out = _finalize_plan(
            {"spoken_response": "ok", "actions": [], "response_language": raw_lang},
            "Notepad کھول دو",
            _CLOSED_BROWSER,
        )
        assert out["response_language"] == "ur"
    # Roman/Hinglish input (no script) keeps the LLM's choice.
    assert _finalize("en", utterance="youtube kholo")["response_language"] == "en"


# ---- browser-back fallback --------------------------------------------------

def test_bare_go_back_with_open_browser():
    p = {"spoken_response": "ok", "actions": []}
    _apply_browser_back_fallback(p, "go back", _OPEN_BROWSER)
    assert p["actions"] == [{"type": "browser_go_back", "params": {}}]


def test_bare_go_back_variants():
    for phrase in ("back", "go back please", "back please", "peeche jao", "wapis jao"):
        p = {"spoken_response": "ok", "actions": []}
        _apply_browser_back_fallback(p, phrase, _OPEN_BROWSER)
        assert p["actions"] == [{"type": "browser_go_back", "params": {}}]


def test_go_back_no_op_when_browser_closed():
    p = {"spoken_response": "ok", "actions": []}
    _apply_browser_back_fallback(p, "go back", _CLOSED_BROWSER)
    assert p["actions"] == []


def test_go_back_no_op_when_plan_has_actions():
    p = {"spoken_response": "ok", "actions": [{"type": "get_time", "params": {}}]}
    _apply_browser_back_fallback(p, "go back", _OPEN_BROWSER)
    assert p["actions"] == [{"type": "get_time", "params": {}}]


def test_go_back_no_op_for_random_chatter():
    p = {"spoken_response": "ok", "actions": []}
    _apply_browser_back_fallback(p, "what is the weather like", _OPEN_BROWSER)
    assert p["actions"] == []


def test_go_back_strips_trailing_punctuation():
    p = {"spoken_response": "ok", "actions": []}
    _apply_browser_back_fallback(p, "go back.", _OPEN_BROWSER)
    assert p["actions"] == [{"type": "browser_go_back", "params": {}}]


def test_go_back_not_repeated():
    p = {"spoken_response": "ok", "actions": []}
    _apply_browser_back_fallback(p, "go back go back", _OPEN_BROWSER)
    assert p["actions"] == []


# ---- _finalize_plan end-to-end ----------------------------------------------

def test_finalize_plan_full_pipeline():
    """Contract -> language -> routing -> back-fallback all applied."""
    result = {
        "spoken_response": "Opening Gmail.",
        "actions": [{"type": "open_application", "params": {"target": "gmail"}}],
        "response_language": "fr",
    }
    out = _finalize_plan(result, "open gmail", browser_state=_CLOSED_BROWSER)
    assert out["response_language"] == "en"
    assert out["actions"] == [
        {"type": "open_url", "params": {"url": "https://mail.google.com"}}
    ]


def test_finalize_plan_back_fallback_only_when_no_actions():
    result = {
        "spoken_response": "Going back.",
        "actions": [],
        "response_language": "en",
    }
    out = _finalize_plan(result, "go back", browser_state=_OPEN_BROWSER)
    assert out["actions"] == [{"type": "browser_go_back", "params": {}}]


# ---- dispatcher plan() wires browser_state into routing ---------------------

def test_plan_dispatcher_routes_website_for_both_backends():
    """plan() must apply routing the same way regardless of the active backend."""
    import assistant.config as cfg
    from assistant import planner_gemini, planner_ollama

    orig_backend = cfg.PLANNER_BACKEND
    orig_ollama_plan = planner_ollama.plan
    orig_gemini_plan = planner_gemini.plan

    def _fake_gemini_plan(text, ctx="", **_):
        return {
            "spoken_response": "Opening Gmail.",
            "actions": [{"type": "open_application", "params": {"target": "gmail"}}],
            "response_language": "en",
        }

    try:
        # Ollama backend.
        cfg.PLANNER_BACKEND = "ollama"
        planner_ollama.plan = _fake_gemini_plan
        out_ollama = planner.plan("open gmail", browser_state=_CLOSED_BROWSER)
        assert out_ollama["actions"] == [
            {"type": "open_url", "params": {"url": "https://mail.google.com"}}
        ]

        # Gemini backend.
        cfg.PLANNER_BACKEND = "gemini"
        planner_gemini.plan = _fake_gemini_plan
        out_gemini = planner.plan("open gmail", browser_state=_CLOSED_BROWSER)
        assert out_gemini["actions"] == [
            {"type": "open_url", "params": {"url": "https://mail.google.com"}}
        ]
    finally:
        cfg.PLANNER_BACKEND = orig_backend
        planner_ollama.plan = orig_ollama_plan
        planner_gemini.plan = orig_gemini_plan


# ---- Search-intent query extraction ------------------------------------------

def test_search_query_extracts_query_from_verb():
    assert _search_query("Search for Python tutorials") == "Python tutorials"
    assert _search_query("search python tutorials") == "python tutorials"
    assert _search_query("look up pandas docs") == "pandas docs"
    assert _search_query("look for tricks") == "tricks"
    assert _search_query("find pets") == "pets"
    assert _search_query("google python") == "python"
    assert _search_query("dhundho python") == "python"
    assert _search_query("khojo result") == "result"


def test_search_query_non_search_chatter_is_none():
    assert _search_query("hello there") is None
    assert _search_query("open notepad") is None
    assert _search_query("what time is it") is None


def test_search_query_google_chrome_means_browser_not_search():
    assert _search_query("google chrome kholo") is None


def test_search_query_lone_verb_is_none():
    assert _search_query("search") is None
    assert _search_query("search for") is None


# ---- Search-intent override --------------------------------------------------

def test_search_override_rescues_navigation_only_plan():
    result = {
        "spoken_response": "Opening YouTube.",
        "actions": [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}],
        "response_language": "en",
    }
    out = _finalize_plan(result, "Search for Python tutorials", _CLOSED_BROWSER)
    assert out["actions"] == [{"type": "browser_search", "params": {"query": "Python tutorials"}}]


def test_search_override_fills_empty_plan():
    out = _finalize_plan(
        {"spoken_response": "ok", "actions": [], "response_language": "en"},
        "search python tutorials",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == [{"type": "browser_search", "params": {"query": "python tutorials"}}]


def test_search_override_preserves_plan_that_already_searches():
    actions = [{"type": "browser_search", "params": {"query": "Python tutorials"}}]
    out = _finalize_plan(
        {"spoken_response": "ok", "actions": actions, "response_language": "en"},
        "Search for Python tutorials",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == actions


def test_search_override_preserves_non_navigation_tools():
    # "find my files" -> plan already chose file_search; never clobber it.
    actions = [{"type": "file_search", "params": {"query": "my files"}}]
    out = _finalize_plan(
        {"spoken_response": "ok", "actions": actions, "response_language": "en"},
        "find my files",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == actions


def test_search_override_ignores_google_chrome():
    actions = [{"type": "open_application", "params": {"target": "chrome"}}]
    out = _finalize_plan(
        {"spoken_response": "ok", "actions": actions, "response_language": "en"},
        "google chrome kholo",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == actions


def test_search_override_unaffected_by_normal_chatter():
    actions = [{"type": "get_time", "params": {}}]
    out = _finalize_plan(
        {"spoken_response": "ok", "actions": actions, "response_language": "en"},
        "what time is it",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == actions


# ---- Website fallback for fully-failed plans ---------------------------------

def test_website_fallback_rescues_empty_plan_go_to():
    out = _finalize_plan(
        {"spoken_response": "I had trouble understanding.", "actions": [], "response_language": "en"},
        "Go to YouTube",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}]
    # The failure text is replaced with an honest action description.
    assert "trouble" not in out["spoken_response"]


def test_website_fallback_rescues_empty_plan_open_site():
    out = _finalize_plan(
        {"spoken_response": "I had trouble understanding.", "actions": [], "response_language": "en"},
        "open gmail",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == [{"type": "open_url", "params": {"url": "https://mail.google.com"}}]


def test_open_notepad_empty_plan_still_deterministically_opens():
    # The website fallback never opens desktop apps — but the deterministic
    # intent layer DOES resolve an obvious "Open Notepad" into an app open.
    out = _finalize_plan(
        {"spoken_response": "I had trouble understanding.", "actions": [], "response_language": "en"},
        "Open Notepad",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == [{"type": "open_application", "params": {"target": "notepad"}}]
    assert "trouble" not in out["spoken_response"]


def test_website_open_prepends_even_when_plan_has_unrelated_action():
    # The website fallback only fires on an empty plan, but the deterministic
    # intent layer still increases an unrelated-model-action plan with the
    # obvious "open gmail" -> the user asked for gmail, so gmail gets opened.
    actions = [{"type": "get_time", "params": {}}]
    out = _finalize_plan(
        {"spoken_response": "ok", "actions": actions, "response_language": "en"},
        "open gmail",
        _CLOSED_BROWSER,
    )
    assert out["actions"] == [
        {"type": "open_url", "params": {"url": "https://mail.google.com"}},
        {"type": "get_time", "params": {}},
    ]


def test_website_fallback_back_phrase_still_wins():
    out = _finalize_plan(
        {"spoken_response": "I had trouble understanding.", "actions": [], "response_language": "en"},
        "go back",
        _OPEN_BROWSER,
    )
    assert out["actions"] == [{"type": "browser_go_back", "params": {}}]


# ---- Deterministic intent layer (open-target / calculator / result-index) ----

_CALC_CTX = {"expression": "25*13", "value": 325, "opened": True}
_OPEN_BROWSER_RESULTS = dict(_OPEN_BROWSER, has_results=True, result_count=3)


def _resolved(utterance, actions, browser_state=_CLOSED_BROWSER, calc=None):
    return _finalize_plan(
        {
            "spoken_response": "ok",
            "actions": list(actions),
            "response_language": "en",
        },
        utterance,
        browser_state=browser_state,
        calculator_state=calc,
    )


def _types(out):
    return [a["type"] for a in out["actions"]]


def test_open_chrome_stays_open_application():
    out = _resolved("open chrome", [{"type": "open_application", "params": {"target": "chrome"}}])
    assert _types(out) == ["open_application"]
    assert out["actions"][0]["params"]["target"] == "chrome"


def test_open_youtube_is_open_url_not_search():
    out = _resolved("open youtube", [])
    assert _types(out) == ["open_url"]
    assert out["actions"][0]["params"]["url"] == "https://www.youtube.com"


def test_open_youtube_model_wrongly_searches_is_corrected():
    # Real qwen behavior: browser_search("YouTube") for "open YouTube".
    out = _resolved(
        "open youtube",
        [{"type": "browser_search", "params": {"query": "YouTube"}}],
    )
    assert _types(out) == ["open_url"]
    assert out["actions"][0]["params"]["url"] == "https://www.youtube.com"


def test_open_youtube_model_wrong_page_is_corrected():
    # Model said it would open YouTube but navigated Bing — must not stand.
    out = _resolved(
        "open youtube",
        [{"type": "open_url", "params": {"url": "https://www.bing.com"}}],
    )
    assert _types(out) == ["open_url"]
    assert out["actions"][0]["params"]["url"] == "https://www.youtube.com"


def test_open_youtube_model_open_application_site_is_corrected():
    out = _resolved(
        "open youtube",
        [{"type": "open_application", "params": {"target": "youtube"}}],
    )
    assert out["actions"] == [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}]


def test_youtube_kholo_hinglish_opens():
    for phrase in ("youtube kholo", "youtube open karo", "browser mein youtube kholo"):
        out = _resolved(phrase, [])
        assert _types(out) == ["open_url"], phrase
        assert out["actions"][0]["params"]["url"] == "https://www.youtube.com"


def test_explicit_known_domain_normalized():
    for phrase in ("open youtube.com", "go to www.youtube.com", "open https://youtube.com"):
        out = _resolved(phrase, [])
        assert _types(out) == ["open_url"], phrase
        assert out["actions"][0]["params"]["url"] == "https://www.youtube.com"


def test_explicit_unknown_domain_still_opens_directly():
    out = _resolved("open wikipedia.org", [])
    assert _types(out) == ["open_url"]
    assert out["actions"][0]["params"]["url"] in (
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://wikipedia.org",
    )


def test_politeness_and_rephrase_still_navigate():
    for phrase in ("Can you get YouTube up?", "Take me to YouTube", "Put YouTube on the browser"):
        out = _resolved(phrase, [])
        assert _types(out) == ["open_url"], phrase
        assert out["actions"][0]["params"]["url"] == "https://www.youtube.com"


def test_search_intent_remains_a_search_on_any_browser_state():
    out = _resolved("search for Python tutorials", [])
    assert _types(out) == ["browser_search"]
    assert out["actions"][0]["params"]["query"] == "Python tutorials"


def test_open_calculator_is_application_not_browser():
    for phrase in ("open calculator", "calculator open karo", "calculator kholo"):
        out = _resolved(phrase, [])
        assert _types(out) == ["open_application"], phrase
        assert out["actions"][0]["params"]["target"] == "calculator"


def test_calculator_task_never_opens_browser():
    # Even with an empty plan, a calculator-frame must NOT fall back to the
    # browser (no about:blank, no open_application chrome, no open_url).
    out = _resolved("25 ko 13 se multiply karo calculator pe", [])
    assert _types(out) == ["calculator_perform"]
    assert out["actions"][0]["params"]["expression"] == "25*13"


def test_calculator_mixed_stem_phrasing():
    out = _resolved("calculator pe 25 multiply 13 karo", [])
    assert _types(out) == ["calculator_perform"]
    assert out["actions"][0]["params"]["expression"] == "25*13"


def test_math_question_is_answered_not_displayed():
    out = _resolved("what is 25 times 13", [])
    assert _types(out) == ["calculate"]
    assert out["actions"][0]["params"]["expression"] == "25*13"


def test_math_infix_and_verb_first_variants():
    for phrase in ("multiply 25 by 13", "25 multiplied by 13", "25 x 13"):
        out = _resolved(phrase, [])
        assert _types(out) == ["calculate"], phrase
        assert out["actions"][0]["params"]["expression"] == "25*13"


def test_deictic_calculator_with_context_uses_previous_expression():
    calc = {"expression": "25*13", "value": 325, "opened": True}
    for phrase in ("do it there", "show that on the calculator", "do that"):
        out = _resolved(phrase, [], calc=calc)
        assert _types(out) == ["calculator_perform"], phrase
        assert out["actions"][0]["params"]["expression"] == "25*13"


def test_deictic_calculator_without_context_is_noop_never_browser():
    out = _resolved("do it there", [])
    assert out["actions"] == []


def test_open_first_result_uses_browser_context():
    out = _resolved("open the first result", [], browser_state=_OPEN_BROWSER_RESULTS)
    assert _types(out) == ["browser_open_result"]
    assert out["actions"][0]["params"]["index"] == 1


def test_ordinal_results_hindi():
    for phrase, idx in (("pehla result kholo", 1), ("dusra result kholo", 2)):
        out = _resolved(phrase, [], browser_state=_OPEN_BROWSER_RESULTS)
        assert _types(out) == ["browser_open_result"], phrase
        assert out["actions"][0]["params"]["index"] == idx


def test_google_chrome_alias_canonicalizes_before_execution():
    assert _canonical_app_target("google-chrome") == "chrome"
    assert _canonical_app_target("Google Chrome") == "chrome"
    assert _canonical_app_target("chrome browser") == "chrome"
    assert _canonical_app_target("browser") == "chrome"
    actions = [{"type": "open_application", "params": {"target": "google-chrome"}}]
    assert _normalize_app_targets(actions) == [{
        "type": "open_application",
        "params": {"target": "chrome"},
    }]


def test_google_chrome_routes_to_open_application():
    for phrase in ("open google chrome", "Google Chrome kholo", "chrome browser kholo"):
        out = _resolved(phrase, [])
        assert _types(out) == ["open_application"], phrase
        assert out["actions"][0]["params"]["target"] == "chrome"


def test_empty_plan_after_failure_never_reuses_previous_error():
    # Regression: turn N's failure ("unknown application: calculator") must
    # never resurface as turn N+1's action/error state.
    out = _resolved("hello there", [])
    assert out["actions"] == []
    assert "calculator" not in out["spoken_response"].lower()


def test_open_youtube_while_browser_open_reuses_no_second_launch():
    st = dict(_OPEN_BROWSER, url="https://www.bing.com")
    out = _resolved(
        "open youtube",
        [{"type": "open_application", "params": {"target": "chrome"}}],
        browser_state=st,
    )
    assert _types(out) == ["open_url"]  # the redundant chrome launch is dropped
    assert out["actions"][0]["params"]["url"] == "https://www.youtube.com"


def test_open_youtube_while_browser_at_youtube_is_noop_change():
    st = dict(_OPEN_BROWSER, url="https://www.youtube.com")
    out = _resolved(
        "open youtube",
        [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}],
        browser_state=st,
    )
    assert out["actions"] == [{"type": "open_url", "params": {"url": "https://www.youtube.com"}}]


def test_calculator_perform_never_yields_browser_search():
    out = _resolved(
        "25 ko 13 se multiply karo calculator pe",
        [{"type": "browser_search", "params": {"query": "25*13"}}],
    )
    assert out["actions"] == [{"type": "calculator_perform", "params": {"expression": "25*13"}}]


def test_open_target_extraction_direct():
    assert _extract_open_target("open youtube") == (
        "site", "https://www.youtube.com", "YouTube",
    )
    assert _extract_open_target("open notepad") == ("app", "notepad", "Notepad")
    assert _extract_open_target("open the first result") is None
    assert _extract_open_target("go back") is None


def test_ordinal_index_direct():
    assert _ordinal_result_index("open the first result") == 1
    assert _ordinal_result_index("the second result") == 2
    assert _ordinal_result_index("pehla result") == 1
    assert _ordinal_result_index("what time is it") is None


def test_open_notepad_never_becomes_website_open():
    out = _resolved("open notepad", [{"type": "browser_search", "params": {"query": "notepad"}}])
    assert out["actions"] == [{"type": "open_application", "params": {"target": "notepad"}}]


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("planner normalization: all tests passed")


if __name__ == "__main__":
    _main()