"""Tests for TTS text cleaning (no Markdown symbols spoken aloud)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.tts import clean_for_speech  # noqa: E402


def test_strips_bold():
    assert clean_for_speech("**Sure!** I'll open Chrome.") == "Sure! I'll open Chrome."


def test_strips_code_blocks():
    assert clean_for_speech("Here is the code:\n```py\nprint(1)\n```\nDone.") == "Here is the code: Done."


def test_strips_inline_code_and_backticks():
    assert clean_for_speech("Use `print()` to output.") == "Use print() to output."


def test_strips_markdown_links_keeps_text():
    assert clean_for_speech("See [Python](https://python.org) docs.") == "See Python docs."


def test_strips_urls_and_pipes():
    assert clean_for_speech("Go to https://example.com | now") == "Go to now"


def test_strips_hashtags():
    assert clean_for_speech("## Heading\nBody text.") == "Heading Body text."


def test_collapses_whitespace():
    assert clean_for_speech("a   b\n\n  c") == "a b c"


def test_empty_input():
    assert clean_for_speech("") == ""


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("cleantext: all tests passed")


if __name__ == "__main__":
    _main()
