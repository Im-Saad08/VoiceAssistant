"""Offline unit tests for audio/tts.py.

No network, no audio device, no edge-tts — `_synthesize` is stubbed with an
async fake that returns a fixed path. These lock in the event-loop bridge:

  - from a thread with NO running loop, _synthesize_blocking just works;
  - from a thread WITH a running loop, the old code raised
        RuntimeError: asyncio.run() cannot be called from a running event loop
    and the fix (worker-thread offload) must return the same result.

Run:  python tests/test_tts.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio import tts  # noqa: E402

FAKE_PATH = "fake_audio.mp3"


async def _stub_synthesize(text: str) -> str:
    """Async stand-in for tts._synthesize — no edge-tts, no network."""
    return FAKE_PATH


def _patch_synthesize():
    original = tts._synthesize
    tts._synthesize = _stub_synthesize  # type: ignore[assignment]
    return original


def test_synthesize_blocking_no_running_loop():
    """Plain synchronous call (no loop running): blocking bridge returns path."""
    original = _patch_synthesize()
    try:
        path = tts._synthesize_blocking("Hello")
        assert path == FAKE_PATH
    finally:
        tts._synthesize = original


async def _call_blocking_inside_running_loop() -> str:
    """Called from inside an event loop — the exact live failure scenario."""
    return tts._synthesize_blocking("Hello from a loop")


def test_synthesize_blocking_with_running_loop_does_not_raise():
    """Regression: must NOT raise RuntimeError from a running event loop."""
    original = _patch_synthesize()
    try:
        path = asyncio.run(_call_blocking_inside_running_loop())
        assert path == FAKE_PATH
    finally:
        tts._synthesize = original


def test_render_empty_speech_returns_none():
    """Empty / markdown-only text is not synthesized at all."""
    assert tts.render("") is None
    assert tts.render("   ``` ```   ") is None


def test_render_returns_synthesized_path():
    """render() calls the blocking bridge and returns the temp path."""
    original = _patch_synthesize()
    try:
        path = tts.render("Hello there.")
        assert path == FAKE_PATH
    finally:
        tts._synthesize = original


def _main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("tts: all tests passed")


if __name__ == "__main__":
    _main()