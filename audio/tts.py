"""Text-to-speech via Microsoft Edge TTS.

Synthesizes to a unique temporary MP3 (so it never collides with or locks a
reused filename), plays it directly, and removes it afterward. Markdown is
stripped before synthesis so symbols are never spoken aloud.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

from assistant.config import TTS_TIMEOUT_SECONDS, TTS_VOICE
from assistant.logging_setup import get_logger

logger = get_logger("tts")


def clean_for_speech(text: str) -> str:
    """Strip Markdown and other symbols so they are not spoken literally."""
    if not text:
        return ""
    # Code blocks.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = text.replace("`", "")
    # Inline formatting.
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("_", " ")
    # Headings and bullets.
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)
    # Markdown links "text (url)" -> keep the label only. Do this BEFORE
    # stripping bare URLs, otherwise the URL regex eats the closing ")".
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    # Remaining bare URLs, pipes, tildes, blockquotes.
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("|", " ").replace("~", " ").replace(">", " ")
    # Collapse whitespace.
    return re.sub(r"\s+", " ", text).strip()


def _synthesize_blocking(text: str) -> str:
    """Synthesize `text` to a temp MP3 path, blocking the caller.

    edge-tts is async while the surrounding API is synchronous, so we bridge
    the two. asyncio.run() is the right bridge from a thread/process with no
    running loop, but it raises
        RuntimeError: asyncio.run() cannot be called from a running event loop
    if the calling thread already owns one (interactive/embedded hosts, or a
    caller that is itself async). In that case we run the coroutine in a
    short-lived worker thread that owns its own loop, then block on its result.
    The blocking behavior is identical either way.
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return asyncio.run(_synthesize(text))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(_synthesize(text)))
        return future.result(timeout=TTS_TIMEOUT_SECONDS + 15.0)


async def _synthesize(text: str) -> str:
    import edge_tts

    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        await asyncio.wait_for(
            edge_tts.Communicate(text, TTS_VOICE).save(path),
            timeout=TTS_TIMEOUT_SECONDS,
        )
        return path
    except Exception:  # noqa: BLE001
        try:
            os.remove(path)
        except OSError:
            pass
        raise


def render(text: str) -> str | None:
    """Synthesize `text` to a temp MP3 and return its path (or None on failure).
    Playback not started — call play() to play the returned path.
    """
    speech = clean_for_speech(text)
    if not speech:
        return None
    try:
        return _synthesize_blocking(speech)
    except Exception:  # noqa: BLE001
        logger.exception("TTS render failed")
        return None


def play(path: str) -> None:
    """Play a pre-rendered MP3 and remove it when done."""
    from audio import playback

    try:
        playback.play_mp3(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            logger.debug("could not remove temp audio (non-fatal)")


def speak(text: str) -> bool:
    """Synthesize + play `text` sequentially. Returns True on success."""
    path = render(text)
    if not path:
        return False
    play(path)
    return True
