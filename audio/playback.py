"""Audio playback via pygame.mixer.

Initialized once at import. pygame must not hold an MP3 open while it is
being deleted, so playback always unloads the track before returning.
"""

from __future__ import annotations

import time

from assistant.config import BASE_DIR
from assistant.logging_setup import get_logger

logger = get_logger("playback")

_mixer = None


def init_mixer() -> None:
    """Initialize the mixer once (safe to call repeatedly)."""
    global _mixer
    if _mixer is not None:
        return
    import pygame

    try:
        pygame.mixer.init()
        _mixer = True
        logger.debug("mixer initialized at %s", pygame.mixer.get_init())
    except Exception:  # noqa: BLE001
        logger.exception("pygame mixer init failed (no audio device?)")


def play_mp3(path: str) -> None:
    """Blockingly play an MP3 file, releasing it before returning."""
    import pygame

    init_mixer()
    if _mixer is None:
        logger.warning("no audio device; skipping playback")
        return
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
    finally:
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.unload()
        except Exception:  # noqa: BLE001
            logger.debug("unload failed (non-fatal)", exc_info=True)
