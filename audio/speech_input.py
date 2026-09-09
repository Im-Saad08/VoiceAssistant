"""Microphone capture and speech recognition."""

from __future__ import annotations

import time

import speech_recognition as sr

from assistant.config import MIC_PHRASE_LIMIT_SECONDS, MIC_TIMEOUT_SECONDS
from assistant.logging_setup import get_logger

logger = get_logger("speech")

_recognizer = sr.Recognizer()


def listen() -> str | None:
    """Capture one phrase from the microphone and recognize it.

    Returns the recognized text, or None if nothing/undecipherable was heard.
    """
    start = time.perf_counter()
    try:
        with sr.Microphone() as source:
            # Brief ambient-noise calibration on each open keeps it responsive
            # without being heavy (0.4s).
            _recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = _recognizer.listen(
                source,
                timeout=MIC_TIMEOUT_SECONDS,
                phrase_time_limit=MIC_PHRASE_LIMIT_SECONDS,
            )
    except sr.WaitTimeoutError:
        logger.info("no speech detected (timeout)")
        return None
    except Exception:  # noqa: BLE001
        logger.exception("microphone capture failed")
        return None

    capture_s = time.perf_counter() - start

    try:
        text = _recognizer.recognize_google(audio)
        logger.info(
            "recognized (%.2fs capture): %s", capture_s, text
        )
        return text.strip()
    except sr.UnknownValueError:
        logger.info("speech not understood")
        return None
    except sr.RequestError as exc:  # noqa: BLE001
        logger.error("recognition service error: %s", exc)
        return None
