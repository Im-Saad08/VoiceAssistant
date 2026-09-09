"""User-confirmation flow for risky actions.

Speaks the question aloud and accepts a yes/no answer either by voice
(English / Urdu / Hinglish) or by typing at the prompt. Defaults to a safe
"No" whenever the answer is unclear.
"""

from __future__ import annotations

from audio import speech_input, tts
from assistant.logging_setup import get_logger

logger = get_logger("confirm")

_YES = {"yes", "yep", "yeah", "sure", "ok", "okay", "go", "do it", "affirmative", "correct", "right",
        "han", "haan", "hanji", "ji", "theek", "theek hai", "sahi", "sahi hai", "hng", "mm"}
_NO = {"no", "nope", "nahi", "nai", "na", "nahin", "nhi", "cancel", "stop", "mat karo", "not"}


def _parse_answer(text: str) -> bool | None:
    words = " ".join(text.lower().split())
    if any(w in words for w in _NO):
        return False
    if any(w in words for w in _YES):
        return True
    return None


def confirm(question: str) -> bool:
    """Ask for confirmation; returns True only if the user clearly agrees."""
    for attempt in range(2):
        tts.speak(question)
        print(f"\n⚠  {question}")
        print("    (Say or type yes / no)")

        spoken = speech_input.listen()
        if spoken:
            answer = _parse_answer(spoken)
            if answer is not None:
                logger.info("confirmation answer (voice): %r", spoken)
                return answer

        try:
            typed = input("    > ").strip()
            if typed:
                answer = _parse_answer(typed)
                if answer is not None:
                    logger.info("confirmation answer (typed): %r", typed)
                    return answer
        except EOFError:
            return False

        if attempt == 0:
            print("    Sorry, I didn't catch that. Please say or type yes / no.")

    print("    No confirmation received; not proceeding.")
    return False
