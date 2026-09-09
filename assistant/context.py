"""Short-term conversational memory.

Keeps the recent turn history and a small set of actionable facts (last
search results, last opened app, detected language) so the planner can resolve
follow-ups like "open the first result" without the user repeating context.
Designed so a persistent store can be swapped in later without touching callers.
"""

from __future__ import annotations

from collections import deque

from assistant.logging_setup import get_logger

logger = get_logger("context")

MAX_TURNS = 8


class ConversationContext:
    def __init__(self) -> None:
        self.turns: deque[dict] = deque(maxlen=MAX_TURNS)
        self.language: str | None = None  # last detected response language
        self.last_action_summary: str | None = None

    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "text": text})

    def add_assistant(self, text: str) -> None:
        self.turns.append({"role": "assistant", "text": text})

    def set_language(self, lang: str | None) -> None:
        if lang:
            self.language = lang

    def history(self) -> list[dict]:
        return list(self.turns)

    def summary(self) -> str:
        """A compact block describing prior context for the planner prompt."""
        lines = []
        for turn in self.turns:
            speaker = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {turn['text']}")
        block = "\n".join(lines[-6:])
        extra = []
        if self.language:
            extra.append(f"Preferred response language: {self.language}")
        if self.last_action_summary:
            extra.append(f"Last action result: {self.last_action_summary}")
        if extra:
            block = block + "\n" + "\n".join(extra)
        return block
