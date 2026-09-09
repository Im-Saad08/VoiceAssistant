"""Voice Assistant — entry point.

Push-to-talk with a typed-input fallback:

  - Press ENTER (empty line) to talk via microphone.
  - Type a message and press ENTER to send text directly.
  - Type 'quit' / 'exit' / 'goodbye' to stop.

Useful for fast iteration and testing even when no mic is available.
"""

from __future__ import annotations

import sys

from assistant.logging_setup import get_logger
from assistant.orchestrator import reset, run_turn

logger = get_logger("main")

_EXIT = {"quit", "exit", "stop", "goodbye", "bye", "band karo"}


def main() -> None:
    reset()
    print("╔═══════════════════════════════════╗")
    print("║       Voice Assistant             ║")
    print("╚═══════════════════════════════════╝")
    print("  Press ENTER to talk via microphone.")
    print("  Or type a message and press ENTER.")
    print("  Say 'goodbye' or type quit to stop.\n")

    while True:
        try:
            user_input = input("You ▸ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nStopped.")
            break

        if not user_input:
            # Empty line → listen via microphone.
            try:
                from audio.speech_input import listen

                user_input = (listen() or "").strip()
            except Exception:  # noqa: BLE001
                logger.exception("microphone listen failed")
                print("  Could not listen. Try typing your message.\n")
                continue

        if not user_input:
            print("  (no speech detected)\n")
            continue

        if user_input.lower() in _EXIT:
            # Exit is handled locally (no Gemini call needed).
            print("Goodbye!")
            try:
                from audio.tts import speak

                speak("Goodbye.")
            except Exception:
                pass
            break

        response = run_turn(user_input)
        if response:
            print(f"  {response}\n")
        else:
            print()

    # Clean shutdown.
    try:
        from tools.browser import close_browser

        close_browser()
    except Exception:
        pass


if __name__ == "__main__":
    main()
