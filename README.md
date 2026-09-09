# Voice Assistant (JARVIS-style)

A lightweight, modular, multilingual desktop voice assistant. You speak naturally
(English / Urdu / Hinglish); it understands what you mean, decides what to do,
operates your computer safely, and keeps you informed aloud.

The **LLM only decides what should happen** — Python validates every planned
action against an allowlisted tool registry before anything executes. The LLM
can never run arbitrary code.

## Run it

```bash
cd "E:\Python Projects\VoiceAssistant"
.\.venv\Scripts\python.exe main.py
```

**Usage:**
- Press **Enter** (empty line) → talk into the microphone.
- Type a message + Enter → send text directly (handy for testing without a mic).
- Type `quit` / `exit` / `goodbye` (or say it) to stop.

Make sure `.env` contains `GEMINI_API_KEY=...`.

## Architecture

```
main.py                  # push-to-talk loop (voice + typed fallback)
assistant/
  config.py              # all settings (model, voice, timing, paths)
  logging_setup.py       # console + rotating file logs (./logs)
  planner.py             # Gemini → structured plan (JSON schema, union-typed)
  router.py              # validate + execute actions safely
  context.py             # short-term conversational memory
  orchestrator.py        # one-turn pipeline: plan → route → respond
audio/
  speech_input.py        # mic capture + recognition (auto language)
  tts.py                 # Edge TTS (unique temp file, no locking) + markdown clean
  playback.py            # pygame playback
tools/
  registry.py            # @tool decorator + allowlist + validation
  applications.py        # open/close Notepad, Calculator, Chrome, ...
  browser.py             # Playwright browser automation (search, navigate, click)
  system.py              # get_time, get_date
  files.py               # file_search, file_open, delete_path (confirm)
safety/
  permissions.py         # auto / confirm safety levels
  confirmations.py       # spoken+typed yes/no confirmation flow
tests/                   # offline + planner integration tests
```

## How a turn flows

```
You:  "Open Chrome and go to YouTube"
  1. recognize speech
  2. planner (Gemini) → { spoken_response, actions: [...] }
  3. router validates each action against the registry (types, required params, safety)
  4. auto actions run; confirm actions ask you first
  5. assistant speaks progress + the response
```

## Multilingual

Gemini detects the language and replies in kind (English → en, Urdu script → ur,
Hinglish → hinglish). There is no hardcoded command list — intent is understood
from natural language, so "Open Notepad", "Can you get the text editor up?",
and "یار Notepad کھول دو" all map to the same action.

## Safety

- **Auto** actions run immediately (open app, navigate, search, get time, ...).
- **Confirm** actions require your spoken/typed "yes" first (delete, shutdown,
  messaging, ...). Unknown/unsafe action types are rejected outright.
- The planner is told which tools exist and may **only** use those.

## Latency notes

- Push-to-talk (no always-on wake word) keeps CPU low on a dual-core PC.
- Gemini is called once per turn; TTS is generated per response.
- Free-tier Gemini is **rate-limited (~20 req/min, and a low daily cap)** — heavy
  testing can hit 429s. The planner retries with the API's suggested backoff,
  but for smooth use consider a paid/billed tier or spacing out requests.

## Known limitations (v1)

- "Open Chrome" drives the assistant's **own Playwright browser** (persistent
  profile in `.browser_profile/`) so search/navigate/click all share one
  session. It does not connect to your personal Chrome with your logins yet
  (CDP connection is a future option).
- Browser `search` extracts result links; quality varies by site.
- WhatsApp is **not implemented** — the tool registry makes adding it a small,
  self-contained module later (`tools/whatsapp.py` + a registered tool).

## Tests

```bash
.\.venv\Scripts\python.exe tests\test_cleantext.py
.\.venv\Scripts\python.exe tests\test_registry.py
.\.venv\Scripts\python.exe tests\test_router.py
.\.venv\Scripts\python.exe tests\test_planner_integration.py   # needs live Gemini quota
```

## Logs

Everything (recognized text, language, intent, actions, timing, errors) is logged
to `./logs/assistant.log`. No secrets are logged.
