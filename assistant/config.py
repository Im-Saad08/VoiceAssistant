"""Central configuration for the voice assistant.

All tunable values live here so the rest of the code never hardcodes
paths, model names, or timing values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load GEMINI_API_KEY from .env (kept out of git via .gitignore).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Planner backend ------------------------------------------------------
# "ollama" (default, local, free) or "gemini" (requires API key, quota-limited).
PLANNER_BACKEND = os.getenv("PLANNER_BACKEND", "ollama")

# ---- Ollama (local planner) -----------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
OLLAMA_RETRY_BACKOFF_SECONDS = float(os.getenv("OLLAMA_RETRY_BACKOFF_SECONDS", "2.0"))
# Ollama inference can be slow on CPU; allow generous timeout.
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120.0"))

# ---- Gemini -------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
# Low temperature keeps intent parsing deterministic.
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
# Retry/backoff for transient server errors (503 high-demand, etc.).
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_BACKOFF_SECONDS = float(os.getenv("GEMINI_RETRY_BACKOFF_SECONDS", "2.0"))

# ---- Text-to-speech ------------------------------------------------------
# Microsoft Edge neural voice. Change to any edge-tts voice name.
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")
# edge-tts is a network service; cap how long a single generation may take.
TTS_TIMEOUT_SECONDS = float(os.getenv("TTS_TIMEOUT_SECONDS", "30.0"))

# ---- Speech recognition --------------------------------------------------
# phrase_time_limit caps a single spoken utterance so a long pause doesn't
# stall the loop; timeout is how long we wait for speech to begin.
MIC_TIMEOUT_SECONDS = float(os.getenv("MIC_TIMEOUT_SECONDS", "5"))
MIC_PHRASE_LIMIT_SECONDS = float(os.getenv("MIC_PHRASE_LIMIT_SECONDS", "12"))
# Leave language unset so recognize_google auto-detects (supports Urdu/Hinglish).

# ---- Logging ---------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"

# ---- Browser (Playwright) --------------------------------------------------
# A persistent headed Chromium profile so the assistant's browser keeps its
# session/state between runs. Set to "" to use a fresh incognito profile.
BROWSER_PROFILE_DIR = os.getenv(
    "BROWSER_PROFILE_DIR",
    str(BASE_DIR / ".browser_profile"),
)
BROWSER_HEADED = os.getenv("BROWSER_HEADED", "1") == "1"
# Default search engine for browser_search.
# "bing" is the default because Google serves a CAPTCHA block to automated
# browsers; "google" and "duckduckgo" are also supported.
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "bing")
