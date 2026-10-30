import os, json, time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = BASE_DIR / "sessions"

SAMPLE_RATE = 22050
INSTRUMENTS = ["piano", "guitar", "drums"]

def new_session_dir():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    i = 1
    while (SESSIONS_DIR / f"session_{i:03d}").exists():
        i += 1
    p = SESSIONS_DIR / f"session_{i:03d}"
    p.mkdir(parents=True, exist_ok=True)
    return p
