"""Turn a vocal take into drum hits. Works for beatboxing as well as humming: every onset in the
audio becomes a hit, unvoiced sounds ("ts", "k", "p") are classified by how bright they are,
and hummed notes are mapped by pitch within the singer's own range (low = kick)."""
import numpy as np
import librosa

KICK, SNARE, CLOSED_HAT, HIGH_TOM = 36, 38, 42, 50   # General MIDI drum notes
NAMES = {KICK: "Kick", SNARE: "Snare", CLOSED_HAT: "Hat", HIGH_TOM: "Tom"}
LANES = (CLOSED_HAT, HIGH_TOM, SNARE, KICK)            # top to bottom, for display
HOP = 256
MIN_GAP = 0.08    # hits closer than this are one hit


def _voiced_note(t, notes):
    return next((n for n in notes if n["start"] - 0.05 <= t < n["start"] + n["duration"]), None)


def detect_hits(audio, sr, melody):
    """[{"time", "drum", "velocity"}] from the audio onsets plus the melody's note starts."""
    notes = melody["notes"] if melody else []
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < sr // 10:
        return []
    env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP, units="time")
    # note starts are the more precise onsets for hummed sounds, so they win over nearby audio onsets
    merged = [n["start"] for n in notes]
    for t in onsets:
        if all(abs(t - m) >= MIN_GAP for m in merged):
            merged.append(float(t))
    merged.sort()

    pitches = [n["midi_note"] for n in notes]
    lo, hi = (min(pitches), max(pitches)) if pitches else (0, 1)
    peak = float(env.max()) or 1.0
    win = int(0.06 * sr)
    hits = []
    for t in merged:
        i = int(t * sr)
        seg = audio[i:i + win]
        if len(seg) < 64 or np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue  # onset with nothing behind it
        strength = env[min(len(env) - 1, int(t * sr / HOP))] / peak
        note = _voiced_note(t, notes)
        if note is not None:
            pos = (note["midi_note"] - lo) / max(1, hi - lo)
            drum = KICK if pos < 0.34 else SNARE if pos < 0.67 else HIGH_TOM
        else:
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            freqs = np.fft.rfftfreq(len(seg), 1 / sr)
            centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))
            drum = KICK if centroid < 1200 else SNARE if centroid < 3500 else CLOSED_HAT
        hits.append({"time": t, "drum": drum, "velocity": int(np.clip(50 + 70 * strength, 40, 120))})
    return hits


def beat_length(hits):
    """Seconds per beat: assume the typical gap between hits is an eighth note (60-170 bpm)."""
    times = np.array([h["time"] for h in hits])
    if len(times) < 3:
        return 0.5
    return float(np.clip(2 * np.median(np.diff(times)), 60 / 170, 1.0))


def hat_groove(hits):
    """Steady eighth-note hi-hats from the first hit to the last, skipping ones the singer made."""
    if len(hits) < 2:
        return []
    beat = beat_length(hits)
    own_hats = [h["time"] for h in hits if h["drum"] == CLOSED_HAT]
    groove = []
    t, k = hits[0]["time"], 0
    while t <= hits[-1]["time"] + beat:
        if not any(abs(t - h) < 0.05 for h in own_hats):
            groove.append({"time": t, "drum": CLOSED_HAT, "velocity": 55 if k % 2 == 0 else 38})
        t += beat / 2
        k += 1
    return groove
