import numpy as np
import librosa
from scipy.ndimage import maximum_filter1d, median_filter

# A note only changes once the pitch drifts this far (semitones) from the current note.
# Above 0.5 so vibrato or a hum that sits between two notes doesn't flip back and forth.
HYSTERESIS = 0.7
# A frame this many dB below the loudest frame within ~100 ms is a breath between notes,
# even when the pitch tracker still calls it voiced. This is what separates repeated notes.
DIP_DB = 10.0


def _runs(mask):
    """(start, end) index pairs of consecutive True values, end exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _label(pitch):
    """Round to semitones, holding the current note until the pitch clearly leaves it."""
    labels = np.empty(len(pitch), dtype=int)
    cur = int(round(pitch[0]))
    for i, p in enumerate(pitch):
        if abs(p - cur) > HYSTERESIS:
            cur = int(round(p))
        labels[i] = cur
    return labels


def _segments(labels):
    """[label, start, end] for each run of equal labels."""
    bounds = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(labels)]))
    return [[int(labels[s]), int(s), int(e)] for s, e in zip(starts, ends)]


def _absorb_short(segs, pitch, min_frames):
    """Fold segments shorter than min_frames (glides, blips) into the closest-pitched neighbour."""
    while len(segs) > 1:
        i = min(range(len(segs)), key=lambda k: segs[k][2] - segs[k][1])
        lab, s, e = segs[i]
        if e - s >= min_frames:
            break
        mean = float(np.mean(pitch[s:e]))
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
        j = min(nbrs, key=lambda k: (abs(segs[k][0] - mean), -(segs[k][2] - segs[k][1])))
        segs[j][1], segs[j][2] = min(segs[j][1], s), max(segs[j][2], e)
        del segs[i]
        # neighbours may now share a label
        merged = [segs[0]]
        for seg in segs[1:]:
            if seg[0] == merged[-1][0]:
                merged[-1][2] = seg[2]
            else:
                merged.append(seg)
        segs = merged
    return segs


def extract_melody(times, f0, voiced_prob, sr=None, hop=None, conf_threshold=0.5,
                   min_note=0.08, merge_gap=0.05, loudness=None):
    times = np.asarray(times, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    prob = np.asarray(voiced_prob, dtype=float)
    if len(times) == 0:
        return {"notes": [], "tempo_bpm": None, "tuning_cents": 0.0}
    frame = float(np.median(np.diff(times))) if len(times) > 1 else 0.016
    min_frames = max(1, int(round(min_note / frame)))

    voiced = (prob >= conf_threshold) & np.isfinite(f0) & (f0 > 0)
    if loudness is not None:
        loud = np.asarray(loudness, dtype=float)
        voiced &= loud >= maximum_filter1d(loud, size=2 * int(round(0.1 / frame)) + 1) - DIP_DB
    midi = np.full(len(f0), np.nan)
    midi[voiced] = librosa.hz_to_midi(f0[voiced])

    runs = [(s, e) for s, e in _runs(voiced) if e - s >= min_frames]
    smooth = midi.copy()
    for s, e in runs:
        smooth[s:e] = median_filter(midi[s:e], size=min(5, e - s), mode="nearest")

    # Most people hum consistently a bit sharp or flat. Estimate that offset (circular mean of
    # the fractional semitone) and remove it, so notes don't land on the wrong side of a boundary.
    used = np.concatenate([smooth[s:e] for s, e in runs]) if runs else np.array([])
    offset = 0.0
    if len(used) >= 10:
        offset = float(np.angle(np.mean(np.exp(2j * np.pi * used))) / (2 * np.pi))
    tuned = smooth - offset

    notes = []
    for s, e in runs:
        segs = _absorb_short(_segments(_label(tuned[s:e])), tuned[s:e], min_frames)
        for _, a, b in segs:
            a, b = s + a, s + b
            pitch = int(round(float(np.median(tuned[a:b]))))
            notes.append({
                "pitch": librosa.midi_to_note(pitch, octave=True, unicode=False),
                "midi_note": pitch,
                "start": float(times[a]),
                "duration": float(times[b - 1] + frame - times[a]),
                "confidence": float(np.mean(prob[a:b])),
                "cents": float((np.median(midi[a:b]) - pitch) * 100),
            })

    # rejoin a note that was split by a momentary confidence dropout
    merged = []
    for n in notes:
        prev = merged[-1] if merged else None
        if prev and prev["midi_note"] == n["midi_note"] and n["start"] - prev["start"] - prev["duration"] < merge_gap:
            prev["duration"] = n["start"] + n["duration"] - prev["start"]
        else:
            merged.append(n)
    return {"notes": merged, "tempo_bpm": None, "tuning_cents": offset * 100}
