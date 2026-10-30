"""Turn a hummed melody into a guitar part: fretboard positions (tab) plus a strummed
chord accompaniment chosen to fit the melody."""
from dataclasses import dataclass, field

import numpy as np

TUNING = (40, 45, 50, 55, 59, 64)            # E2 A2 D3 G3 B3 E4, low string first
STRING_NAMES = ("E", "A", "D", "G", "B", "e")
MAX_FRET = 19
LOWEST, HIGHEST = TUNING[0], TUNING[-1] + MAX_FRET
PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")

# Krumhansl-Kessler key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# (semitones above tonic, quality, how much we like it a priori). Minor uses the major V
# from harmonic minor, which is what most songs do.
DIATONIC = {
    "major": [(0, "", 1.0), (2, "m", 0.3), (4, "m", 0.1), (5, "", 0.6), (7, "", 0.7), (9, "m", 0.4)],
    "minor": [(0, "m", 1.0), (3, "", 0.3), (5, "m", 0.6), (7, "", 0.6), (8, "", 0.5), (10, "", 0.4)],
}
# bonus for common moves, keyed by (from degree, to degree)
TRANSITIONS = {(7, 0): 0.5, (5, 7): 0.2, (5, 0): 0.2, (2, 7): 0.3, (9, 5): 0.2, (8, 10): 0.2, (10, 0): 0.3}

# Open-position shapes, low E to high e; None = muted string
OPEN_SHAPES = {
    "C": (None, 3, 2, 0, 1, 0), "D": (None, None, 0, 2, 3, 2), "E": (0, 2, 2, 1, 0, 0),
    "G": (3, 2, 0, 0, 0, 3), "A": (None, 0, 2, 2, 2, 0), "Am": (None, 0, 2, 2, 1, 0),
    "Dm": (None, None, 0, 2, 3, 1), "Em": (0, 2, 2, 0, 0, 0),
}
BARRE_SHAPES = {  # (string the root is on, shape relative to the barre)
    "": [(0, (0, 2, 2, 1, 0, 0)), (1, (None, 0, 2, 2, 2, 0))],
    "m": [(0, (0, 2, 2, 0, 0, 0)), (1, (None, 0, 2, 2, 1, 0))],
}

# Strum pattern per bar, in beats: D . D U . U D U
STRUM = [(0.0, "down", 80), (1.0, "down", 60), (1.5, "up", 45), (2.5, "up", 45), (3.0, "down", 60), (3.5, "up", 45)]
STRUM_SPREAD = 0.012  # seconds between strings in one strum


@dataclass
class Chord:
    name: str
    root: int
    quality: str
    degree: int
    start: float
    end: float

    @property
    def shape(self):
        return chord_shape(self.root, self.quality)


@dataclass
class GuitarArrangement:
    notes: list                     # melody notes, shifted onto the fretboard
    frets: list                     # (string index, fret) per note
    key: str = ""
    chords: list = field(default_factory=list)
    bar: float = 2.0                # seconds per chord slot
    shift: int = 0                  # semitones the melody was moved to fit the guitar


def fit_to_range(pitches):
    """Octave shift (in semitones) that puts the melody on the fretboard; 0 when it already fits."""
    if not pitches:
        return 0
    p = np.array(pitches)
    outside = lambda k: int(np.sum((p + k < LOWEST) | (p + k > HIGHEST)))
    return min((12 * k for k in range(-3, 4)), key=lambda k: (outside(k), abs(k)))


def fingering(pitches):
    """Choose a (string, fret) for every note, minimising hand movement (Viterbi over positions)."""
    if not pitches:
        return []
    cands = []
    for p in pitches:
        c = [(s, p - o) for s, o in enumerate(TUNING) if 0 <= p - o <= MAX_FRET]
        cands.append(c or [(0, 0) if p < LOWEST else (5, MAX_FRET)])

    def local(pos):
        f = pos[1]
        return 0.1 * f + (1.5 if f > 12 else 0.0)

    def move(a, b):
        (sa, fa), (sb, fb) = a, b
        if fa == 0 or fb == 0:  # open strings don't need the fretting hand
            hand = 0.5
        else:
            d = abs(fa - fb)
            hand = 0.3 * d + max(0, d - 3)  # cheap within a 4-fret span, expensive beyond
        return hand + 0.3 * abs(sa - sb)

    cost = [local(c) for c in cands[0]]
    back = []
    for prev, cur in zip(cands, cands[1:]):
        step, ptr = [], []
        for c in cur:
            j = min(range(len(prev)), key=lambda k: cost[k] + move(prev[k], c))
            step.append(cost[j] + move(prev[j], c) + local(c))
            ptr.append(j)
        cost = step
        back.append(ptr)
    i = int(np.argmin(cost))
    path = [i]
    for ptr in reversed(back):
        i = ptr[i]
        path.append(i)
    path.reverse()
    return [cands[k][i] for k, i in enumerate(path)]


def detect_key(notes):
    """(tonic pitch class, 'major' | 'minor') by correlating duration-weighted pitch classes."""
    hist = np.zeros(12)
    for n in notes:
        hist[n["midi_note"] % 12] += n["duration"]
    if not hist.any():
        return 0, "major"
    best = max(((np.corrcoef(hist, np.roll(prof, t))[0, 1], t, mode)
                for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)) for t in range(12)),
               key=lambda r: r[0])
    return best[1], best[2]


def bar_length(notes):
    """Guess seconds per chord from note spacing: about four typical onsets per chord."""
    onsets = np.array([n["start"] for n in notes])
    if len(onsets) < 3:
        return 2.0
    return float(np.clip(4 * np.median(np.diff(onsets)), 1.2, 3.0))


def harmonize(notes, tonic, mode, bar):
    """Pick one diatonic chord per slot of `bar` seconds (Viterbi over slots), then merge repeats."""
    if not notes:
        return []
    end = max(n["start"] + n["duration"] for n in notes)
    t0 = notes[0]["start"]
    n_slots = max(1, int(np.ceil((end - t0) / bar - 0.25)))
    vocab = DIATONIC[mode]

    def fit(slot, deg, quality):
        lo, hi = t0 + slot * bar, t0 + (slot + 1) * bar
        root = (tonic + deg) % 12
        tones = {root, (root + (3 if quality == "m" else 4)) % 12, (root + 7) % 12}
        score = 0.0
        for n in notes:
            overlap = min(hi, n["start"] + n["duration"]) - max(lo, n["start"])
            if overlap <= 0:
                continue
            pc = n["midi_note"] % 12
            w = overlap * (1.5 if abs(n["start"] - lo) < 0.15 * bar else 1.0)  # downbeat notes matter more
            score += w * ((1.15 if pc == root else 1.0) if pc in tones else -0.6)
        return score / bar

    def prior(slot, deg, p):
        bonus = p
        if deg == 0 and slot in (0, n_slots - 1):
            bonus += 0.5  # start and finish on the tonic
        return 0.5 * bonus

    score = [fit(0, d, q) + prior(0, d, p) for d, q, p in vocab]
    back = []
    for slot in range(1, n_slots):
        step, ptr = [], []
        for d, q, p in vocab:
            here = fit(slot, d, q) + prior(slot, d, p)
            trans = [score[j] + TRANSITIONS.get((vocab[j][0], d), 0.0) - (0.15 if vocab[j][0] == d else 0.0)
                     for j in range(len(vocab))]
            j = int(np.argmax(trans))
            step.append(trans[j] + here)
            ptr.append(j)
        score = step
        back.append(ptr)
    i = int(np.argmax(score))
    path = [i]
    for ptr in reversed(back):
        i = ptr[i]
        path.append(i)
    path.reverse()

    chords = []
    for slot, i in enumerate(path):
        d, q, _ = vocab[i]
        root = (tonic + d) % 12
        if chords and chords[-1].degree == d:
            chords[-1].end += bar
        else:
            s = t0 + slot * bar
            chords.append(Chord(PC_NAMES[root] + q, root, q, d, s, s + bar))
    return chords


def chord_shape(root, quality):
    """Fret per string for a chord: the open shape if there is one, else the lowest barre shape."""
    name = PC_NAMES[root] + quality
    if name in OPEN_SHAPES:
        return OPEN_SHAPES[name]
    options = []
    for string, rel in BARRE_SHAPES[quality]:
        base = (root - TUNING[string]) % 12 or 12
        options.append((base, tuple(None if f is None else f + base for f in rel)))
    return min(options)[1]


def arrange(melody):
    notes = [dict(n) for n in melody["notes"]]
    shift = fit_to_range([n["midi_note"] for n in notes])
    for n in notes:
        n["midi_note"] += shift
    tonic, mode = detect_key(notes)
    bar = bar_length(notes)
    return GuitarArrangement(
        notes=notes,
        frets=fingering([n["midi_note"] for n in notes]),
        key=f"{PC_NAMES[tonic]} {mode}",
        chords=harmonize(notes, tonic, mode, bar),
        bar=bar,
        shift=shift,
    )


def strum_events(arr):
    """(pitch, start, end, velocity) for every string hit in the chord accompaniment."""
    hits = []
    for ch in arr.chords:
        strings = [TUNING[s] + f for s, f in enumerate(ch.shape) if f is not None]
        ring_until = ch.end + (1.0 if ch is arr.chords[-1] else 0.0)  # let the last chord ring out
        bar_start = ch.start
        while bar_start < ch.end - 1e-6:
            for beat, direction, vel in STRUM:
                t = bar_start + beat * arr.bar / 4
                if t >= ch.end - 1e-6:
                    break
                order = strings if direction == "down" else strings[::-1][:4]  # upstrokes catch the top strings
                hits += [(t + k * STRUM_SPREAD, p, vel - 4 * k, ring_until) for k, p in enumerate(order)]
            bar_start += arr.bar
    hits.sort()
    # a string rings until it is struck again or the chord changes
    events = []
    for i, (t, p, v, ring_until) in enumerate(hits):
        nxt = next((h[0] for h in hits[i + 1:] if h[1] == p), ring_until)
        events.append((p, t, max(t + 0.05, min(nxt, ring_until)), v))
    return events


def shape_text(shape):
    return "".join("x" if f is None else str(f) if f < 10 else f"({f})" for f in shape)


def tab_text(arr, per_line=16):
    lines = [f"Key: {arr.key}    Tuning: E A D G B E",
             "Chords: " + "  ".join(f"{name} {shape_text(c.shape)}"
                                    for name, c in {c.name: c for c in arr.chords}.items())]
    if arr.shift:
        lines.append(f"(melody moved {arr.shift:+d} semitones to fit the guitar)")
    lines.append("")
    if not arr.notes:
        return "\n".join(lines + ["(no notes)"])
    for i0 in range(0, len(arr.notes), per_line):
        top, rows, prev = "", [""] * 6, None
        for n, (string, fret) in zip(arr.notes[i0:i0 + per_line], arr.frets[i0:i0 + per_line]):
            ch = next((c for c in arr.chords if c.start - 1e-6 <= n["start"] < c.end), None)
            label = ch.name if ch is not None and ch is not prev else ""
            prev = ch or prev
            w = max(len(str(fret)), len(label)) + 2
            top += label.ljust(w)
            for s in range(6):
                rows[s] += "-" + (str(fret) if s == string else "").ljust(w - 1, "-")
        lines.append("   " + top.rstrip())  # lines the names up over the fret numbers
        lines += [f"{STRING_NAMES[s]}|{rows[s]}-|" for s in reversed(range(6))]
        lines.append("")
    return "\n".join(lines)
