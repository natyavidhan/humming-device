from functools import lru_cache

import numpy as np
import pretty_midi
import soundfile as sf
from scipy.signal import lfilter

RELEASE = 0.12        # seconds of fade after a note is let go
MAX_RING = 4.0        # longest a single note is synthesised for


def is_guitar(program):
    return 24 <= program <= 31  # General MIDI guitar family


def _piano(freq, dur, vel, sr):
    """Additive piano-ish tone: slightly stretched partials, each decaying, upper ones faster."""
    n = int((dur + RELEASE) * sr)
    t = np.arange(n) / sr
    decay = 0.8 + freq / 350.0                      # high notes die away faster
    bright = 0.35 + 0.45 * vel                       # harder hits have more upper partials
    wave = np.zeros(n)
    rng = np.random.default_rng(int(freq * 100))
    for k in range(1, 13):
        fk = k * freq * np.sqrt(1 + 0.0004 * k * k)  # string stiffness stretches the partials
        if fk > sr * 0.45:
            break
        amp = bright ** (k - 1) / k
        env = 0.65 * np.exp(-decay * (1 + 0.4 * k) * t) + 0.35 * np.exp(-decay * 0.25 * (1 + 0.2 * k) * t)
        phase = rng.uniform(0, 2 * np.pi)
        # two strings per note, a hair out of tune with each other, give the slow piano shimmer
        wave += amp * env * (np.sin(2 * np.pi * fk * 0.9997 * t + phase) + np.sin(2 * np.pi * fk * 1.0003 * t))
    attack = min(int(0.004 * sr), n)
    wave[:attack] *= np.linspace(0, 1, attack)
    return wave * vel * 0.25


@lru_cache(maxsize=256)
def _pluck_raw(pitch, vel_q, sr):
    """Karplus-Strong plucked string, MAX_RING seconds long. Cached: strums reuse the same strings."""
    freq = pretty_midi.note_number_to_hz(pitch)
    vel = vel_q / 16.0
    period = sr / freq
    # loop delay = N samples + 0.5 (averaging filter) + allpass fraction, tuned to the exact period
    N = int(np.floor(period - 0.5 - 0.1))
    frac = period - 0.5 - N
    C = (1 - frac) / (1 + frac)
    t60 = float(np.clip(3.5 * np.sqrt(110.0 / freq), 0.7, 4.0))
    g = 10 ** (-3.0 / (t60 * freq))                 # loop gain per trip for that ring time

    rng = np.random.default_rng(pitch * 31 + vel_q)
    burst = rng.uniform(-1, 1, N + 2)
    soft = 0.6 - 0.45 * vel                          # softer picking = darker attack
    burst = lfilter([1 - soft], [1, -soft], burst)
    pick = max(1, int(0.13 * period))                # plucking 13% along the string
    burst[pick:] -= burst[:-pick]

    n = int(MAX_RING * sr)
    x = np.zeros(n)
    x[:len(burst)] = burst
    # y[n] = x[n] + C x[n-1] - C y[n-1] + g/2 (allpass * averager) applied at y[n-N]
    a = np.zeros(N + 3)
    a[0], a[1] = 1.0, C
    a[N] -= 0.5 * g * C
    a[N + 1] -= 0.5 * g * (1 + C)
    a[N + 2] -= 0.5 * g
    y = lfilter([1.0, C], a, x)
    return y / (np.max(np.abs(y)) + 1e-9) * vel


def _guitar(pitch, dur, vel, sr):
    raw = _pluck_raw(int(pitch), int(round(vel * 16)), sr)
    n = min(len(raw), int((dur + RELEASE) * sr))
    wave = raw[:n].copy()
    hold = min(n, int(dur * sr))
    tail = n - hold
    if tail > 0:  # finger lifts / palm mutes the string
        wave[hold:] *= np.exp(-np.arange(tail) / (0.03 * sr))
    return wave * 0.35


@lru_cache(maxsize=64)
def _drum(note, vel_q, sr):
    """Synthesised GM drum hit: 36 kick, 38 snare, 42 closed hat, anything else a tom."""
    vel = vel_q / 16.0
    rng = np.random.default_rng(note)
    if note == 36:      # sine dropping 150 -> 45 Hz, plus a click
        n = int(0.45 * sr)
        t = np.arange(n) / sr
        f = 45 + 105 * np.exp(-t / 0.035)
        wave = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.15)
        wave[:int(0.003 * sr)] += rng.uniform(-0.5, 0.5, int(0.003 * sr))
    elif note == 38:    # body tone + band of noise for the wires
        n = int(0.3 * sr)
        t = np.arange(n) / sr
        noise = lfilter([1, -1], [1, -0.6], rng.uniform(-1, 1, n))
        wave = 0.6 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.05) + 0.7 * noise * np.exp(-t / 0.09)
    elif note in (42, 44, 46):  # high-passed noise, short
        n = int(0.09 * sr)
        t = np.arange(n) / sr
        wave = 0.5 * lfilter([1, -1], [1, 0.3], rng.uniform(-1, 1, n)) * np.exp(-t / 0.025)
    else:               # tom: pitched sweep, higher note number = higher drum
        n = int(0.4 * sr)
        t = np.arange(n) / sr
        base = 80 * 2 ** ((note - 41) / 12)
        f = base * (1 + 0.5 * np.exp(-t / 0.04))
        wave = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.18)
    return wave * vel * 0.5


def render(pm, sr=22050):
    total = pm.get_end_time() + RELEASE + 0.5
    audio = np.zeros(int(total * sr) + 1)
    for inst in pm.instruments:
        guitar = is_guitar(inst.program)
        for note in inst.notes:
            dur = min(note.end - note.start, MAX_RING)
            if dur <= 0:
                continue
            vel = note.velocity / 127.0
            if inst.is_drum:
                wave = _drum(note.pitch, int(round(vel * 16)), sr)
            elif guitar:
                wave = _guitar(note.pitch, dur, vel, sr)
            else:
                wave = _piano(pretty_midi.note_number_to_hz(note.pitch), dur, vel, sr)
            start = int(note.start * sr)
            wave = wave[:len(audio) - start]
            audio[start:start + len(wave)] += wave
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8
    return audio.astype(np.float32)


def render_midi(midi_path, wav_path, sr=22050):
    sf.write(wav_path, render(pretty_midi.PrettyMIDI(midi_path), sr), sr)
    return wav_path
