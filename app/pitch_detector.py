import numpy as np

# Humming range, roughly C2 to C6. Narrowing the search band cuts octave errors.
FMIN, FMAX = 65.0, 1050.0

_det = None


def _detector():
    # load the ONNX model on first use instead of at import time
    global _det
    if _det is None:
        from swift_f0 import SwiftF0
        _det = SwiftF0()
    return _det


def detect_pitch(audio, sr, with_loudness=False):
    res = _detector().detect(np.asarray(audio, dtype=np.float32), sr, fmin=FMIN, fmax=FMAX)
    if with_loudness:
        return res.timestamps, res.pitch_hz, res.confidence, res.loudness_db
    return res.timestamps, res.pitch_hz, res.confidence
