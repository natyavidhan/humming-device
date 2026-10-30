import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

# Cap on normalisation gain, so a near-silent take isn't blown up into loud noise that the
# pitch detector would read as voiced frames.
MAX_GAIN = 20.0


def preprocess(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 32:
        return audio
    audio = audio - np.mean(audio)
    # remove rumble / handling noise below the lowest hummed note
    audio = sosfiltfilt(butter(2, 60, btype="highpass", fs=sr, output="sos"), audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio * min(0.9 / peak, MAX_GAIN)
    return audio


def voiced_regions(audio, sr):
    intervals = librosa.effects.split(audio, top_db=40)
    return [(s/sr, e/sr) for s,e in intervals]
