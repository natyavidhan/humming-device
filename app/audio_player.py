import sounddevice as sd
import soundfile as sf

def play(path, block=False):
    # non-blocking by default so a button press doesn't freeze the window until playback ends
    audio, sr = sf.read(path)
    sd.play(audio, sr)
    if block:
        sd.wait()

def stop():
    sd.stop()
