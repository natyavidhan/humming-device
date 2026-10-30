import ctypes
import math
import queue
import sys, os
import threading
import time
import tkinter as tk
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageTk
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import audio_preprocessor, pitch_detector, melody_extractor, midi_generator, instrument_renderer, guitar, drums

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

HERE = Path(__file__).parent
SAVE_DIR = Path.home() / "Desktop" if (Path.home() / "Desktop").is_dir() else Path.home() / "Hum"
ORANGE, DARK, BLACK, WHITE, GREY = "#CE5719", "#A8470F", "#000000", "#FFFFFF", "#3A3A3A"
W, H = 380, 470
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ANCHOR = {"nw": "la", "ne": "ra", "center": "mm", "w": "lm", "e": "rm"}
RECORD_SECONDS = 8
SR = 22050
INSTRUMENTS = ("Piano", "Guitar", "Drums")

@lru_cache(maxsize=None)
def load_font(px):
    for f in (HERE / "Michroma-Regular.ttf", "bahnschrift.ttf", "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(str(f), px)
        except OSError:
            continue
    return ImageFont.load_default()

def midi_name(m):
    m = int(round(m))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

def note_stats(melody):
    notes = melody["notes"] if melody else []
    if not notes:
        return {"Median": "-", "Spread": "-", "Notes": "0", "Tuning": "-"}
    m = np.array([n["midi_note"] for n in notes])
    return {"Median": midi_name(np.median(m)), "Spread": f"{m.max() - m.min()} st",
            "Notes": str(len(notes)), "Tuning": f"{melody['tuning_cents']:+.0f} c"}

def guitar_stats(arr):
    if arr is None or not arr.notes:
        return {"Key": "-", "Chords": "-", "Frets": "-", "Notes": "0"}
    tonic, mode = arr.key.split()
    frets = [f for _, f in arr.frets]
    return {"Key": f"{tonic} {mode[:3]}", "Chords": str(len({c.name for c in arr.chords})),
            "Frets": f"{min(frets)}-{max(frets)}", "Notes": str(len(arr.notes))}

def drum_stats(hits):
    if not hits:
        return {"Tempo": "-", "Hits": "0", "Kicks": "0", "Snares": "0"}
    count = lambda d: str(sum(h["drum"] == d for h in hits))
    return {"Tempo": f"{60 / drums.beat_length(hits):.0f} bpm", "Hits": str(len(hits)),
            "Kicks": count(drums.KICK), "Snares": count(drums.SNARE)}

def mic_name():
    try:
        return sd.query_devices(kind="input")["name"]
    except Exception:
        return "No microphone found"

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.S = max(1.0, self.root.winfo_fpixels("1i") / 96)
        self.K = self.S * 2
        self.root.title("Hum")
        self.root.geometry(f"{int(W * self.S)}x{int(H * self.S)}+60+60")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.cv = tk.Canvas(self.root, width=int(W * self.S), height=int(H * self.S), bg=BLACK, highlightthickness=0)
        self.cv.pack()
        self.item = self.cv.create_image(0, 0, anchor="nw")
        self.cv.bind("<Button-1>", self.on_click)
        self.mic = mic_name()
        # work done on background threads hands its results back through here; Tk is main-thread only
        self.ui_calls = queue.Queue()
        self.state, self.msg, self.error = "idle", "", ""
        self.chunks, self.levels, self.peak, self.stream, self.t0 = [], [], 0.0, None, 0.0
        self.melody, self.arr, self.duration = None, None, 0.0
        self.instrument, self.backing = "Piano", True
        self.previews, self.rendering = {}, False
        self.playing, self.play_t0, self.play_len, self.saved = False, 0.0, 0.0, None
        self.hit = []
        self.drawn_minute = None
        self.draw()
        self.tick()

    def run(self):
        self.root.mainloop()

    def close(self):
        self.stop_play()
        if self.stream is not None:
            self.stream.close()
        self.root.destroy()

    def post(self, fn):
        self.ui_calls.put(fn)

    def rr(self, x1, y1, x2, y2, r=8, fill=None, outline=None, width=1):
        K = self.K
        self.d.rounded_rectangle([x1 * K, y1 * K, x2 * K, y2 * K], radius=r * K, fill=fill, outline=outline, width=max(1, int(width * K)))

    def rect(self, x1, y1, x2, y2, fill):
        K = self.K
        self.d.rectangle([x1 * K, y1 * K, x2 * K, y2 * K], fill=fill)

    def line(self, x1, y1, x2, y2, fill, width=1):
        K = self.K
        self.d.line([x1 * K, y1 * K, x2 * K, y2 * K], fill=fill, width=max(1, int(width * K)))

    def text(self, x, y, s, size, color=WHITE, anchor="nw"):
        K = self.K
        self.d.text((x * K, y * K), s, fill=color, font=load_font(int(size * 1.35 * K)), anchor=ANCHOR[anchor])

    def button(self, name, x1, y1, x2, y2, label, filled=True, size=8):
        if filled:
            self.rr(x1, y1, x2, y2, fill=ORANGE)
        else:
            self.rr(x1, y1, x2, y2, fill=BLACK, outline=ORANGE, width=2)
        self.text((x1 + x2) / 2, (y1 + y2) / 2, label, size, BLACK if filled else ORANGE, "center")
        self.hit.append((name, x1, y1, x2, y2))

    def dots(self, y):
        for x in range(40, 340, 6):
            self.rect(x, y, x + 3, y + 3, BLACK)

    def draw(self):
        self.img = Image.new("RGB", (int(W * self.K), int(H * self.K)), BLACK)
        self.d = ImageDraw.Draw(self.img)
        self.hit = []
        now = datetime.now()
        self.drawn_minute = now.minute
        self.text(40, 40, f"{now:%A} {now.day}", 8)
        self.text(38, 58, f"{now:%H:%M}", 26)
        {"idle": self.screen_record, "recording": self.screen_record, "analyzing": self.screen_record,
         "output": self.screen_output, "save": self.screen_save}[self.state]()
        out = self.img.resize((int(W * self.S), int(H * self.S)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(out)
        self.cv.itemconfig(self.item, image=self.photo)

    def screen_record(self):
        rec = self.state == "recording"
        title = {"idle": "Start recording", "recording": "Recording  -  tap to stop", "analyzing": "Listening back..."}
        self.text(38, 212, title[self.state], 8)
        self.rr(28, 240, 352, 310, fill=ORANGE)
        self.dots(272)
        for i, lv in enumerate(self.levels[-50:]):
            h = min(1.0, lv * 8) * 26
            self.rect(40 + i * 6, 273.5 - h, 43 + i * 6, 273.5 + h, BLACK)
        remaining = max(0.0, RECORD_SECONDS - (time.time() - self.t0)) if rec else RECORD_SECONDS
        self.text(350, 318, f"{remaining:.1f}", 8, anchor="ne")
        if self.error:
            self.text(38, 350, self.error[:46], 6, ORANGE)
        else:
            self.text(38, 350, f"Mic: {self.mic[:40]}", 6, "#8A8A8A")
        if self.state != "analyzing":
            self.hit.append(("panel", 28, 240, 352, 310))

    def screen_output(self):
        msg = self.error or ("Rendering..." if self.rendering else self.msg)
        self.text(38, 112, msg[:16], 8, ORANGE if self.error else WHITE)
        for i, name in enumerate(INSTRUMENTS):
            x1 = 166 + i * 64
            self.button(f"inst_{name}", x1, 102, x1 + 58, 124, name, self.instrument == name, 6)
        self.draw_roll(28, 134, 352, 288)
        self.button("play", 28, 298, 108, 330, "Stop" if self.playing else "Play", True, 8)
        if self.instrument != "Piano":
            what = "Chords" if self.instrument == "Guitar" else "Hi-hat"
            self.button("backing", 116, 298, 236, 330, f"{what} {'on' if self.backing else 'off'}", self.backing, 7)
        stats = {"Piano": lambda: note_stats(self.melody), "Guitar": lambda: guitar_stats(self.arr),
                 "Drums": lambda: drum_stats(self.view_notes())}[self.instrument]()
        for i, (k, v) in enumerate(stats.items()):
            x1, y1 = 28 + (i % 2) * 168, 342 + (i // 2) * 38
            self.rr(x1, y1, x1 + 156, y1 + 30, fill=BLACK, outline=GREY)
            self.text(x1 + 10, y1 + 15, k.upper(), 6, WHITE, "w")
            self.text(x1 + 146, y1 + 15, v, 9, ORANGE, "e")
        self.button("next", 28, 424, 352, 456, "Generate & Save", True, 9)

    def screen_save(self):
        self.text(38, 212, "Saved" if self.saved else "Save", 8)
        self.rr(28, 240, 352, 310, fill=ORANGE)
        if self.error:
            self.text(46, 258, "Could not save", 10, BLACK)
            self.text(46, 284, self.error[:40], 6, BLACK)
        elif self.saved:
            self.text(46, 258, f"Saved to {SAVE_DIR.name}", 10, BLACK)
            self.text(46, 284, self.saved, 6, BLACK)
        else:
            self.text(46, 258, "Generating...", 10, BLACK)
        self.button("new", 28, 326, 352, 372, "Start new", False, 9)
        if self.saved:
            folder = str(SAVE_DIR)
            self.text(38, 392, f"Folder: {folder if len(folder) <= 44 else '...' + folder[-41:]}", 6)

    def view_notes(self):
        if self.instrument == "Drums":
            return self.melody.get("drum_hits", []) if self.melody else []
        if self.instrument == "Guitar" and self.arr is not None:
            return self.arr.notes
        return self.melody["notes"] if self.melody else []

    def draw_roll(self, x1, y1, x2, y2):
        self.rr(x1, y1, x2, y2, fill=ORANGE)
        if self.instrument == "Drums":
            return self.draw_drum_lanes(x1, y1, x2, y2)
        guitar_mode = self.instrument == "Guitar" and self.arr is not None
        notes = self.view_notes()
        ms = [n["midi_note"] for n in notes]
        lo, hi = (min(ms) - 1, max(ms) + 1) if ms else (57, 69)
        if hi - lo < 12:
            mid = (lo + hi) / 2
            lo, hi = mid - 6, mid + 6
        left, right = x1 + 44, x2 - 10
        top, bottom = y1 + (26 if guitar_mode else 10), y2 - 10
        ypos = lambda m: bottom - (m - lo) / (hi - lo) * (bottom - top)
        dur = max(self.duration, 0.5)
        xpos = lambda t: left + min(t / dur, 1.0) * (right - left)
        for m in range(math.ceil(lo), math.floor(hi) + 1):
            if m % 12 in (0, 7):
                self.line(left, ypos(m), right, ypos(m), DARK)
                self.text(x1 + 8, ypos(m), midi_name(m), 6, BLACK, "w")
        if guitar_mode and self.backing:
            for ch in self.arr.chords:
                self.line(xpos(ch.start), y1 + 8, xpos(ch.start), bottom, DARK)
                self.text(xpos(ch.start) + 3, y1 + 13, ch.name, 6, BLACK, "w")
        for i, (n, m) in enumerate(zip(notes, ms)):
            bx1 = xpos(n["start"])
            bx2 = max(bx1 + 5, xpos(n["start"] + n["duration"]))
            self.rr(bx1, ypos(m) - 4, bx2, ypos(m) + 4, r=4, fill=BLACK)
            if guitar_mode:  # fret number, string shown by the tab file
                self.text(bx1, ypos(m) - 10, str(self.arr.frets[i][1]), 5, BLACK, "w")
        if self.playing:
            px = xpos(time.time() - self.play_t0)
            self.line(px, top, px, bottom, BLACK, 2)

    def draw_drum_lanes(self, x1, y1, x2, y2):
        left, right = x1 + 44, x2 - 10
        top, bottom = y1 + 10, y2 - 10
        lane_h = (bottom - top) / len(drums.LANES)
        dur = max(self.duration, 0.5)
        xpos = lambda t: left + min(t / dur, 1.0) * (right - left)
        hits = self.view_notes() + (drums.hat_groove(self.view_notes()) if self.backing else [])
        for i, d in enumerate(drums.LANES):
            y = top + (i + 0.5) * lane_h
            self.line(left, y, right, y, DARK)
            self.text(x1 + 8, y, drums.NAMES[d], 6, BLACK, "w")
            for h in hits:
                if h["drum"] == d:
                    r = 2 + 3 * h["velocity"] / 127
                    self.rr(xpos(h["time"]) - r, y - r, xpos(h["time"]) + r, y + r, r=r, fill=BLACK)
        if self.playing:
            px = xpos(time.time() - self.play_t0)
            self.line(px, top, px, bottom, BLACK, 2)

    def tick(self):
        dirty = False
        while not self.ui_calls.empty():
            self.ui_calls.get_nowait()()
            dirty = True
        if self.state == "recording":
            self.levels.append(self.peak)
            self.peak = 0.0
            if time.time() - self.t0 >= RECORD_SECONDS:
                self.stop_recording()
            dirty = True
        if self.playing:
            if time.time() - self.play_t0 > self.play_len:
                self.playing = False
            dirty = True
        if self.state == "idle" and datetime.now().minute != self.drawn_minute:
            dirty = True  # keep the clock current without redrawing every frame
        if dirty:
            self.draw()
        self.root.after(80, self.tick)

    def on_click(self, e):
        x, y = e.x / self.S, e.y / self.S
        for name, x1, y1, x2, y2 in self.hit:
            if x1 <= x <= x2 and y1 <= y <= y2:
                try:
                    self.act(name)
                except Exception as err:
                    self.error = f"Error: {err}"
                self.draw()
                return

    def act(self, name):
        if name == "panel":
            self.start_recording() if self.state == "idle" else self.stop_recording()
        elif name == "play":
            self.toggle_play()
        elif name.startswith("inst_"):
            self.stop_play()
            self.instrument = name.split("_", 1)[1]
        elif name == "backing":
            self.stop_play()
            self.backing = not self.backing
        elif name == "next":
            self.stop_play()
            self.generate_and_save()
        elif name == "new":
            self.state, self.saved, self.error, self.msg = "idle", None, "", ""
            self.melody, self.arr, self.previews, self.levels = None, None, {}, []

    def start_recording(self):
        self.error, self.levels, self.chunks, self.peak = "", [], [], 0.0

        def on_audio(indata, frames, t, status):
            self.chunks.append(indata[:, 0].copy())
            self.peak = max(self.peak, float(np.sqrt(np.mean(indata ** 2))))

        try:
            self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=on_audio)
            self.stream.start()
        except Exception as err:
            self.stream, self.error = None, f"Mic error: {err}"
            return
        self.t0 = time.time()
        self.state = "recording"

    def stop_recording(self):
        if self.state != "recording":
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)
        self.state = "analyzing"
        threading.Thread(target=self.analyze, args=(audio,), daemon=True).start()

    def analyze(self, audio):
        try:
            if len(audio) < SR // 2:
                raise ValueError("Recording too short")
            clean = audio_preprocessor.preprocess(audio, SR)
            times, f0, prob, loud = pitch_detector.detect_pitch(clean, SR, with_loudness=True)
            melody = melody_extractor.extract_melody(times, f0, prob, SR, loudness=loud)
            arr = guitar.arrange(melody)
            melody["drum_hits"] = drums.detect_hits(clean, SR, melody)
        except Exception as err:
            self.post(lambda err=err: self.analysis_failed(err))
            return
        self.post(lambda: self.show_result(audio, melody, arr))

    def analysis_failed(self, err):
        self.state, self.error = "idle", f"Error: {err}"

    def show_result(self, audio, melody, arr):
        self.melody, self.arr, self.previews = melody, arr, {}
        self.duration = len(audio) / SR
        heard = len(melody["notes"]) or len(melody["drum_hits"])
        self.msg = f"Heard {self.duration:.1f}s" if heard else "Heard nothing"
        self.state = "output"

    def toggle_play(self):
        if self.playing:
            return self.stop_play()
        if self.rendering or not self.view_notes():
            return
        key = (self.instrument, self.backing)
        if key in self.previews:
            return self.start_play(self.previews[key])
        self.rendering = True
        threading.Thread(target=self.render_preview, args=(key,), daemon=True).start()

    def render_preview(self, key):
        try:
            pm, _ = midi_generator.build_midi(self.melody, key[0], key[1])
            audio = instrument_renderer.render(pm, SR)
        except Exception as err:
            def failed(err=err):
                self.rendering, self.error = False, f"Error: {err}"
            self.post(failed)
            return

        def done():
            self.rendering = False
            self.previews[key] = audio
            # only start if the user hasn't switched instrument while we were rendering
            if key == (self.instrument, self.backing) and self.state == "output":
                self.start_play(audio)
        self.post(done)

    def start_play(self, audio):
        sd.play(audio, SR)
        self.playing, self.play_t0, self.play_len = True, time.time(), len(audio) / SR

    def stop_play(self):
        if self.playing:
            sd.stop()
        self.playing = False

    def generate_and_save(self):
        self.state, self.saved, self.error = "save", None, ""
        melody, key, cached = self.melody, (self.instrument, self.backing), self.previews.get((self.instrument, self.backing))

        def worker():
            try:
                SAVE_DIR.mkdir(parents=True, exist_ok=True)
                stem = SAVE_DIR / f"hum-{key[0].lower()}-{datetime.now():%Y%m%d-%H%M%S}"
                pm, arr = midi_generator.build_midi(melody, *key)
                pm.write(str(stem) + ".mid")
                audio = cached if cached is not None else instrument_renderer.render(pm, SR)
                sf.write(str(stem) + ".wav", audio, SR)
                exts = ".mid .wav"
                if arr is not None:
                    Path(str(stem) + ".txt").write_text(guitar.tab_text(arr), encoding="utf-8")
                    exts += " .txt"
            except Exception as err:
                self.post(lambda err=err: setattr(self, "error", str(err)))
                return
            self.post(lambda: setattr(self, "saved", f"{stem.name}  {exts}"))
        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    App().run()
