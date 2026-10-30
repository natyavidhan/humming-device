import tkinter as tk
from tkinter import messagebox
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import audio_recorder, audio_preprocessor, pitch_detector, melody_extractor, midi_generator, instrument_renderer, audio_player, session_manager, guitar, drums
from app.config import INSTRUMENTS

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Humming to Piano / Guitar / Drums")
        self.geometry("400x380")
        self.sess=None; self.audio=None; self.sr=22050; self.melody=None
        tk.Label(self,text="Humming to Piano / Guitar / Drums",font=("Arial",16)).pack(pady=10)
        self.dur=tk.IntVar(value=8)
        tk.Scale(self,from_=5,to=30,orient="horizontal",variable=self.dur,label="Seconds").pack()
        self.instrument=tk.StringVar(value=INSTRUMENTS[0]); self.backing=tk.BooleanVar(value=True)
        row=tk.Frame(self); row.pack()
        for name in INSTRUMENTS:
            tk.Radiobutton(row,text=name.title(),value=name,variable=self.instrument).pack(side="left")
        tk.Checkbutton(row,text="Backing (chords / hi-hat)",variable=self.backing).pack(side="left")
        tk.Button(self,text="Record",command=self.rec,width=20).pack(pady=5)
        tk.Button(self,text="Analyze",command=self.analyze,width=20).pack(pady=5)
        self.count=tk.Label(self,text="",font=("Arial",12)); self.count.pack()
        tk.Button(self,text="Play Recorded",command=self.playrec,width=20).pack(pady=5)

    def rec(self):
        dur=self.dur.get()
        self.count.config(text=f"Recording {dur}s")
        def worker():
            self.audio,self.sr=audio_recorder.record(duration=dur,sr=22050)
            self.sess=session_manager.SessionManager(); self.sess.create()
            audio_recorder.save(str(self.sess.path("humming.wav")),self.audio,self.sr)
            self.after(0, lambda: (self.count.config(text="Done"), messagebox.showinfo("Done","Recorded")))
        # countdown
        def tick(rem):
            if rem<=0: return
            self.count.config(text=f"Recording... {rem}s")
            self.after(1000, lambda: tick(rem-1))
        tick(dur)
        threading.Thread(target=worker,daemon=True).start()

    def playrec(self):
        if self.sess:
            audio_player.play(str(self.sess.path("humming.wav")))

    def analyze(self):
        if self.sess is None:
            messagebox.showerror("Error","Record first"); return
        audio,sr=audio_recorder.load(str(self.sess.path("humming.wav")))
        clean=audio_preprocessor.preprocess(audio,sr)
        times,f0,prob,loud=pitch_detector.detect_pitch(clean,sr,with_loudness=True)
        melody=melody_extractor.extract_melody(times,f0,prob,sr,loudness=loud)
        melody["drum_hits"]=drums.detect_hits(clean,sr,melody)
        self.melody=melody
        self.sess.save_json("melody.json",melody)
        inst=self.instrument.get()
        _,arr=midi_generator.melody_to_midi(melody,str(self.sess.path(f"{inst}.mid")),inst,self.backing.get())
        wav=str(self.sess.path(f"{inst}_preview.wav"))
        instrument_renderer.render_midi(str(self.sess.path(f"{inst}.mid")),wav)
        # second window
        win=tk.Toplevel(self)
        win.title(f"{inst.title()} Output")
        txt=tk.Text(win,height=15,width=70 if arr else 50,font=("Courier",10) if arr else None); txt.pack()
        if arr:
            tab=guitar.tab_text(arr)
            self.sess.path("guitar_tab.txt").write_text(tab,encoding="utf-8")
            txt.insert("end",tab)
        elif inst=="drums":
            for h in melody["drum_hits"]:
                txt.insert("end",f"{h['time']:6.2f}s  {drums.NAMES[h['drum']]:<5} vel={h['velocity']}\n")
        else:
            for n in melody["notes"]:
                txt.insert("end",f"{n['pitch']} start={n['start']:.2f} dur={n['duration']:.2f}\n")
        tk.Button(win,text=f"Play {inst.title()}",command=lambda: audio_player.play(wav)).pack(pady=5)
        messagebox.showinfo("Done","Analyzed")

if __name__=="__main__":
    App().mainloop()
