import pretty_midi

from app import drums, guitar

PROGRAMS = {"piano": 0, "guitar": 25}  # GM: acoustic grand, steel-string acoustic


def build_midi(melody, instrument="piano", backing=True):
    """PrettyMIDI for the take. Returns (midi, GuitarArrangement or None).

    piano:  the melody as played.
    guitar: the melody fitted to the fretboard, plus strummed chords if `backing`.
    drums:  one hit per sound in the take (melody["drum_hits"], filled in by the caller from the
            audio, else one per note), plus a steady hi-hat if `backing`."""
    instrument = instrument.lower()
    pm = pretty_midi.PrettyMIDI()

    if instrument == "drums":
        hits = melody.get("drum_hits")
        if hits is None:
            hits = [{"time": n["start"], "drum": drums.SNARE, "velocity": 90} for n in melody["notes"]]
        kit = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
        for h in hits + (drums.hat_groove(hits) if backing else []):
            kit.notes.append(pretty_midi.Note(velocity=h["velocity"], pitch=h["drum"], start=h["time"], end=h["time"] + 0.1))
        pm.instruments.append(kit)
        return pm, None

    arr = guitar.arrange(melody) if instrument == "guitar" else None
    notes = arr.notes if arr else melody["notes"]

    lead = pretty_midi.Instrument(program=PROGRAMS[instrument], name=f"{instrument.title()} melody")
    for n in notes:
        lead.notes.append(pretty_midi.Note(velocity=100, pitch=n["midi_note"], start=n["start"], end=n["start"] + n["duration"]))
    pm.instruments.append(lead)

    if arr and backing and arr.chords:
        acc = pretty_midi.Instrument(program=PROGRAMS["guitar"], name="Guitar chords")
        for pitch, start, end, vel in guitar.strum_events(arr):
            acc.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
        pm.instruments.append(acc)
    return pm, arr


def melody_to_midi(melody, path, instrument="piano", backing=True):
    pm, arr = build_midi(melody, instrument, backing)
    pm.write(str(path))
    return path, arr
