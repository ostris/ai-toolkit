"""Minimal in-memory stand-in for the parts of pretty_midi that SheetSage2's ABC path uses:
PrettyMIDI(resolution=...) / PrettyMIDI(bytes-like), .instruments, .resolution, .write(stream);
Instrument(program, name=...) with .notes/.name/.program/.is_drum; Note(velocity, pitch, start, end).

"Bytes" are our own serialization (the only consumer is this shim), but note times are rounded to
MIDI ticks on write exactly as a real file round trip would (120 bpm default tempo: ticks per second
= 2 * resolution), because the ABC builder is documented to follow the serialized MIDI timing."""

import json
from io import BytesIO

_MAGIC = b"AITK-MIDI-JSON\n"


class Note:
    def __init__(self, velocity, pitch, start, end):
        self.velocity = int(velocity)
        self.pitch = int(pitch)
        self.start = float(start)
        self.end = float(end)

    def __repr__(self):
        return f"Note(start={self.start:.6f}, end={self.end:.6f}, pitch={self.pitch}, velocity={self.velocity})"


class Instrument:
    def __init__(self, program, is_drum=False, name=""):
        self.program = int(program)
        self.is_drum = bool(is_drum)
        self.name = name
        self.notes = []


class PrettyMIDI:
    def __init__(self, midi_file=None, resolution=220, initial_tempo=120.0):
        self.resolution = int(resolution)
        self.initial_tempo = float(initial_tempo)
        self.instruments = []
        if midi_file is not None:
            if isinstance(midi_file, (bytes, bytearray)):
                data = bytes(midi_file)
            elif hasattr(midi_file, "read"):
                data = midi_file.read()
            else:
                with open(str(midi_file), "rb") as f:
                    data = f.read()
            if not data.startswith(_MAGIC):
                raise ValueError("pretty_midi shim can only read MIDI written by this shim")
            payload = json.loads(data[len(_MAGIC) :].decode("utf-8"))
            self.resolution = int(payload["resolution"])
            self.initial_tempo = float(payload["initial_tempo"])
            for inst in payload["instruments"]:
                instrument = Instrument(inst["program"], inst["is_drum"], inst["name"])
                instrument.notes = [Note(n["velocity"], n["pitch"], n["start"], n["end"]) for n in inst["notes"]]
                self.instruments.append(instrument)

    def _ticks_per_second(self):
        return self.resolution * self.initial_tempo / 60.0

    def _quantize(self, seconds):
        tps = self._ticks_per_second()
        return int(round(seconds * tps)) / tps

    def write(self, filename):
        payload = {
            "resolution": self.resolution,
            "initial_tempo": self.initial_tempo,
            "instruments": [
                {
                    "program": i.program,
                    "is_drum": i.is_drum,
                    "name": i.name,
                    "notes": [
                        {"velocity": n.velocity, "pitch": n.pitch, "start": self._quantize(n.start), "end": self._quantize(n.end)}
                        for n in i.notes
                    ],
                }
                for i in self.instruments
            ],
        }
        data = _MAGIC + json.dumps(payload).encode("utf-8")
        if hasattr(filename, "write"):
            filename.write(data)
        else:
            with open(str(filename), "wb") as f:
                f.write(data)

    def get_end_time(self):
        return max((n.end for i in self.instruments for n in i.notes), default=0.0)
