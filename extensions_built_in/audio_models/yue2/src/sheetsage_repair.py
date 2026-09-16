"""Lenient ABC rebuild for SheetSage2 transcriptions whose strict notation export failed.

Upstream's ``export_result`` rejects the whole sheet on any inconsistency in the decoded events
(non-consecutive beat IDs in a bar, a note shorter than a subbeat, overlapping quantized notes,
a chord change shorter than the grid, an unparsable chord symbol). The decoded events are still
mostly right, so this rebuilds the same inputs with the offending rows repaired or dropped and
runs upstream's own ``generate_abc_from_data`` again, so the result is still validated ABC.
A quality gate refuses sheets that needed more than ``MAX_REPAIR_FRACTION`` of beats or notes touched.
"""

import importlib
from typing import List, Tuple

import numpy as np

MAX_REPAIR_FRACTION = 0.2


class AbcRepairError(ValueError):
    pass


def _modules(model):
    pkg = type(model).__module__.rsplit(".", 1)[0]
    return importlib.import_module(pkg + ".notation_sheetsage2"), importlib.import_module(pkg + ".exports_sheetsage2"), importlib.import_module(pkg + ".midi_sheetsage2")


def _beat_rows(events) -> Tuple[List[list], int]:
    """exports.rhythm_rows without raising on off-grid eighth positions; returns rows + skipped count."""
    from fractions import Fraction

    rows, meter, skipped = [], None, 0
    for event in events:
        rhythm = event["values"].get("rhythm", {})
        meter = rhythm.get("meter", meter)
        eighth = rhythm.get("eighth_position")
        if eighth is None or meter is None:
            continue
        position = Fraction(int(eighth) * int(meter[1]), 8)
        if position.denominator != 1 or not 0 <= position < meter[0]:
            skipped += 1
            continue
        rows.append([float(event["time"]), int(position) + 1, int(meter[0]), int(meter[1])])
    return rows, skipped


def _repair_beats(rows: List[list]) -> Tuple[List[list], int]:
    """Strictly increasing times; consecutive beat IDs inside each bar. A repeated ID within half a beat
    period is a duplicated row (dropped); any other non-increase starts a new bar; skipped IDs are filled."""
    if len(rows) < 2:
        raise AbcRepairError("fewer than two decoded beats")
    period = float(np.median(np.diff([r[0] for r in rows]))) if len(rows) > 2 else 0.5
    out, changed = [], 0
    for row in rows:
        if out and row[0] <= out[-1][0]:
            changed += 1
            continue
        if out and row[1] == out[-1][1] and row[0] - out[-1][0] < 0.5 * period:
            changed += 1
            continue
        if out:
            expected = 1 if row[1] <= out[-1][1] else out[-1][1] + 1
            if expected > row[2]:  # ran past the declared numerator: the model missed a downbeat
                expected = 1
            if expected != row[1]:
                changed += 1
                row = [row[0], expected, row[2], row[3]]
        out.append(row)
    if all(r[1] != 1 for r in out):
        raise AbcRepairError("no downbeat in the decoded beats")
    return out, changed


def _extend_beats(rows, duration, notes):
    """Same tail extension as export_result: continue the final tempo to the audio/last note end."""
    period = float(np.median(np.diff([b[0] for b in rows[-9:]])))
    if period <= 0:
        raise AbcRepairError("decoded beats must increase in time")
    beats = [list(b) for b in rows]
    end = max(duration, max((n[1] for n in notes), default=0))
    while beats[-1][0] < end - 1e-6:
        prev = beats[-1]
        beats.append([prev[0] + period, prev[1] % prev[2] + 1, prev[2], prev[3]])
    return beats


def _merge_short_intervals(rows, subbeat_times, quantize) -> Tuple[List[list], int]:
    """Rows that quantize to zero subbeats are absorbed into the previous row (or the next, for the first)."""
    out, merged = [], 0
    for row in rows:
        start_t, end_t = quantize(row[0], subbeat_times), quantize(row[1], subbeat_times)
        if end_t <= start_t:
            merged += 1
            if out:
                out[-1][1] = max(out[-1][1], row[1])
            continue
        if out and out[-1][1] > row[0]:  # keep contiguity after a merge
            row = [out[-1][1], row[1], row[2]]
            if row[1] <= row[0]:
                merged += 1
                continue
        out.append(list(row))
    return out, merged


def _filter_notes(notes, subbeat_times, notation) -> Tuple[List[list], int]:
    """Drop notes that quantize to nothing or land on subbeats another note of the same track already holds."""
    boundaries = notation._subbeat_boundaries(subbeat_times)
    kept, dropped = [], 0
    for track in (0, 1):
        taken = np.zeros(len(subbeat_times), dtype=bool)
        for note in sorted((n for n in notes if n[3] == track), key=lambda n: (n[0], n[1], n[2])):
            start_t = max(0, min(int(np.searchsorted(boundaries, note[0])), len(taken) - 1))
            end_t = max(0, min(int(np.searchsorted(boundaries, note[1])), len(taken) - 1))
            if end_t <= start_t or taken[start_t:end_t].any():
                dropped += 1
                continue
            taken[start_t:end_t] = True
            kept.append(list(note))
    return sorted(kept), dropped


def repair_abc(model, result: dict, melody_only: bool = False) -> Tuple[str, str]:
    """Rebuild ABC for a transcribe() result whose ``abc`` is None. Returns (abc, summary) or raises AbcRepairError."""
    notation, exports, midi_mod = _modules(model)
    events = result["events"]
    duration = float(result["duration_seconds"])
    notes = []
    for event in events:
        start = float(event["time"])
        for note in event["values"].get("melody", ()):
            end = min(duration, float(note["end_time"]))
            if end > start:
                notes.append([start, end, int(note["pitch"]), int(note["track"])])
    notes.sort()
    clean, _ = exports.notation_notes(notes)

    rows, skipped = _beat_rows(events)
    rows, changed = _repair_beats(rows)
    beats = _extend_beats(rows, duration, clean)
    n_beats = len(rows) + skipped
    if (changed + skipped) > MAX_REPAIR_FRACTION * n_beats:
        raise AbcRepairError(f"{changed + skipped}/{n_beats} beat rows needed repair")

    parsed = notation._parse_beats(notation._row_entries([[str(x) for x in b] for b in beats], "beats"), "beats")
    measures, _ = notation.infer_measures(parsed, meter_conflict="infer")
    subbeat_times, _, _ = notation._build_grid(parsed, measures)

    intervals = {}
    merged = 0
    for field in ("chord", "key", "structure"):
        rows = exports.interval_rows(events, field, duration)
        rows = [[max(beats[0][0], a), min(beats[-1][0], b), v] for a, b, v in rows if b > beats[0][0] and a < beats[-1][0]]
        if field != "structure":
            rows, m = _merge_short_intervals(rows, subbeat_times, notation._quantize_time)
            merged += m
        intervals[field] = rows
    if not intervals["key"]:
        raise AbcRepairError("no key was decoded")
    bad_chords = 0
    for row in intervals["chord"]:
        try:
            notation.chord_symbol_to_abc(row[2])
        except notation.AbcRebuildError:
            row[2] = "N"
            bad_chords += 1
    keys = []
    for row in intervals["key"]:
        try:
            notation.key_symbol_to_abc(row[2])
            keys.append(row)
        except notation.AbcRebuildError:
            if keys:
                keys[-1][1] = row[1]
    if not keys:
        raise AbcRepairError("no usable key")
    intervals["key"] = keys

    kept, dropped = _filter_notes(clean, subbeat_times, notation)
    if dropped > MAX_REPAIR_FRACTION * max(1, len(clean)):
        raise AbcRepairError(f"{dropped}/{len(clean)} notes could not be placed on the beat grid")

    try:
        text, _ = notation.generate_abc_from_data(
            midi_mod.midi_bytes(exports._midi(kept)), beats, intervals["chord"], intervals["key"], intervals["structure"], melody_only=melody_only
        )
    except ValueError as exc:
        raise AbcRepairError(f"rebuild still failed: {exc}") from exc
    summary = f"beats repaired {changed + skipped}/{n_beats}, notes dropped {dropped}/{len(clean)}, intervals merged {merged}, chords blanked {bad_chords}"
    return text, summary
