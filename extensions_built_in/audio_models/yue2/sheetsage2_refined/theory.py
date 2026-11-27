"""Pitch names, Tsumugi chord qualities and key modes shared by the readers and the vocabulary."""

from __future__ import annotations

PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
    "B#": 0,
}
SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
DEGREES = ("1", "b2", "2", "b3", "3", "4", "b5", "5", "b6", "6", "b7", "7")

# The 62 chord qualities of chord_dataset/quality.json, in its id order ("N" is
# separate).  Token ids follow this order, so it is frozen.
QUALITIES = (
    "5",
    "",
    "-5",
    "m",
    "dim",
    "aug",
    "sus2",
    "sus4",
    "6",
    "7",
    "M7",
    "m6",
    "m7",
    "mM7",
    "7-5",
    "m7-5",
    "aug7",
    "augM7",
    "7sus4",
    "dim7",
    "add9",
    "madd9",
    "69",
    "7(9)",
    "7(13)",
    "7(b9)",
    "7(#9)",
    "7(#11)",
    "7(b13)",
    "7-5(b13)",
    "M7(9)",
    "M7(b9)",
    "M7(13)",
    "M7(#11)",
    "m69",
    "m7(9)",
    "m7(11)",
    "m7(13)",
    "m7(b9)",
    "mM7(9)",
    "mM7(13)",
    "aug7(9)",
    "augM7(#9)",
    "add9(#11)",
    "m7-5(b9)",
    "7(9,13)",
    "7(9,b13)",
    "7(9,#11)",
    "7(b9,#9)",
    "7(b9,13)",
    "7(b9,b13)",
    "7(b9,#11)",
    "7(#9,13)",
    "7(#9,b13)",
    "7(#11,13)",
    "m7(9,11)",
    "m7(9,13)",
    "M7(9,13)",
    "M7(9,#11)",
    "7(9,#11,13)",
    "7(9,#11,b13)",
    "M7(9,#11,13)",
)

# Tsumugi quality -> (mir_eval shorthand, extra degrees).  Tensions written in
# parentheses after these bases, e.g. ``7(9,13)``, are appended to the extras.
QUALITY_BASE = {
    "": ("maj", ()),
    "m": ("min", ()),
    "5": ("5", ()),
    "-5": ("maj", ("*5", "b5")),
    "dim": ("dim", ()),
    "aug": ("aug", ()),
    "sus2": ("sus2", ()),
    "sus4": ("sus4", ()),
    "6": ("maj6", ()),
    "7": ("7", ()),
    "M7": ("maj7", ()),
    "m6": ("min6", ()),
    "m7": ("min7", ()),
    "mM7": ("minmaj7", ()),
    "7-5": ("7", ("*5", "b5")),
    "m7-5": ("hdim7", ()),
    "aug7": ("aug", ("b7",)),
    "augM7": ("aug", ("7",)),
    "7sus4": ("sus4", ("b7",)),
    "dim7": ("dim7", ()),
    "add9": ("maj", ("9",)),
    "madd9": ("min", ("9",)),
    "69": ("maj6", ("9",)),
    "m69": ("min6", ("9",)),
}


def normalize_quality(quality: str) -> str:
    return quality.replace(" ", "")


def chord_to_mir_eval(root: str | int | None, quality: str, bass: str | int | None) -> str:
    """Convert a Tsumugi chord to mir_eval syntax; ``root``/``bass`` are names or pitch classes."""

    if root in ("", "N", None) or quality == "N":
        return "N"
    root_pc = root if isinstance(root, int) else PITCH_CLASS[root]
    base, _, tensions = normalize_quality(quality).partition("(")
    shorthand, extras = QUALITY_BASE[base]
    degrees = [*extras, *(t for t in tensions.rstrip(")").split(",") if t)]
    label = f"{SHARP_NAMES[root_pc]}:{shorthand}" + (f"({','.join(degrees)})" if degrees else "")
    if bass not in (None, ""):
        interval = ((bass if isinstance(bass, int) else PITCH_CLASS[bass]) - root_pc) % 12
        if interval:
            label += "/" + DEGREES[interval]
    return label


# Key modes, named so that the two common ones match the coarse "C:major" /
# "C:minor" labels.  Each is identified by the steps between its scale degrees,
# as Hooktheory writes them.
MODES = (
    "major",
    "dorian",
    "phrygian",
    "lydian",
    "mixolydian",
    "minor",
    "locrian",
    "harmonic_minor",
    "phrygian_dominant",
)
MODE_OF_STEPS = {
    (2, 2, 1, 2, 2, 2): "major",
    (2, 1, 2, 2, 2, 1): "dorian",
    (1, 2, 2, 2, 1, 2): "phrygian",
    (2, 2, 2, 1, 2, 2): "lydian",
    (2, 2, 1, 2, 2, 1): "mixolydian",
    (2, 1, 2, 2, 1, 2): "minor",
    (1, 2, 2, 1, 2, 2): "locrian",
    (2, 1, 2, 2, 1, 3): "harmonic_minor",
    (1, 3, 1, 2, 1, 2): "phrygian_dominant",
}
# Tonalities that hand-reviewed key labels may name but key_mode has no token for
# (a blues / dominant tonality is not a mode).
LABEL_ONLY_TONALITIES = ("blues",)
# Modes with a major third count as "major" for the coarse key vocabulary.
MAJOR_LIKE = frozenset({"major", "lydian", "mixolydian", "blues"})


def coarse_key(key: str | None) -> str | None:
    """``"D:dorian"`` -> ``"D:minor"``; coarse keys and ``None`` pass through."""

    if key is None:
        return None
    tonic, mode = key.split(":")
    return f"{tonic}:{'major' if mode in MAJOR_LIKE else 'minor'}"


def chord_tone_templates():
    """``[12 * len(QUALITIES), 12]`` pitch-class sets of every root x quality, in chord_detail order."""

    import mir_eval
    import numpy as np

    templates = np.zeros((12 * len(QUALITIES), 12), dtype=np.float32)
    for index, quality in enumerate(QUALITIES):
        for root in range(12):
            chord_root, bits, _ = mir_eval.chord.encode(chord_to_mir_eval(root, quality, None), reduce_extended_chords=True)
            templates[12 * index + root] = np.roll(bits, chord_root)
    return templates
