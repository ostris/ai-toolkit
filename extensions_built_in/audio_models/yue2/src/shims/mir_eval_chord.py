"""Chord-label helpers vendored from mir_eval (mir_eval/chord.py), so the SheetSage2 remote code
runs without the mir_eval package. Unmodified except for the removed module preamble.

The MIT License (MIT)

Copyright (c) 2014 Colin Raffel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import re
import warnings

import numpy as np


BITMAP_LENGTH = 12
NO_CHORD = "N"
NO_CHORD_ENCODED = -1, np.array([0] * BITMAP_LENGTH), -1
X_CHORD = "X"
X_CHORD_ENCODED = -1, np.array([-1] * BITMAP_LENGTH), -1


class InvalidChordException(Exception):
    r"""Exception class for suspect / invalid chord labels"""

    def __init__(self, message="", chord_label=None):
        self.message = message
        self.chord_label = chord_label
        self.name = self.__class__.__name__
        super().__init__(message)


# --- Chord Primitives ---
def _pitch_classes():
    r"""Map from pitch class (str) to semitone (int)."""
    pitch_classes = ["C", "D", "E", "F", "G", "A", "B"]
    semitones = [0, 2, 4, 5, 7, 9, 11]
    return {c: s for c, s in zip(pitch_classes, semitones)}


def _scale_degrees():
    r"""Map scale degrees (str) to semitones (int)."""
    degrees = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
    semitones = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 17, 19, 21]
    return {d: s for d, s in zip(degrees, semitones)}


# Maps pitch classes (strings) to semitone indexes (ints).
PITCH_CLASSES = _pitch_classes()


def pitch_class_to_semitone(pitch_class):
    r"""Convert a pitch class to semitone.

    Parameters
    ----------
    pitch_class : str
        Spelling of a given pitch class, e.g. 'C#', 'Gbb'

    Returns
    -------
    semitone : int
        Semitone value of the pitch class.

    """
    semitone = 0
    for idx, char in enumerate(pitch_class):
        if char == "#" and idx > 0:
            semitone += 1
        elif char == "b" and idx > 0:
            semitone -= 1
        elif idx == 0:
            semitone = PITCH_CLASSES.get(char)
        else:
            raise InvalidChordException(
                "Pitch class improperly formed: %s" % pitch_class
            )
    return semitone % 12


# Maps scale degrees (strings) to semitone indexes (ints).
SCALE_DEGREES = _scale_degrees()


def scale_degree_to_semitone(scale_degree):
    r"""Convert a scale degree to semitone.

    Parameters
    ----------
    scale_degree : str
        Spelling of a relative scale degree, e.g. 'b3', '7', '#5'

    Returns
    -------
    semitone : int
        Relative semitone of the scale degree, wrapped to a single octave

    Raises
    ------
    InvalidChordException if `scale_degree` is invalid.
    """
    semitone = 0
    offset = 0
    if scale_degree.startswith("#"):
        offset = scale_degree.count("#")
        scale_degree = scale_degree.strip("#")
    elif scale_degree.startswith("b"):
        offset = -1 * scale_degree.count("b")
        scale_degree = scale_degree.strip("b")

    semitone = SCALE_DEGREES.get(scale_degree, None)
    if semitone is None:
        raise InvalidChordException(
            "Scale degree improperly formed: {}, expected one of {}.".format(
                scale_degree, list(SCALE_DEGREES.keys())
            )
        )
    return semitone + offset


def scale_degree_to_bitmap(scale_degree, modulo=False, length=BITMAP_LENGTH):
    """Create a bitmap representation of a scale degree.

    Note that values in the bitmap may be negative, indicating that the
    semitone is to be removed.

    Parameters
    ----------
    scale_degree : str
        Spelling of a relative scale degree, e.g. 'b3', '7', '#5'
    modulo : bool, default=True
        If a scale degree exceeds the length of the bit-vector, modulo the
        scale degree back into the bit-vector; otherwise it is discarded.
    length : int, default=12
        Length of the bit-vector to produce

    Returns
    -------
    bitmap : np.ndarray, in [-1, 0, 1], len=`length`
        Bitmap representation of this scale degree.
    """
    sign = 1
    if scale_degree.startswith("*"):
        sign = -1
        scale_degree = scale_degree.strip("*")
    edit_map = [0] * length
    sd_idx = scale_degree_to_semitone(scale_degree)
    if sd_idx < length or modulo:
        edit_map[sd_idx % length] = sign
    return np.array(edit_map)


# Maps quality strings to bitmaps, corresponding to relative pitch class
# semitones, i.e. vector[0] is the tonic.
QUALITIES = {
    #           1     2     3     4  5     6     7
    "maj": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "aug": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "dim": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    "sus2": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "minmaj7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    "maj6": [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    "min6": [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    "dim7": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "hdim7": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    "aug7": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    "maj9": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min9": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "9": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "min11": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "11": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "maj13": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min13": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "13": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "1": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "5": [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}


def quality_to_bitmap(quality):
    """Return the bitmap for a given quality.

    Parameters
    ----------
    quality : str
        Chord quality name.

    Returns
    -------
    bitmap : np.ndarray
        Bitmap representation of this quality (12-dim).

    """
    if quality not in QUALITIES:
        raise InvalidChordException(
            "Unsupported chord quality shorthand: '%s' "
            "Did you mean to reduce extended chords?" % quality
        )
    return np.array(QUALITIES[quality])


# Maps extended chord qualities to the subset above, translating additional
# voicings to extensions as a set of scale degrees (strings).
# TODO(ejhumphrey): Revisit how minmaj7's are mapped. This is how TMC did it,
#   but MMV handles it like a separate quality (rather than an add7).
EXTENDED_QUALITY_REDUX = {
    "minmaj7": ("min", {"7"}),
    "maj9": ("maj7", {"9"}),
    "min9": ("min7", {"9"}),
    "9": ("7", {"9"}),
    "b9": ("7", {"b9"}),
    "#9": ("7", {"#9"}),
    "11": ("7", {"9", "11"}),
    "#11": ("7", {"9", "#11"}),
    "13": ("7", {"9", "11", "13"}),
    "b13": ("7", {"9", "11", "b13"}),
    "min11": ("min7", {"9", "11"}),
    "maj13": ("maj7", {"9", "11", "13"}),
    "min13": ("min7", {"9", "11", "13"}),
}


def reduce_extended_quality(quality):
    """Map an extended chord quality to a simpler one, moving upper voices to
    a set of scale degree extensions.

    Parameters
    ----------
    quality : str
        Extended chord quality to reduce.

    Returns
    -------
    base_quality : str
        New chord quality.
    extensions : set
        Scale degrees extensions for the quality.

    """
    return EXTENDED_QUALITY_REDUX.get(quality, (quality, set()))


# --- Chord Label Parsing ---
# This monster regexp is pulled from the JAMS chord namespace,
# which is in turn derived from the context-free grammar of
# Harte et al., 2005.
CHORD_RE = re.compile(
    r"""^((N|X)|(([A-G](b*|#*))((:(maj|min|dim|aug|1|5|sus2|sus4|maj6|min6|7|maj7|min7|dim7|hdim7|minmaj7|aug7|9|maj9|min9|11|maj11|min11|13|maj13|min13)(\((\*?((b*|#*)([1-9]|1[0-3]?))(,\*?((b*|#*)([1-9]|1[0-3]?)))*)\))?)|(:\((\*?((b*|#*)([1-9]|1[0-3]?))(,\*?((b*|#*)([1-9]|1[0-3]?)))*)\)))?((/((b*|#*)([1-9]|1[0-3]?)))?)?))$"""
)  # nopep8


def validate_chord_label(chord_label):
    """Test for well-formedness of a chord label.

    Parameters
    ----------
    chord_label : str
        Chord label to validate.
    """
    if not CHORD_RE.match(chord_label):
        raise InvalidChordException("Invalid chord label: " "{}".format(chord_label))
    pass


def split(chord_label, reduce_extended_chords=False):
    """Parse a chord label into its four constituent parts:
        - root
        - quality shorthand
        - scale degrees
        - bass

    Note: Chords lacking quality AND interval information are major.
      - If a quality is specified, it is returned.
      - If an interval is specified WITHOUT a quality, the quality field is
        empty.

    Some examples::

        'C' -> ['C', 'maj', {}, '1']
        'G#:min(*b3,*5)/5' -> ['G#', 'min', {'*b3', '*5'}, '5']
        'A:(3)/6' -> ['A', '', {'3'}, '6']

    Parameters
    ----------
    chord_label : str
        A chord label.
    reduce_extended_chords : bool
        Whether to map the upper voicings of extended chords (9's, 11's, 13's)
        to semitone extensions. (Default value = False)

    Returns
    -------
    chord_parts : list
        Split version of the chord label.

    """
    chord_label = str(chord_label)
    validate_chord_label(chord_label)
    if chord_label == NO_CHORD:
        return [chord_label, "", set(), ""]

    bass = "1"
    if "/" in chord_label:
        chord_label, bass = chord_label.split("/")

    scale_degrees = set()
    omission = False
    if "(" in chord_label:
        chord_label, scale_degrees = chord_label.split("(")
        omission = "*" in scale_degrees
        scale_degrees = scale_degrees.strip(")")
        scale_degrees = {i.strip() for i in scale_degrees.split(",")}

    # Note: Chords lacking quality AND added interval information are major.
    #   If a quality shorthand is specified, it is returned.
    #   If an interval is specified WITHOUT a quality, the quality field is
    #     empty.
    #   Intervals specifying omissions MUST have a quality.
    if omission and ":" not in chord_label:
        raise InvalidChordException(
            "Intervals specifying omissions MUST have a quality."
        )
    quality = "" if scale_degrees else "maj"
    if ":" in chord_label:
        chord_root, quality_name = chord_label.split(":")
        # Extended chords (with ":"s) may not explicitly have Major qualities,
        # so only overwrite the default if the string is not empty.
        if quality_name:
            quality = quality_name.lower()
    else:
        chord_root = chord_label

    if reduce_extended_chords:
        quality, addl_scale_degrees = reduce_extended_quality(quality)
        scale_degrees.update(addl_scale_degrees)

    return [chord_root, quality, scale_degrees, bass]


def join(chord_root, quality="", extensions=None, bass=""):
    r"""Join the parts of a chord into a complete chord label.

    Parameters
    ----------
    chord_root : str
        Root pitch class of the chord, e.g. 'C', 'Eb'
    quality : str
        Quality of the chord, e.g. 'maj', 'hdim7'
        (Default value = '')
    extensions : list
        Any added or absent scaled degrees for this chord, e.g. ['4', '\*3']
        (Default value = None)
    bass : str
        Scale degree of the bass note, e.g. '5'.
        (Default value = '')

    Returns
    -------
    chord_label : str
        A complete chord label.

    """
    chord_label = chord_root
    if quality or extensions:
        chord_label += ":%s" % quality
    if extensions:
        chord_label += "(%s)" % ",".join(extensions)
    if bass and bass != "1":
        chord_label += "/%s" % bass
    validate_chord_label(chord_label)
    return chord_label


# --- Chords to Numerical Representations ---
def encode(chord_label, reduce_extended_chords=False, strict_bass_intervals=False):
    """Translate a chord label to numerical representations for evaluation.

    Parameters
    ----------
    chord_label : str
        Chord label to encode.
    reduce_extended_chords : bool
        Whether to map the upper voicings of extended chords (9's, 11's, 13's)
        to semitone extensions.
        (Default value = False)
    strict_bass_intervals : bool
        Whether to require that the bass scale degree is present in the chord.
        (Default value = False)

    Returns
    -------
    root_number : int
        Absolute semitone of the chord's root.
    semitone_bitmap : np.ndarray, dtype=int
        12-dim vector of relative semitones in the chord spelling.
    bass_number : int
        Relative semitone of the chord's bass note, e.g. 0=root, 7=fifth, etc.
    """
    if chord_label == NO_CHORD:
        return NO_CHORD_ENCODED
    if chord_label == X_CHORD:
        return X_CHORD_ENCODED
    chord_root, quality, scale_degrees, bass = split(
        chord_label, reduce_extended_chords=reduce_extended_chords
    )

    root_number = pitch_class_to_semitone(chord_root)
    bass_number = scale_degree_to_semitone(bass) % 12

    semitone_bitmap = quality_to_bitmap(quality)
    semitone_bitmap[0] = 1

    for scale_degree in scale_degrees:
        semitone_bitmap += scale_degree_to_bitmap(scale_degree, reduce_extended_chords)

    semitone_bitmap = (semitone_bitmap > 0).astype(np.int64)
    if not semitone_bitmap[bass_number] and strict_bass_intervals:
        raise InvalidChordException(
            "Given bass scale degree is absent from this chord: " "%s" % chord_label,
            chord_label,
        )
    else:
        semitone_bitmap[bass_number] = 1
    return root_number, semitone_bitmap, bass_number


def encode_many(chord_labels, reduce_extended_chords=False):
    """Translate a set of chord labels to numerical representations for sane
    evaluation.

    Parameters
    ----------
    chord_labels : list
        Set of chord labels to encode.
    reduce_extended_chords : bool
        Whether to map the upper voicings of extended chords (9's, 11's, 13's)
        to semitone extensions.
        (Default value = False)

    Returns
    -------
    root_number : np.ndarray, dtype=int
        Absolute semitone of the chord's root.
    interval_bitmap : np.ndarray, dtype=int
        12-dim vector of relative semitones in the given chord quality.
    bass_number : np.ndarray, dtype=int
        Relative semitones of the chord's bass notes.

    """
    num_items = len(chord_labels)
    roots, basses = np.zeros([2, num_items], dtype=np.int64)
    semitones = np.zeros([num_items, 12], dtype=np.int64)
    local_cache = dict()
    for i, label in enumerate(chord_labels):
        result = local_cache.get(label, None)
        if result is None:
            result = encode(label, reduce_extended_chords)
            local_cache[label] = result
        roots[i], semitones[i], basses[i] = result
    return roots, semitones, basses


def rotate_bitmap_to_root(bitmap, chord_root):
    """Circularly shift a relative bitmap to its absolute pitch classes.

    For clarity, the best explanation is an example. Given 'G:Maj', the root
    and quality map are as follows::

        root=5
        quality=[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # Relative chord shape

    After rotating to the root, the resulting bitmap becomes::

        abs_quality = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]  # G, B, and D

    Parameters
    ----------
    bitmap : np.ndarray, shape=(12,)
        Bitmap of active notes, relative to the given root.
    chord_root : int
        Absolute pitch class number.

    Returns
    -------
    bitmap : np.ndarray, shape=(12,)
        Absolute bitmap of active pitch classes.

    """
    bitmap = np.asarray(bitmap)
    assert bitmap.ndim == 1, "Currently only 1D bitmaps are supported."
    idxs = list(np.nonzero(bitmap))
    idxs[-1] = (idxs[-1] + chord_root) % 12
    abs_bitmap = np.zeros_like(bitmap)
    abs_bitmap[tuple(idxs)] = 1
    return abs_bitmap
