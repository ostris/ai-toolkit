"""SheetSage2-Refined: lead sheets from audio with a fine-tuned SheetSage2.

Beats, downbeats and meters, melody, detailed chords (bass + root x 62 qualities) and keys (with
modes and modulations) for whole songs.  See ``pipeline.SheetSage2Refined`` and ``python -m sheetsage2_refined``.
Without the decoder: ``beats.HeadBeats``, ``melody.HeadMelody``, ``sections.HeadSections`` and ``keys.HeadKeys``.
"""

from .pipeline import SheetSage2Refined
from .transcribe import events_to_annotations

__all__ = ["SheetSage2Refined", "events_to_annotations"]
