"""One audio decoder for the toolkit: PyAV (already a dependency) straight to float32.

torchaudio.load (torchcodec backend) drops every other block of Microsoft ADPCM WAVs:
half the samples come back under the original sample-rate label, so the audio plays
at 2x speed with no error. PyAV decodes them correctly, and it opens video containers
too, so audio files and video soundtracks share this one path.
"""
import av
import numpy as np
import torch


def load_audio(path: str):
    """Decode the first audio stream of ``path`` -> ([channels, samples] float32, sample_rate)."""
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp")  # planar float32, native layout and rate
        chunks = []
        for frame in container.decode(stream):
            chunks += [f.to_ndarray() for f in resampler.resample(frame)]
        chunks += [f.to_ndarray() for f in resampler.resample(None)]
    if not chunks:
        raise ValueError(f"no audio decoded from {path}")
    return torch.from_numpy(np.concatenate(chunks, axis=1)), stream.codec_context.sample_rate
