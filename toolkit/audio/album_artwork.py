import io
import os
import numpy as np
import av
from PIL import Image, ImageDraw


ARTWORK_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATH = os.path.join(ARTWORK_DIR, "ostris_logo.jpg")
WAVEFORM_COLOR = (0xFB, 0xBF, 0x24, 230)  # #fbbf24 at 90% opacity
ARTWORK_SIZE = 1024


def load_waveform(audio_path: str, num_samples: int = 512) -> np.ndarray:
    """Load audio and return a downsampled waveform envelope using PyAV."""
    container = av.open(audio_path)
    stream = container.streams.audio[0]
    stream.codec_context.thread_type = "AUTO"

    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()
        # mix down to mono
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        frames.append(arr)
    container.close()

    audio = np.concatenate(frames)

    # downsample to num_samples bins by taking max absolute value per bin
    bin_size = len(audio) // num_samples
    if bin_size == 0:
        bin_size = 1
    trimmed = audio[: bin_size * num_samples]
    bins = trimmed.reshape(num_samples, bin_size)
    envelope = np.max(np.abs(bins), axis=1)

    # normalize to 0-1
    peak = envelope.max()
    if peak > 0:
        envelope = envelope / peak
    return envelope


def create_artwork(waveform: np.ndarray, size: int = ARTWORK_SIZE) -> Image.Image:
    """Create album artwork with logo background and waveform overlay."""
    bg = Image.open(BACKGROUND_PATH).convert("RGBA").resize((size, size), Image.LANCZOS)

    # draw waveform on separate overlay for alpha compositing
    wave_overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(wave_overlay)

    num_bars = len(waveform)
    padding = int(size * 0.02)
    draw_w = size - 2 * padding
    bar_width = max(1, draw_w / num_bars)
    center_y = size // 2

    max_amp = (size // 2) * 0.85  # leave a little margin

    for i, amp in enumerate(waveform):
        x = padding + i * bar_width
        h = amp * max_amp
        y_top = center_y - h
        y_bot = center_y + h
        draw.rectangle(
            [x, y_top, x + bar_width - 1, y_bot],
            fill=WAVEFORM_COLOR,
        )

    bg = Image.alpha_composite(bg, wave_overlay)
    return bg.convert("RGB")


def add_album_artwork(song_path: str) -> None:
    """Add album artwork with waveform visualization to an MP3 file."""
    from mutagen.id3 import ID3, APIC, ID3NoHeaderError

    if not os.path.isfile(song_path):
        raise FileNotFoundError(f"Audio file not found: {song_path}")

    waveform = load_waveform(song_path)
    artwork = create_artwork(waveform)

    # encode artwork to JPEG bytes in memory
    buf = io.BytesIO()
    artwork.save(buf, format="JPEG", quality=85)
    artwork_data = buf.getvalue()

    # embed into MP3 via mutagen ID3 tags
    try:
        tags = ID3(song_path)
    except ID3NoHeaderError:
        tags = ID3()

    tags.delall("APIC")
    tags.add(
        APIC(
            encoding=3,  # UTF-8
            mime="image/jpeg",
            type=3,  # front cover
            desc="Cover",
            data=artwork_data,
        )
    )
    tags.save(song_path, v2_version=3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add album artwork with waveform to an MP3 file"
    )
    parser.add_argument("mp3", help="Path to the MP3 file")
    args = parser.parse_args()

    add_album_artwork(args.mp3)
