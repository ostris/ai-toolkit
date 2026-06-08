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


def make_video(song_path: str, video_size: int = 512) -> str:
    """Create an MP4 video with album artwork as a static image for the duration of the audio."""
    if not os.path.isfile(song_path):
        raise FileNotFoundError(f"Audio file not found: {song_path}")

    waveform = load_waveform(song_path)
    artwork = create_artwork(waveform)
    artwork = artwork.resize((video_size, video_size), Image.LANCZOS)

    # get audio duration
    container = av.open(song_path)
    duration = float(container.duration) / av.time_base
    container.close()

    # output path: same name as input but .mp4, in the same directory
    base, _ = os.path.splitext(song_path)
    output_path = base + ".mp4"

    fps = 1  # static image, 1 fps is enough
    total_frames = max(1, int(duration * fps))

    # convert artwork to numpy array for video encoding
    frame_data = np.array(artwork)

    out_container = av.open(output_path, mode="w")
    video_stream = out_container.add_stream("libx264", rate=fps)
    video_stream.width = video_size
    video_stream.height = video_size
    video_stream.pix_fmt = "yuv420p"

    for _ in range(total_frames):
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in video_stream.encode(frame):
            out_container.mux(packet)

    # flush
    for packet in video_stream.encode():
        out_container.mux(packet)

    out_container.close()

    # mux audio into the video using ffmpeg via subprocess
    import subprocess
    final_path = base + "_final.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", output_path,
            "-i", song_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_path,
        ],
        check=True,
        capture_output=True,
    )

    # replace silent video with final muxed version
    os.replace(final_path, output_path)

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create an MP4 video with album artwork from an audio file"
    )
    parser.add_argument("audio", help="Path to the audio file")
    args = parser.parse_args()

    out = make_video(args.audio)
    print(f"Created video: {out}")
