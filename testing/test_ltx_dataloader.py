import time

from torch.utils.data import DataLoader
import sys
import os
import argparse
from tqdm import tqdm
import torch
from torchvision.io import write_video
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.data_loader import get_dataloader_from_datasets, trigger_dataloader_setup_epoch
from toolkit.config_modules import DatasetConfig

parser = argparse.ArgumentParser()
# parser.add_argument('dataset_folder', type=str, default='input')
parser.add_argument('dataset_folder', type=str)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=121)
parser.add_argument('--output_path', type=str, default='output/dataset_test')


args = parser.parse_args()

if args.output_path is None:
    raise ValueError('output_path is required for this test script')

if args.output_path is not None:
    args.output_path = os.path.abspath(args.output_path)
    os.makedirs(args.output_path, exist_ok=True)

dataset_folder = args.dataset_folder
resolution = 512
bucket_tolerance = 64
batch_size = 1
frame_rate = 24


## make fake sd
class FakeSD:
    def __init__(self):
        self.use_raw_control_images = False
    
    def encode_control_in_text_embeddings(self, *args, **kwargs):
        return None

    def get_bucket_divisibility(self):
        return 32

dataset_config = DatasetConfig(
    dataset_path=dataset_folder,
    resolution=resolution,
    default_caption='default',
    buckets=True,
    bucket_tolerance=bucket_tolerance,
    shrink_video_to_frames=True,
    num_frames=args.num_frames,
    do_i2v=True,
    fps=frame_rate,
    do_audio=True,
    debug=True,
    audio_preserve_pitch=False,
    audio_normalize=True

)

dataloader: DataLoader = get_dataloader_from_datasets([dataset_config], batch_size=batch_size, sd=FakeSD())


def _tensor_to_uint8_video(frames_fchw: torch.Tensor) -> torch.Tensor:
    """
    frames_fchw: [F, C, H, W] float/uint8
    returns: [F, H, W, C] uint8 on CPU
    """
    x = frames_fchw.detach()

    if x.dtype != torch.uint8:
        x = x.to(torch.float32)

        # Heuristic: if negatives exist, assume [-1,1] normalization; else assume [0,1]
        if torch.isfinite(x).all():
            if x.min().item() < 0.0:
                x = x * 0.5 + 0.5
        x = x.clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8)
    else:
        x = x.to(torch.uint8)

    # [F,C,H,W] -> [F,H,W,C]
    x = x.permute(0, 2, 3, 1).contiguous().cpu()
    return x


def _mux_with_ffmpeg(video_in: str, wav_in: str, mp4_out: str):
    # Copy video stream, encode audio to AAC, align to shortest
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_in,
            "-i",
            wav_in,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            mp4_out,
        ],
        check=True,
    )


# run through an epoch ang check sizes
dataloader_iterator = iter(dataloader)
idx = 0
for epoch in range(args.epochs):
    for batch in tqdm(dataloader):
        batch: 'DataLoaderBatchDTO'
        img_batch = batch.tensor
        frames = 1
        if len(img_batch.shape) == 5:
            frames = img_batch.shape[1]
            batch_size, frames, channels, height, width = img_batch.shape
        else:
            batch_size, channels, height, width = img_batch.shape
        
        # load audio
        audio_tensor = batch.audio_tensor # all file items contatinated on the batch dimension
        audio_data = batch.audio_data  # list of raw audio data per item in the batch

        # llm save the videos here with audio and video as mp4
        fps = getattr(dataset_config, "fps", None)
        if fps is None or fps <= 0:
            fps = 1.0

        # Ensure we can iterate items even if batch_size > 1
        for b in range(batch_size):
            # Get per-item frames as [F,C,H,W]
            if len(img_batch.shape) == 5:
                frames_fchw = img_batch[b]
            else:
                # single image: [C,H,W] -> [1,C,H,W]
                frames_fchw = img_batch[b].unsqueeze(0)

            video_uint8 = _tensor_to_uint8_video(frames_fchw)
            out_mp4 = os.path.join(args.output_path, f"{idx:06d}_{b:02d}.mp4")

            # Pick audio for this item (prefer audio_data list; fallback to audio_tensor)
            item_audio = None
            item_sr = None

            if isinstance(audio_data, (list, tuple)) and len(audio_data) > b:
                ad = audio_data[b]
                if isinstance(ad, dict) and ("waveform" in ad) and ("sample_rate" in ad) and ad["waveform"] is not None:
                    item_audio = ad["waveform"]
                    item_sr = int(ad["sample_rate"])
            elif audio_tensor is not None and torch.is_tensor(audio_tensor):
                # audio_tensor expected [B, C, L] (or [C,L] if batch collate differs)
                if audio_tensor.dim() == 3 and audio_tensor.shape[0] > b:
                    item_audio = audio_tensor[b]
                elif audio_tensor.dim() == 2 and b == 0:
                    item_audio = audio_tensor
                if item_audio is not None:
                    # best-effort sample rate from audio_data if present but not per-item dict
                    if isinstance(audio_data, dict) and "sample_rate" in audio_data:
                        try:
                            item_sr = int(audio_data["sample_rate"])
                        except Exception:
                            item_sr = None

            # Write mp4 (with audio if available) using ffmpeg muxing (torchvision audio muxing is unreliable)
            tmp_video = out_mp4 + ".tmp_video.mp4"
            tmp_wav = out_mp4 + ".tmp_audio.wav"
            try:
                # Always write video-only first
                write_video(tmp_video, video_uint8, fps=float(fps), video_codec="libx264")

                if item_audio is not None and item_sr is not None and item_audio.numel() > 0:
                    import torchaudio

                    wav = item_audio.detach()
                    # torchaudio.save expects [channels, samples]
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)
                    torchaudio.save(tmp_wav, wav.cpu().to(torch.float32), int(item_sr))

                    # Mux to final mp4
                    _mux_with_ffmpeg(tmp_video, tmp_wav, out_mp4)
                else:
                    # No audio: just move video into place
                    os.replace(tmp_video, out_mp4)

            except Exception as e:
                # Best-effort fallback: leave a playable video-only file
                try:
                    if os.path.exists(tmp_video):
                        os.replace(tmp_video, out_mp4)
                    else:
                        write_video(out_mp4, video_uint8, fps=float(fps), video_codec="libx264")
                except Exception:
                    raise

                if hasattr(dataset_config, 'debug') and dataset_config.debug:
                    print(f"Warning: failed to mux audio into mp4 for {out_mp4}: {e}")

            finally:
                # Cleanup temps (don't leave separate wavs lying around)
                try:
                    if os.path.exists(tmp_video):
                        os.remove(tmp_video)
                except Exception:
                    pass
                try:
                    if os.path.exists(tmp_wav):
                        os.remove(tmp_wav)
                except Exception:
                    pass

            time.sleep(0.2)

        idx += 1
    # if not last epoch
    if epoch < args.epochs - 1:
        trigger_dataloader_setup_epoch(dataloader)

print('done')
