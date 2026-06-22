import math
import torch
import torch.nn.functional as F
import torchaudio
    
def time_stretch_preserve_pitch(waveform: torch.Tensor, sample_rate: int, target_samples: int) -> torch.Tensor:
    """
    waveform: [C, L] float tensor (CPU or GPU)
    returns:  [C, target_samples] float tensor
    Pitch-preserving time stretch to match target_samples.
    """


    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.to(torch.float32)

    src_len = waveform.shape[-1]
    if src_len == 0 or target_samples <= 0:
        return waveform[..., :0]

    if src_len == target_samples:
        return waveform

    # rate > 1.0 speeds up (shorter), rate < 1.0 slows down (longer)
    rate = float(src_len) / float(target_samples)

    # Use sample_rate to pick STFT params
    win_seconds = 0.046
    hop_seconds = 0.0115

    n_fft_target = int(sample_rate * win_seconds)
    n_fft = 1 << max(8, int(math.floor(math.log2(max(256, n_fft_target)))))  # >=256, pow2
    win_length = n_fft
    hop_length = max(64, int(sample_rate * hop_seconds))
    hop_length = min(hop_length, win_length // 2)

    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )  # [C, F, T] complex

    # IMPORTANT: n_freq must match STFT's frequency bins (n_fft//2 + 1)
    stretcher = torchaudio.transforms.TimeStretch(
        n_freq=stft.shape[-2],
        hop_length=hop_length,
        fixed_rate=rate,
    ).to(waveform.device)

    stft_stretched = stretcher(stft)  # [C, F, T']

    stretched = torch.istft(
        stft_stretched,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=target_samples,
    )

    if stretched.shape[-1] > target_samples:
        stretched = stretched[..., :target_samples]
    elif stretched.shape[-1] < target_samples:
        stretched = F.pad(stretched, (0, target_samples - stretched.shape[-1]))

    return stretched
