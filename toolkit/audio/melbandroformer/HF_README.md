---
license: mit
pipeline_tag: audio-to-audio
tags:
  - music-source-separation
  - vocals
  - mel-band-roformer
  - ai-toolkit
---

# Mel-Band RoFormer (vocals) for ai-toolkit

Safetensors repack of Kimberley Jensen's Mel-Band RoFormer vocal separation model, used by
[ai-toolkit](https://github.com/ostris/ai-toolkit) to split songs into a vocals track and an
instrumental track (`instrumental = mix - vocals`).

**These weights are not ours.** All credit goes to the original authors:

- Weights: [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) (`MelBandRoformer.ckpt`, MIT) by Kimberley Jensen
- Training framework and config: [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
  ([`config_vocals_mel_band_roformer_kj.yaml`](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml)) by Roman Solovyev
- Architecture implementation: [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) by Phil Wang
- Paper: [Mel-Band RoFormer for Music Source Separation](https://arxiv.org/abs/2310.01809), Wang et al. 2023

## Files

| file | notes |
|---|---|
| `melbandroformer_vocals_kj.safetensors` | fp32, byte-identical tensors to the original `.ckpt`. Model kwargs and inference defaults (`chunk_size`, `num_overlap`) are stored in the safetensors metadata, so no separate config is needed. |

Conversion script: [`scripts/convert_melbandroformer.py`](https://github.com/ostris/ai-toolkit/blob/main/scripts/convert_melbandroformer.py).

## Usage

ai-toolkit downloads this file automatically on first use:

```bash
python -m toolkit.audio.melbandroformer song.flac
# -> song_vocals.flac, song_instrumental.flac
```

```python
from toolkit.audio.melbandroformer import load_melbandroformer, separate
model = load_melbandroformer(device="cuda", compile=True)
vocals, instrumental = separate(model, wav, sample_rate)  # wav: [channels, samples]
```

Input: mono or stereo at any sample rate (resampled to 44.1 kHz internally, output at the input rate).

## License

MIT, same as the original weights and code.
