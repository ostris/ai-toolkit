"""Convert an MSST / lucidrains Mel-Band RoFormer .ckpt into a self-describing safetensors file
(model kwargs + inference defaults in the metadata) that toolkit.audio.melbandroformer loads.

    python scripts/convert_melbandroformer.py                       # KimberleyJSN vocals model -> MODELS_PATH/checkpoints
    python scripts/convert_melbandroformer.py --ckpt x.ckpt --config msst_config.yaml --out x.safetensors
"""
import argparse
import inspect
import json
import os
import sys

TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLKIT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(TOOLKIT_ROOT, ".env"))  # MODELS_PATH etc., before toolkit.paths is imported

import torch
import yaml
from safetensors.torch import save_file

from toolkit.paths import MODELS_PATH
from toolkit.audio.melbandroformer.model import MelBandRoformer
from toolkit.audio.melbandroformer.separate import DEFAULT_WEIGHTS

# https://huggingface.co/KimberleyJSN/melbandroformer (MIT), config from
# MSST configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml
KJ_VOCALS = dict(
    repo_id="KimberleyJSN/melbandroformer",
    filename="MelBandRoformer.ckpt",
    license="MIT",
    model=dict(
        dim=384,
        depth=6,
        stereo=True,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0,
        ff_dropout=0,
        sample_rate=44100,
        stft_n_fft=2048,
        stft_hop_length=441,
        stft_win_length=2048,
        stft_normalized=False,
        mask_estimator_depth=2,
    ),
    inference=dict(chunk_size=352800, num_overlap=2),
    stems=["vocals"],
)


def config_from_msst_yaml(path):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # MSST configs carry !!python/tuple
    allowed = set(inspect.signature(MelBandRoformer.__init__).parameters) - {"self"}
    model = {k: v for k, v in cfg["model"].items() if k in allowed}
    dropped = sorted(set(cfg["model"]) - allowed)
    if dropped:
        print(f"ignoring non-inference model keys: {dropped}")
    assert cfg["model"].get("linear_transformer_depth", 0) == 0, "linear transformer layers are not vendored"
    inference = dict(
        chunk_size=cfg.get("inference", {}).get("chunk_size", cfg["audio"]["chunk_size"]),
        num_overlap=cfg["inference"]["num_overlap"],
    )
    training = cfg.get("training", {})
    stems = [training["target_instrument"]] if training.get("target_instrument") else training.get("instruments", [])
    return model, inference, stems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=None, help="local .ckpt (default: download KimberleyJSN/melbandroformer)")
    parser.add_argument("--config", default=None, help="MSST yaml config (default: KJ vocals config)")
    parser.add_argument("--out", default=os.path.join(MODELS_PATH, "checkpoints", DEFAULT_WEIGHTS))
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    if args.config:
        model_cfg, inference_cfg, stems = config_from_msst_yaml(args.config)
        source = args.ckpt
        license_ = ""
    else:
        model_cfg, inference_cfg, stems = KJ_VOCALS["model"], KJ_VOCALS["inference"], KJ_VOCALS["stems"]
        source = f"{KJ_VOCALS['repo_id']}/{KJ_VOCALS['filename']}"
        license_ = KJ_VOCALS["license"]

    ckpt = args.ckpt
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(KJ_VOCALS["repo_id"], KJ_VOCALS["filename"])

    print(f"loading {ckpt}")
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    # validate against the vendored module before writing anything
    model = MelBandRoformer(**model_cfg)
    model.load_state_dict(state_dict, strict=True)

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    # shared rotary freqs alias one storage in the ckpt; safetensors needs distinct tensors
    tensors = {k: v.detach().to(dtype).clone().contiguous() for k, v in state_dict.items()}

    metadata = dict(
        klass="MelBandRoformer",
        config=json.dumps(model_cfg),
        inference=json.dumps(inference_cfg),
        stems=json.dumps(stems),
        source=str(source),
        license=license_,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)
    n_params = sum(v.numel() for v in tensors.values())
    print(f"wrote {args.out} ({n_params / 1e6:.1f}M params, {args.dtype}, {os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
