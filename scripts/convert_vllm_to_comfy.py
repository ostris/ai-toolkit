"""Convert a huggingface LLM / VLM checkpoint into a single ComfyUI-format
safetensors file, quantized with convrot8 (ComfyUI's ``int8_tensorwise`` +
``convrot`` layout, which comfy_kitchen runs natively and is bit-identical to
the toolkit's convrot8 storage).

ComfyUI ships its text-generation models (the CLIPLoader / TextGenerate path)
as one safetensors file holding the huggingface state dict, with quantized
layers marked two equivalent ways, both of which this script writes:
  - a ``<layer>.comfy_quant`` uint8 JSON tensor baked into the state dict
    (what comfy's _load_quantized_module and the toolkit's
    comfy_quant_import read)
  - a ``_quantization_metadata`` JSON blob in the safetensors file metadata
    (comfy's documented checkpoint format; converted to the markers at load)

Quantized layer storage (per-output-row symmetric int8 on regular-Hadamard
rotated weights, rotation block = min(256, largest power-of-4 divisor of
in_features)):
    <layer>.weight        int8  [out, in]           (or [E, out, in] for MoE banks)
    <layer>.weight_scale  fp32  [out]               (or [E, out])
    <layer>.comfy_quant   {"format": "int8_tensorwise", "convrot": true,
                           "convrot_groupsize": G}  (+ "num_experts": E for banks)

Per-expert MoE linears (``...experts.<i>.gate_proj/up_proj/down_proj.weight``)
are fused into the 3D banks comfy's ops.MoEExperts expects:
``...experts.gate_up_proj.weight`` ([E, 2I, H], gate rows first, matching how
transformers fuses the same checkpoints) and ``...experts.down_proj.weight``
([E, H, I]).

Note: the toolkit's own import_comfy_quantized_layers only understands 2D
linear/embedding markers, not the 3D expert banks — the banks are for comfy.

Usage:
    python scripts/convert_vllm_to_comfy.py Qwen/Qwen3-Omni-30B-A3B-Instruct \
        /path/out/qwen3_omni_30b_thinker_convrot8.safetensors
    python scripts/convert_vllm_to_comfy.py <repo_or_local_dir> out.safetensors --no-quant
"""

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.util.convrot_quant import (
    largest_pow4_divisor,
    quantize_int8_rows,
    rotate,
)

# convrot8 eligibility (mirrors ConvRotInt8Quantizer.can_quantize)
MAX_ROT = 256
MIN_ROT = 16

EXPERT_KEY_RE = re.compile(
    r"^(?P<bank>.*\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


def comfy_quant_marker(conf: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


class ArchHandler:
    """Generic model: keep every key, no expert fusion.

    Subclasses override to drop components, rename keys, and add
    architecture-specific quantization excludes.
    """

    # substrings/suffixes never quantized: output head and embeddings are
    # quality-critical, MoE router logits are precision-sensitive
    exclude_contains = ("embed_tokens", "pos_embed", "lm_head")
    exclude_suffixes = (".mlp.gate.weight",)

    def __init__(self, config: dict):
        self.config = config

    def map_key(self, key: str):
        """Return the output key, or None to drop the tensor."""
        return key

    def num_experts(self):
        for path in (
            ("num_experts",),
            ("text_config", "num_experts"),
            ("thinker_config", "text_config", "num_experts"),
        ):
            node = self.config
            for p in path:
                if not isinstance(node, dict) or p not in node:
                    node = None
                    break
                node = node[p]
            if isinstance(node, int):
                return node
        return None

    def is_excluded(self, key: str) -> bool:
        if any(s in key for s in self.exclude_contains):
            return True
        return any(key.endswith(s) for s in self.exclude_suffixes)


class Qwen3OmniHandler(ArchHandler):
    """Qwen3-Omni: keep only the thinker (the VLM — talker and code2wav are
    speech synthesis, dead weight for text generation) and strip its prefix
    so keys start at model./visual./audio_tower./lm_head. like comfy's other
    qwen checkpoints."""

    def map_key(self, key: str):
        if not key.startswith("thinker."):
            return None
        return key[len("thinker.") :]


ARCH_HANDLERS = {
    "Qwen3OmniMoeForConditionalGeneration": Qwen3OmniHandler,
}


def resolve_model_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        name_or_path,
        allow_patterns=["*.safetensors", "model.safetensors.index.json", "config.json"],
    )


def shard_files(model_dir: str):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        return sorted(set(index["weight_map"].values()))
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return ["model.safetensors"]
    raise FileNotFoundError(f"No model.safetensors(.index.json) in {model_dir}")


def rot_for(in_features: int) -> int:
    return min(MAX_ROT, largest_pow4_divisor(in_features))


def can_quantize_2d(out_features: int, in_features: int) -> bool:
    return (
        in_features % 16 == 0
        and out_features % 8 == 0
        and rot_for(in_features) >= MIN_ROT
    )


@torch.no_grad()
def quantize_rows(weight: torch.Tensor, rot: int, device):
    """convrot8: per-row symmetric int8 on the rotated weight. Accepts
    [out, in] or [E, out, in]; scales come back fp32 [out, 1] / [E, out, 1]
    (the trailing 1 is comfy_kitchen's per-channel scale convention — its
    int8 dequant broadcasts scale directly against [out, in])."""
    shape = weight.shape
    w = weight.to(device=device, dtype=torch.float32)
    q, scales = quantize_int8_rows(rotate(w, rot).reshape(-1, shape[-1]))
    return q.reshape(shape).cpu(), scales.reshape(shape[:-1] + (1,)).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="hf repo id or local model directory")
    parser.add_argument("output", help="output .safetensors path")
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="just repack to a single file, no quantization",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="dtype for non-quantized float tensors",
    )
    parser.add_argument(
        "--extra-exclude",
        default="",
        help="comma-separated substrings of keys to keep unquantized",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run quantization math on",
    )
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    extra_exclude = tuple(s for s in args.extra_exclude.split(",") if s.strip())

    model_dir = resolve_model_dir(args.model)
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    handler_cls = ArchHandler
    for arch in config.get("architectures", []):
        if arch in ARCH_HANDLERS:
            handler_cls = ARCH_HANDLERS[arch]
            break
    handler = handler_cls(config)
    print(f"Architecture handler: {handler_cls.__name__}")

    num_experts = handler.num_experts()
    quantize = not args.no_quant

    out_sd = {}
    quant_layers = {}
    # per-(bank, proj) accumulation of expert weights until all E arrive
    pending_experts = {}

    def add_quantized(layer: str, weight: torch.Tensor, conf_extra=None):
        rot = rot_for(weight.shape[-1])
        q, scales = quantize_rows(weight, rot, args.device)
        conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": rot}
        if conf_extra:
            conf.update(conf_extra)
        out_sd[f"{layer}.weight"] = q
        out_sd[f"{layer}.weight_scale"] = scales
        out_sd[f"{layer}.comfy_quant"] = comfy_quant_marker(conf)
        quant_layers[layer] = conf

    def flush_bank(bank: str):
        """Fuse a completed expert group into comfy MoEExperts banks and
        quantize them. gate/up fuse into one [E, 2I, H] bank (gate rows
        first), down stays its own [E, H, I] bank."""
        gate = pending_experts.pop((bank, "gate_proj"), None)
        up = pending_experts.pop((bank, "up_proj"), None)
        down = pending_experts.pop((bank, "down_proj"), None)
        if (
            gate is None
            or up is None
            or down is None
            or None in gate
            or None in up
            or None in down
        ):
            raise RuntimeError(f"Incomplete expert group for {bank}")
        e = len(gate)
        gate_up = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate, up)])
        down = torch.stack(down)
        for name, bankw in (("gate_up_proj", gate_up), ("down_proj", down)):
            layer = f"{bank}.{name}"
            if quantize and can_quantize_2d(bankw.shape[1], bankw.shape[2]):
                add_quantized(layer, bankw, conf_extra={"num_experts": e})
            else:
                out_sd[f"{layer}.weight"] = bankw.to(dtype)

    def bank_complete(bank: str) -> bool:
        counts = [
            sum(w is not None for w in pending_experts.get((bank, p), []))
            for p in ("gate_proj", "up_proj", "down_proj")
        ]
        return num_experts is not None and counts == [num_experts] * 3

    shards = shard_files(model_dir)
    for shard in tqdm(shards, desc="Shards"):
        with safe_open(
            os.path.join(model_dir, shard), framework="pt", device="cpu"
        ) as f:
            for key in f.keys():
                out_key = handler.map_key(key)
                if out_key is None:
                    continue
                tensor = f.get_tensor(key)

                m = EXPERT_KEY_RE.match(out_key)
                if m is not None and quantize:
                    bank, idx, proj = (
                        m.group("bank"),
                        int(m.group("idx")),
                        m.group("proj"),
                    )
                    slot = pending_experts.setdefault(
                        (bank, proj), [None] * (num_experts or idx + 1)
                    )
                    if idx >= len(slot):
                        slot.extend([None] * (idx + 1 - len(slot)))
                    slot[idx] = tensor
                    if bank_complete(bank):
                        flush_bank(bank)
                    continue

                excluded = handler.is_excluded(out_key) or any(
                    s in out_key for s in extra_exclude
                )
                if (
                    quantize
                    and not excluded
                    and out_key.endswith(".weight")
                    and tensor.ndim == 2
                    and can_quantize_2d(*tensor.shape)
                ):
                    add_quantized(out_key[: -len(".weight")], tensor)
                elif tensor.is_floating_point():
                    out_sd[out_key] = tensor.to(dtype)
                else:
                    out_sd[out_key] = tensor

    # experts that never completed during streaming (unknown count or
    # shard-straddling groups): fuse whatever is fully populated now
    for bank in sorted({b for (b, _p) in pending_experts}):
        flush_bank(bank)

    metadata = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": quant_layers}
        )
    }
    total_bytes = sum(t.numel() * t.element_size() for t in out_sd.values())
    print(
        f"Saving {len(out_sd)} tensors ({total_bytes / 1e9:.2f} GB, "
        f"{len(quant_layers)} quantized layers) to {args.output}"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(out_sd, args.output, metadata=metadata)
    print("Done")


if __name__ == "__main__":
    main()
