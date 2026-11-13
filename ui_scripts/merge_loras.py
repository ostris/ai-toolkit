"""Merge a list of LoRAs into a single checkpoint."""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open


def log(message: str) -> None:
    print(message, flush=True)


def higher_precision(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    """Pick whichever of the two dtypes is more precise (fp16 beats bf16)."""
    if a == b:
        return a
    if a.is_floating_point and b.is_floating_point:
        return a if torch.finfo(a).eps <= torch.finfo(b).eps else b
    return torch.promote_types(a, b)


def pad_to_shape(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Zero-pad a tensor up to the given shape (e.g. to match a larger LoRA rank)."""
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    out = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    out[tuple(slice(0, s) for s in tensor.shape)] = tensor
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge a list of LoRAs into a single checkpoint."
    )
    parser.add_argument(
        "--loras",
        required=True,
        help='JSON list of {"path": "...", "strength": 1.0} entries.',
    )
    parser.add_argument("--output", required=True, help="Output .safetensors path.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to merge on (cpu, cuda, cuda:1, mps). Output is always saved from CPU.",
    )
    args = parser.parse_args()

    try:
        loras = json.loads(args.loras)
    except json.JSONDecodeError as e:
        print(f"Failed to parse --loras JSON: {e}", file=sys.stderr, flush=True)
        return 2

    if not isinstance(loras, list) or len(loras) == 0:
        print("--loras must be a non-empty JSON list.", file=sys.stderr, flush=True)
        return 2

    device = torch.device(args.device)

    log(f"Merging {len(loras)} LoRA(s) on {device}.")

    merged: dict[str, torch.Tensor] = {}
    # dtype each key is saved as: the most precise dtype seen for that key
    save_dtypes: dict[str, torch.dtype] = {}

    metadata = {}

    for i, entry in enumerate(loras):
        if not isinstance(entry, dict) or "path" not in entry:
            print(
                f"LoRA entry {i} must be an object with a 'path' field.",
                file=sys.stderr,
                flush=True,
            )
            return 2

        path = entry["path"]
        strength = float(entry.get("strength", 1.0))

        if not os.path.isfile(path):
            print(f"LoRA file not found: {path}", file=sys.stderr, flush=True)
            return 2

        log(f"[{i + 1}/{len(loras)}] Loading {path} (strength={strength})")
        state_dict = load_file(path, device=str(device))

        for key, tensor in state_dict.items():
            # accumulate in at least float32, cast back to the source dtype on save
            work_dtype = higher_precision(tensor.dtype, torch.float32)
            scaled = tensor.to(work_dtype) * strength
            if key in merged:
                existing = merged[key]
                save_dtypes[key] = higher_precision(save_dtypes[key], tensor.dtype)
                if existing.dtype != scaled.dtype:
                    work_dtype = higher_precision(existing.dtype, scaled.dtype)
                    existing = existing.to(work_dtype)
                    scaled = scaled.to(work_dtype)
                    merged[key] = existing
                if existing.shape != scaled.shape:
                    if existing.ndim != scaled.ndim:
                        print(
                            f"Cannot merge key '{key}': incompatible shapes "
                            f"{tuple(existing.shape)} vs {tuple(scaled.shape)}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 2
                    # Different LoRA ranks: zero-pad both to the max size on
                    # each mismatched dim so they can merge as equal sizes.
                    target = tuple(
                        max(a, b) for a, b in zip(existing.shape, scaled.shape)
                    )
                    existing = pad_to_shape(existing, target)
                    scaled = pad_to_shape(scaled, target)
                    merged[key] = existing
                merged[key].add_(scaled)
            else:
                merged[key] = scaled
                save_dtypes[key] = tensor.dtype
        del state_dict

        if i == 0:
            # For the first LoRA, also copy over all non-tensor metadata (e.g. base model info)
            with safe_open(path, framework="pt") as f:
                metadata_to_keep = [
                    "version",
                    "format",
                    "ss_base_model_version",
                    "software",
                ]
                orig_metadata = f.metadata() or {}
                for meta_key in metadata_to_keep:
                    if meta_key in orig_metadata:
                        metadata[meta_key] = orig_metadata[meta_key]

    log("Moving to CPU")
    final = {k: v.to(save_dtypes[k]).cpu().contiguous() for k, v in merged.items()}
    merged.clear()

    log(f"Saving merged checkpoint to {args.output}")
    save_file(final, args.output, metadata=metadata)

    print(
        json.dumps(
            {
                "ok": True,
                "output": args.output,
                "num_loras": len(loras),
                "num_keys": len(final),
                "device": str(device),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
