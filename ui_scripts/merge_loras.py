"""Merge a list of LoRAs into a single checkpoint."""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open


DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def log(message: str) -> None:
    print(message, flush=True)


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
        "--save_dtype",
        default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Dtype of the saved tensors (merging is always done in float32).",
    )
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
    save_dtype = DTYPE_MAP[args.save_dtype]

    log(f"Merging {len(loras)} LoRA(s) on {device}, saving as {args.save_dtype}.")

    merged: dict[str, torch.Tensor] = {}

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
            scaled = tensor.to(torch.float32) * strength
            if key in merged:
                merged[key].add_(scaled)
            else:
                merged[key] = scaled
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
                orig_metadata = f.metadata()
                for meta_key in metadata_to_keep:
                    if meta_key in orig_metadata:
                        metadata[meta_key] = orig_metadata[meta_key]

    log(f"Casting to {args.save_dtype} and moving to CPU")
    final = {k: v.to(save_dtype).cpu().contiguous() for k, v in merged.items()}
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
                "save_dtype": args.save_dtype,
                "device": str(device),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
