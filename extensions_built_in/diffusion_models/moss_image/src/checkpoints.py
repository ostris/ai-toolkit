"""Where MOSS-Image weights come from.

Weights live in the ComfyUI layout under MODELS_PATH, named like the
Comfy-Org/MOSS-Image repack (DiT: the vendor's keys with q/k/v fused).

- ``OpenMOSS-Team/MOSS-Image-SFT`` / ``-Base`` (vendor, the UI defaults): a
  repack DiT file already on disk wins (best precision for the requested
  qtype; int8 only for convrot). Otherwise Base downloads the repack's bf16
  (byte-identical to the vendor shards); SFT, which the repack has no bf16
  of, downloads the vendor's diffusers shards once and merges them into
  ``diffusion_models/moss_image_sft_bf16.safetensors`` in the Comfy layout.
  Other hub repos (fine-tunes) merge into a file named after the repo, never
  substituted.
- ``Comfy-Org/MOSS-Image``: the repack's SFT DiT (int8 convrot only; a local
  merged bf16 SFT also counts).
- ``org/repo/path/file.safetensors``: that file, downloaded to its comfy spot.

The vendor text_encoder/ is stock Qwen3.5-9B (A_log / gated norm stored in
bf16), so the text encoder is a comfy file: the repack's int8 convrot or
Comfy-Org/Qwen3.5's bf16, whichever is local (ranked for the requested qtype),
else the one matching the qtype is downloaded. The VAE is the shared FLUX.2
file: same encoder + BatchNorm stats as the vendor vae/ (which is the BFL
weights rounded to bf16), full decoder in place of the small one. The
tokenizer comes from the vendor repo.
"""

import json
import math
import os
import shutil
import struct
from typing import Callable, List, Optional

import torch
from safetensors import safe_open

from toolkit.models.v2.resolver import (
    comfy_precision_rank,
    resolve_comfy_candidates,
    resolve_comfy_file,
)
from toolkit.paths import MODELS_PATH

SFT_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
BASE_REPO = "OpenMOSS-Team/MOSS-Image-Base"
COMFY_REPO = "Comfy-Org/MOSS-Image"
QWEN35_9B_REPO = "Qwen/Qwen3.5-9B"

VENDOR_VAE_FILE = "vae/diffusion_pytorch_model.safetensors"
VENDOR_TRANSFORMER_INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
COMFY_VAE_FILE = "vae/flux2-vae.safetensors"

# variant -> (files in the repack, extra local-only candidates). The merged
# vendor SFT is the bf16 SFT the repack does not ship.
COMFY_TRANSFORMER_FILES = {
    "sft": (
        ["diffusion_models/moss_image_sft_int8_convrot_attn.safetensors"],
        ["diffusion_models/moss_image_sft_bf16.safetensors"],
    ),
    "base": (
        [
            "diffusion_models/moss_image_base_int8_convrot_attn.safetensors",
            "diffusion_models/moss_image_base_bf16.safetensors",
        ],
        [],
    ),
}
REPO_VARIANT = {SFT_REPO: "sft", BASE_REPO: "base", COMFY_REPO: "sft"}
# vendor repo -> the repack's file holding its weights byte-identical (same
# name and layout the shard merge writes): downloaded instead of merging
VENDOR_MIRROR = {BASE_REPO: "diffusion_models/moss_image_base_bf16.safetensors"}

# (repo, file) candidates for the Qwen3.5-9B text encoder
COMFY_TEXT_ENCODER_FILES = [
    (COMFY_REPO, "text_encoders/qwen3.5_9b_int8_convrot.safetensors"),
    ("Comfy-Org/Qwen3.5", "text_encoders/qwen3.5_9b_bf16.safetensors"),
]


def comfy_transformer_rel(repo_id: str) -> str:
    name = repo_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
    return f"diffusion_models/{name}_bf16.safetensors"


def _staging_dir(category: str) -> str:
    # same filesystem as the target so the final rename is a move; removed
    # afterwards together with the hub's .cache bookkeeping
    return os.path.join(MODELS_PATH, category, ".moss_image_download")


def resolve_hub_file(
    ref: str,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """A local path, or 'org/repo/rel/file.safetensors' found at / downloaded
    to its comfy spot under MODELS_PATH."""
    if os.path.isfile(ref):
        return ref
    parts = ref.split("/")
    if len(parts) < 3:
        raise FileNotFoundError(
            f"{ref} is not a local file or an 'org/repo/path/file.safetensors' reference"
        )
    return resolve_comfy_candidates(
        ["/".join(parts[2:])],
        repo_id="/".join(parts[:2]),
        hf_token=hf_token,
        status_fn=status_fn,
    )


def comfy_transformer_file(
    repo_id: str,
    qtype: Optional[str] = None,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    variant = REPO_VARIANT.get(repo_id)
    if variant is not None:
        repo_files, local_extra = COMFY_TRANSFORMER_FILES[variant]
        candidates = repo_files + local_extra
        if repo_id != COMFY_REPO and not (qtype or "").lower().startswith("convrot"):
            # int8 only stands in for the vendor bf16 when it attaches as-is
            candidates = [c for c in candidates if "int8" not in c]
        found = resolve_comfy_candidates(
            candidates, repo_id=COMFY_REPO, local_only=True, qtype=qtype
        )
        if found is not None:
            return found
        if repo_id == COMFY_REPO:
            return resolve_comfy_candidates(
                repo_files,
                repo_id=COMFY_REPO,
                hf_token=hf_token,
                status_fn=status_fn,
                qtype=qtype,
            )
        if repo_id in VENDOR_MIRROR:
            return resolve_comfy_candidates(
                [VENDOR_MIRROR[repo_id]],
                repo_id=COMFY_REPO,
                hf_token=hf_token,
                status_fn=status_fn,
            )

    rel_path = comfy_transformer_rel(repo_id)
    found = resolve_comfy_file(rel_path, repo_id, local_only=True)
    if found is not None:
        return found
    from huggingface_hub import hf_hub_download

    target = os.path.join(MODELS_PATH, rel_path)
    staging = _staging_dir("diffusion_models")
    if status_fn is not None:
        status_fn(f"Downloading the {repo_id} transformer to {target}")
    index_path = hf_hub_download(
        repo_id, VENDOR_TRANSFORMER_INDEX, token=hf_token, local_dir=staging
    )
    with open(index_path) as f:
        shard_names = sorted(set(json.load(f)["weight_map"].values()))
    shards = [
        hf_hub_download(
            repo_id, f"transformer/{name}", token=hf_token, local_dir=staging
        )
        for name in shard_names
    ]
    merge_safetensors(shards, target, fuse_qkv=True)
    shutil.rmtree(staging, ignore_errors=True)
    return target


def comfy_text_encoder_file(
    qtype: Optional[str] = None,
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    rels = [rel for _, rel in COMFY_TEXT_ENCODER_FILES]
    found = resolve_comfy_candidates(rels, repo_id=COMFY_REPO, local_only=True, qtype=qtype)
    if found is not None:
        return found
    repo_id, rel = min(
        COMFY_TEXT_ENCODER_FILES, key=lambda c: comfy_precision_rank(c[1], qtype=qtype)
    )
    return resolve_comfy_candidates(
        [rel], repo_id=repo_id, hf_token=hf_token, status_fn=status_fn
    )


def comfy_vae_file(
    hf_token: Optional[str] = None,
    status_fn: Optional[Callable[[str], None]] = None,
) -> str:
    return resolve_comfy_candidates(
        [COMFY_VAE_FILE], repo_id=COMFY_REPO, hf_token=hf_token, status_fn=status_fn
    )


_CAST_TO_BF16 = ("F32", "F16", "F64")


def merge_safetensors(
    shards: List[str],
    out_path: str,
    fuse_qkv: bool = False,
    chunk_bytes: int = 64 << 20,
):
    """Concatenate safetensors shards into one bf16 file. bf16 (and non-float)
    tensors are copied as raw bytes; other float tensors are cast to bf16.
    ``fuse_qkv`` writes each ``*.proj.{q,k,v}.weight`` triple as one
    ``*.proj.qkv.weight`` (row concat of contiguous tensors = byte concat).
    Written to ``.part`` and renamed, so ``out_path`` may be one of the shards."""
    tensors = {}
    for path in shards:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        header.pop("__metadata__", None)
        for name, meta in header.items():
            start, end = meta["data_offsets"]
            tensors[name] = (meta["dtype"], meta["shape"], path, 8 + header_len + start, end - start)

    # out name -> (dtype, shape, [(src name, path, start, size)])
    entries = {}
    for name, (dtype, shape, path, start, size) in tensors.items():
        if fuse_qkv and name.endswith((".proj.k.weight", ".proj.v.weight")):
            continue
        if fuse_qkv and name.endswith(".proj.q.weight"):
            base = name[: -len("q.weight")]
            parts = [base + f"{p}.weight" for p in ("q", "k", "v")]
            srcs = [tensors[p] for p in parts]
            if any(s[0] != dtype or s[1][1:] != shape[1:] for s in srcs):
                raise ValueError(f"{base}q/k/v parts do not fuse")
            fused_shape = [sum(s[1][0] for s in srcs)] + list(shape[1:])
            entries[base + "qkv.weight"] = (
                dtype,
                fused_shape,
                [(p, s[2], s[3], s[4]) for p, s in zip(parts, srcs)],
            )
            continue
        entries[name] = (dtype, shape, [(name, path, start, size)])

    out_header, offset = {}, 0
    for name in sorted(entries):
        dtype, shape, parts = entries[name]
        size = sum(p[3] for p in parts)
        if dtype in _CAST_TO_BF16:
            dtype, size = "BF16", 2 * math.prod(shape)
        out_header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    out_header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    part_path = out_path + ".part"
    handles = {}
    try:
        with open(part_path, "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for name in sorted(entries):
                dtype, _, parts = entries[name]
                for src_name, path, start, size in parts:
                    if dtype in _CAST_TO_BF16:
                        with safe_open(path, "pt") as f:
                            tensor = f.get_tensor(src_name).to(torch.bfloat16)
                        out.write(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                        continue
                    src = handles.get(path)
                    if src is None:
                        src = handles[path] = open(path, "rb")
                    src.seek(start)
                    remaining = size
                    while remaining:
                        chunk = src.read(min(chunk_bytes, remaining))
                        out.write(chunk)
                        remaining -= len(chunk)
    finally:
        for src in handles.values():
            src.close()
    os.replace(part_path, out_path)
