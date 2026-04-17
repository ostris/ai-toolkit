#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


GENERIC_CONFIG_PATTERNS = ["*.json", "*.txt", "*.model"]

SDXL_CONFIG_PATTERNS = [
    "model_index.json",
    "scheduler/*.json",
    "text_encoder/*.json",
    "text_encoder_2/*.json",
    "tokenizer/*.json",
    "tokenizer/*.txt",
    "tokenizer/*.model",
    "tokenizer_2/*.json",
    "tokenizer_2/*.txt",
    "tokenizer_2/*.model",
    "unet/*.json",
    "vae/*.json",
    "vae_1_0/*.json",
    "vae_decoder/*.json",
    "vae_encoder/*.json",
]


@dataclass(frozen=True)
class RepoSpec:
    repo_id: str
    allow_patterns: list[str]
    required_files: list[str]


REPO_SPECS: dict[str, RepoSpec] = {
    "stable-diffusion-v1-5/stable-diffusion-v1-5": RepoSpec(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        allow_patterns=GENERIC_CONFIG_PATTERNS,
        required_files=["model_index.json", "tokenizer/merges.txt"],
    ),
    "runwayml/stable-diffusion-v1-5": RepoSpec(
        repo_id="runwayml/stable-diffusion-v1-5",
        allow_patterns=GENERIC_CONFIG_PATTERNS,
        required_files=["model_index.json", "tokenizer/merges.txt"],
    ),
    "stabilityai/stable-diffusion-xl-base-1.0": RepoSpec(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        allow_patterns=SDXL_CONFIG_PATTERNS,
        required_files=[
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "tokenizer/merges.txt",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/vocab.json",
            "unet/config.json",
            "vae/config.json",
            "vae_1_0/config.json",
            "vae_decoder/config.json",
            "vae_encoder/config.json",
        ],
    ),
    "openai/clip-vit-large-patch14": RepoSpec(
        repo_id="openai/clip-vit-large-patch14",
        allow_patterns=GENERIC_CONFIG_PATTERNS,
        required_files=["config.json", "tokenizer_config.json", "vocab.json", "merges.txt"],
    ),
    "stabilityai/stable-diffusion-2-1": RepoSpec(
        repo_id="stabilityai/stable-diffusion-2-1",
        allow_patterns=GENERIC_CONFIG_PATTERNS,
        required_files=["model_index.json", "tokenizer/merges.txt"],
    ),
}


DEFAULT_REPOS = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "openai/clip-vit-large-patch14",
]


def validate_snapshot(repo_id: str, snapshot_dir: str, required_files: list[str]) -> None:
    missing = [
        relative_path
        for relative_path in required_files
        if not Path(snapshot_dir, relative_path).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"Snapshot for {repo_id} is missing required files:\n{formatted}")


def download_repo(spec: RepoSpec) -> None:
    print(f"Downloading config cache for {spec.repo_id}", flush=True)
    snapshot_dir = snapshot_download(
        repo_id=spec.repo_id,
        allow_patterns=spec.allow_patterns,
        token=os.environ.get("HF_TOKEN") or None,
    )
    validate_snapshot(spec.repo_id, snapshot_dir, spec.required_files)
    print(f"Validated config cache for {spec.repo_id} at {snapshot_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        action="append",
        choices=sorted(REPO_SPECS),
        help="Download only the specified repo. Can be provided multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_ids = args.repo or DEFAULT_REPOS
    for repo_id in repo_ids:
        download_repo(REPO_SPECS[repo_id])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise
