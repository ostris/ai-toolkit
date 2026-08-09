"""Run AI Toolkit training jobs on Beam Cloud.

Usage:
    BEAM_GPU=RTX4090 python run_beam.py config/my_beam_job.yaml
    BEAM_GPU=H100 BEAM_POOL=ai-toolkit-h100 python run_beam.py config/job.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from beam import Image, Volume, function


APP_DIR = "/mnt/code"
OUTPUT_DIR = "/volumes/ai-toolkit-output"
CACHE_DIR = "/volumes/ai-toolkit-cache"

SERVERLESS_GPUS = ("A10G", "RTX4090", "RTX5090")
ON_DEMAND_GPUS = (
    "RTX4090",
    "RTX5090",
    "RTXPro6000",
    "A6000",
    "L40S",
    "A100-80",
    "H100",
    "H200",
    "B200",
)
SUPPORTED_GPUS = tuple(dict.fromkeys((*SERVERLESS_GPUS, *ON_DEMAND_GPUS)))


@dataclass(frozen=True)
class BeamGpuConfig:
    gpu: str
    pool: Optional[str]

    @property
    def mode(self) -> str:
        return "on-demand" if self.pool else "serverless"


def resolve_beam_gpu_config(
    environ: Optional[Mapping[str, str]] = None,
) -> BeamGpuConfig:
    """Validate the explicitly selected Beam GPU and optional reserved pool."""

    values = os.environ if environ is None else environ
    gpu = values.get("BEAM_GPU", "").strip()
    pool = values.get("BEAM_POOL", "").strip() or None

    if not gpu:
        raise ValueError(
            "BEAM_GPU is required. "
            f"Serverless: {', '.join(SERVERLESS_GPUS)}. "
            f"On-demand with BEAM_POOL: {', '.join(ON_DEMAND_GPUS)}."
        )

    if gpu not in SUPPORTED_GPUS:
        raise ValueError(
            f"Unsupported BEAM_GPU={gpu!r}. "
            f"Supported GPUs: {', '.join(SUPPORTED_GPUS)}."
        )

    if pool is None and gpu not in SERVERLESS_GPUS:
        raise ValueError(
            f"BEAM_GPU={gpu} is on-demand only. Reserve it with "
            f"'beam machine reserve --gpu {gpu} --ttl 3h --name <pool>', "
            "then set BEAM_POOL=<pool>."
        )

    if pool is not None and gpu not in ON_DEMAND_GPUS:
        raise ValueError(
            f"BEAM_GPU={gpu} is not supported in an on-demand pool. "
            f"On-demand GPUs: {', '.join(ON_DEMAND_GPUS)}."
        )

    return BeamGpuConfig(gpu=gpu, pool=pool)


beam_gpu_config = resolve_beam_gpu_config()

output_volume = Volume(name="ai-toolkit-output", mount_path=OUTPUT_DIR)
cache_volume = Volume(name="ai-toolkit-cache", mount_path=CACHE_DIR)

# Beam synchronizes the repository to /mnt/code at invocation time. The image
# only contains the system and Python dependencies needed to run AI Toolkit.
image = Image.from_dockerfile("docker/Dockerfile.beam", context_dir=".")


def _print_end_message(jobs_completed: int, jobs_failed: int) -> None:
    print("")
    print("========================================")
    print("Result:")
    print(f" - {jobs_completed} completed job{'' if jobs_completed == 1 else 's'}")
    if jobs_failed:
        print(f" - {jobs_failed} failure{'' if jobs_failed == 1 else 's'}")
    print("========================================")


@function(
    name="ai-toolkit-training",
    image=image,
    gpu=beam_gpu_config.gpu,
    pool=beam_gpu_config.pool,
    cpu=4,
    memory="32Gi",
    timeout=7200,
    retries=0,
    volumes=[output_volume, cache_volume],
    secrets=["HF_TOKEN"],
    env={
        "BEAM_GPU": beam_gpu_config.gpu,
        "BEAM_POOL": beam_gpu_config.pool or "",
        "DISABLE_TELEMETRY": "YES",
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_HOME": f"{CACHE_DIR}/torch",
    },
)
def train(
    config_files: list[str],
    recover: bool = False,
    name: Optional[str] = None,
) -> dict[str, int]:
    """Run one or more AI Toolkit config files in a Beam GPU container."""

    import sys

    os.chdir(APP_DIR)
    sys.path.insert(0, APP_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Beam training container started without an available CUDA GPU")

    requested_gpu = os.environ.get("BEAM_GPU", "unknown")
    requested_pool = os.environ.get("BEAM_POOL", "")
    print(f"Beam execution mode: {'on-demand' if requested_pool else 'serverless'}")
    print(f"Requested Beam GPU: {requested_gpu}")
    if requested_pool:
        print(f"Requested Beam pool: {requested_pool}")
    print(f"Actual CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

    if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
        torch.autograd.set_detect_anomaly(True)

    # Keep this import remote-only: the local Beam client environment does not
    # need AI Toolkit's complete training dependency set.
    from toolkit.job import get_job

    jobs_completed = 0
    jobs_failed = 0

    print(f"Running {len(config_files)} job{'' if len(config_files) == 1 else 's'}")
    print(f"Training outputs will be saved to: {OUTPUT_DIR}")
    print(f"Model caches will be saved to: {CACHE_DIR}")

    for config_file in config_files:
        job = None
        training_started_at = None
        try:
            print(f"Loading config: {config_file}")
            job = get_job(config_file, name)
            job.config["process"][0]["training_folder"] = OUTPUT_DIR
            training_started_at = time.monotonic()
            job.run()
            jobs_completed += 1
        except Exception as exc:
            print(f"Error running job: {exc}")
            jobs_failed += 1
            if not recover:
                _print_end_message(jobs_completed, jobs_failed)
                raise
        finally:
            if training_started_at is not None:
                elapsed_seconds = time.monotonic() - training_started_at
                print(f"Training elapsed time ({config_file}): {elapsed_seconds:.1f}s")
            if job is not None:
                try:
                    job.cleanup()
                except Exception as cleanup_exc:
                    print(f"Warning: job cleanup failed: {cleanup_exc}")

    _print_end_message(jobs_completed, jobs_failed)
    return {"completed": jobs_completed, "failed": jobs_failed}


def _to_remote_path(config_file: str, project_root: Path) -> str:
    """Translate an absolute in-repository config path to Beam's sync root."""

    path = Path(config_file).expanduser()
    if not path.is_absolute():
        return config_file

    try:
        relative_path = path.resolve().relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Config file must be inside the repository so Beam can sync it: {config_file}"
        ) from exc

    return f"{APP_DIR}/{relative_path.as_posix()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Toolkit training on Beam Cloud")
    parser.add_argument(
        "config_files",
        nargs="+",
        help="Config file paths. Multiple files run sequentially in one container.",
    )
    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        help="Continue with the next config if a job fails.",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Replace the [name] tag in the config file.",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parent
    remote_config_files = [
        _to_remote_path(config_file, repository_root) for config_file in args.config_files
    ]
    train.remote(remote_config_files, recover=args.recover, name=args.name)
