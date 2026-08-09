"""Run AI Toolkit training jobs on Beam Cloud.

Usage:
    BEAM_GPU=RTX4090 python run_beam.py config/my_beam_job.yaml
    BEAM_GPU=H100 BEAM_POOL=ai-toolkit-h100 python run_beam.py config/job.yaml
    BEAM_GPU=RTX4090 uv run beam deploy run_beam.py:ai_toolkit_gui
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from beam import Image, Pod, Volume, function
from beta9.env import is_remote


APP_DIR = "/mnt/code"
REPOSITORY_ROOT = Path(__file__).resolve().parent
BEAM_DOCKERFILE = REPOSITORY_ROOT / "docker" / "Dockerfile.beam"
OUTPUT_DIR = "/volumes/ai-toolkit-output"
CACHE_DIR = "/volumes/ai-toolkit-cache"
UI_STATE_DIR = "/volumes/ai-toolkit-ui-state"
UI_DB_PATH = f"{UI_STATE_DIR}/aitk_db.db"
UI_DATASETS_DIR = f"{UI_STATE_DIR}/datasets"
UI_DATA_DIR = f"{UI_STATE_DIR}/data"
UI_MODELS_DIR = f"{CACHE_DIR}/models"

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
DEFAULT_KEEP_WARM_SECONDS = 1800


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


def resolve_keep_warm_seconds(
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Read the Pod inactivity timeout, allowing ``-1`` for no timeout."""

    values = os.environ if environ is None else environ
    raw_value = values.get("BEAM_KEEP_WARM_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_KEEP_WARM_SECONDS

    try:
        keep_warm_seconds = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "BEAM_KEEP_WARM_SECONDS must be an integer number of seconds "
            f"(0 or greater, or -1 for no timeout), got {raw_value!r}."
        ) from exc

    if keep_warm_seconds < -1:
        raise ValueError(
            "BEAM_KEEP_WARM_SECONDS must be 0 or greater, or -1 for no timeout. "
            f"Got {keep_warm_seconds}."
        )

    return keep_warm_seconds


beam_gpu_config = resolve_beam_gpu_config()
beam_keep_warm_seconds = resolve_keep_warm_seconds()

output_volume = Volume(name="ai-toolkit-output", mount_path=OUTPUT_DIR)
cache_volume = Volume(name="ai-toolkit-cache", mount_path=CACHE_DIR)
ui_state_volume = Volume(name="ai-toolkit-ui-state", mount_path=UI_STATE_DIR)


def _beam_image_context() -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Create a stable image-only context without configs, datasets, or caches."""

    temp_dir = tempfile.TemporaryDirectory(prefix="ai-toolkit-beam-image-")
    context = Path(temp_dir.name)
    shutil.copy2(BEAM_DOCKERFILE, context / "Dockerfile")
    for filename in ("requirements.txt", "requirements_base.txt"):
        shutil.copy2(REPOSITORY_ROOT / filename, context / filename)

    def ignore_ui(_directory: str, names: list[str]) -> set[str]:
        ignored = {".next", ".git", "node_modules", ".env", ".env.local"}
        return {name for name in names if name in ignored or name.startswith(".env.")}

    shutil.copytree(
        REPOSITORY_ROOT / "ui",
        context / "ui",
        ignore=ignore_ui,
    )
    return context, temp_dir


def _build_beam_image() -> Image:
    """Build/cache the CUDA image from only dependency and UI inputs.

    Training configs, datasets, outputs, and Python source are synced as runtime
    files by Beam and therefore do not invalidate this image's build cache.
    """

    if is_remote():
        # The remote worker receives the already-built image metadata. Avoid
        # touching local paths when this module is imported inside the worker.
        return Image.from_dockerfile("docker/Dockerfile.beam", context_dir=".")

    context, temp_dir = _beam_image_context()
    try:
        image = Image.from_dockerfile(str(context / "Dockerfile"), context_dir=str(context))
    finally:
        # from_dockerfile has synchronously uploaded and hashed the context.
        temp_dir.cleanup()

    # NATTEN and Flash Attention are CUDA extensions compiled in the Dockerfile.
    # Use the explicitly selected GPU for the first build; Beam reuses the
    # resulting image while this stable context remains unchanged.
    return image.build_with_gpu(gpu=beam_gpu_config.gpu)


# Beam synchronizes the full repository separately at invocation time. Only
# dependency manifests and UI source participate in the image cache key.
image = _build_beam_image()


def _beam_ui_entrypoint() -> list[str]:
    """Build the UI startup command without exposing any secret values."""

    startup = (
        "set -euo pipefail; "
        ': "${AI_TOOLKIT_AUTH:?Beam secret AI_TOOLKIT_AUTH is required}"; '
        'mkdir -p "$TRAINING_FOLDER" "$DATASETS_FOLDER" "$DATA_ROOT" '
        '"$MODELS_PATH" "$UI_STATE_DIR"; '
        "cd /mnt/code/ui; "
        "npx prisma db push --skip-generate; "
        "exec npm run start"
    )
    return ["bash", "-lc", startup]


# The GUI is a GPU Pod. It runs the existing Next.js UI and its cron worker in
# the same container as run.py, matching the local UI behavior. The inactivity
# timeout is configurable: use a finite value for cost control, or -1 when a
# long training run must survive a closed browser.
ai_toolkit_gui = Pod(
    name="ai-toolkit-ui",
    image=image,
    gpu=beam_gpu_config.gpu,
    pool=beam_gpu_config.pool,
    cpu=4,
    memory="32Gi",
    ports=[8675],
    keep_warm_seconds=beam_keep_warm_seconds,
    authorized=False,
    entrypoint=_beam_ui_entrypoint(),
    volumes=[output_volume, cache_volume, ui_state_volume],
    secrets=["HF_TOKEN", "AI_TOOLKIT_AUTH"],
    env={
        "BEAM_GPU": beam_gpu_config.gpu,
        "BEAM_POOL": beam_gpu_config.pool or "",
        "BEAM_KEEP_WARM_SECONDS": str(beam_keep_warm_seconds),
        "DATABASE_URL": f"file:{UI_DB_PATH}",
        "DATA_ROOT": UI_DATA_DIR,
        "DATASETS_FOLDER": UI_DATASETS_DIR,
        "DISABLE_TELEMETRY": "YES",
        "HF_HOME": f"{CACHE_DIR}/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MODELS_PATH": UI_MODELS_DIR,
        "NODE_ENV": "production",
        "TRAINING_FOLDER": OUTPUT_DIR,
        "TORCH_HOME": f"{CACHE_DIR}/torch",
        "UI_STATE_DIR": UI_STATE_DIR,
    },
)


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
