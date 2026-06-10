"""Shared remote contract for the remote-GPU training pipeline.

Every cross-module string lives here: remote filesystem layout, sentinel
filenames, tmux session naming, run-name validation, disk-sizing constants,
sample/checkpoint filename grammars, and log matchers. preflight, transport,
launch, monitor, and lifecycle all import from this module so the contract
cannot drift between units.

Filename grammars mirror jobs/process/BaseSDTrainProcess.py:
  samples:     {ms_epoch}__{step:09d}_{count}.{ext}   (double underscore)
  checkpoints: {job_name}_{step:09d}.safetensors
  final save:  {job_name}.safetensors                  (no step suffix)
"""

import os
import re
from enum import Enum

# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------

# Run names flow into root-executed shell strings (tmux session names, remote
# paths, the generated wrapper). Validate before anything else (R28).
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_run_name(run_name: str):
    if not RUN_NAME_RE.match(run_name or ""):
        raise ValueError(
            f"invalid run name {run_name!r}: must match [A-Za-z0-9_-]{{1,64}} "
            "(letters, digits, hyphen, underscore only)"
        )
    return run_name


def shell_quote(value: str) -> str:
    """Quote a value as an escaped single-quoted shell literal."""
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


# Pipeline-private known-hosts file, shared by EVERY ssh path (transport's
# ssh/rsync and pod's readiness probe). RunPod recycles IP:port pairs across
# customers, so the user's global known_hosts would hard-fail later pods in
# BatchMode once a recycled endpoint presents a different host key.
SSH_KNOWN_HOSTS_FILE = os.path.expanduser("~/.aitk_remote_known_hosts")

# Dedicated pipeline identity. The pipeline runs ssh with BatchMode=yes, which
# cannot prompt for a passphrase — a passphrase-protected personal key fails
# silently as "Permission denied (publickey)" (live-validated failure mode).
# Generate with: ssh-keygen -t ed25519 -N "" -f ~/.ssh/aitk_remote_ed25519
SSH_IDENTITY_FILE = os.path.expanduser("~/.ssh/aitk_remote_ed25519")


def ssh_identity_args() -> list:
    """-i/-o IdentitiesOnly args when the dedicated pipeline key exists."""
    if os.path.exists(SSH_IDENTITY_FILE):
        return ["-i", SSH_IDENTITY_FILE, "-o", "IdentitiesOnly=yes"]
    return []


# ---------------------------------------------------------------------------
# Remote filesystem layout (keyed by run name to prevent collisions)
# ---------------------------------------------------------------------------

REMOTE_RUNS_ROOT = "/workspace/runs"
REMOTE_REPO_DIR = "/app/ai-toolkit"

# rsync overlay allowlist: these directories/files are wholly local-fork-owned
# on the pod (--delete scoped INSIDE each dir, never at repo root).
OVERLAY_ALLOWLIST = ["extensions_built_in", "toolkit", "jobs", "run.py"]

# Upload exclude patterns: macOS AppleDouble junk and trainer latent caches.
UPLOAD_EXCLUDES = ["._*", ".DS_Store", "_latent_cache", "__pycache__"]


def remote_run_root(run_name: str) -> str:
    return f"{REMOTE_RUNS_ROOT}/{run_name}"


def remote_dataset_dir(run_name: str) -> str:
    return f"{remote_run_root(run_name)}/dataset"


def remote_ctrl_dir(run_name: str) -> str:
    return f"{remote_run_root(run_name)}/ctrl"


def remote_training_folder(run_name: str) -> str:
    """Value preflight writes into the derived config's training_folder."""
    return f"{remote_run_root(run_name)}/output"


def remote_job_dir(run_name: str, job_name: str = None) -> str:
    """Where the trainer actually writes: {training_folder}/{config.name}."""
    return f"{remote_training_folder(run_name)}/{job_name or run_name}"


def remote_samples_dir(run_name: str, job_name: str = None) -> str:
    return f"{remote_job_dir(run_name, job_name)}/samples"


def remote_config_path(run_name: str) -> str:
    return f"{remote_run_root(run_name)}/config.yaml"


def remote_log_path(run_name: str) -> str:
    return f"{remote_run_root(run_name)}/log.txt"


def remote_loss_db_path(run_name: str, job_name: str = None) -> str:
    return f"{remote_job_dir(run_name, job_name)}/loss_log.db"


# ---------------------------------------------------------------------------
# Sentinels and wrapper artifacts
# ---------------------------------------------------------------------------

EXIT_CODE_FILE = "exit_code"   # written by the wrapper when run.py exits
TIMED_OUT_FILE = "timed_out"   # written by the max-runtime timer before force-stop
PULLED_OK_FILE = "pulled.ok"   # touched by the laptop's verified final pull (ack)
WRAPPER_SCRIPT = "train_wrapper.sh"
TIMER_SCRIPT = "timer.sh"
SELF_STOP_SCRIPT = "self_stop.sh"  # provided by pod.py, uploaded by transport


def remote_sentinel_path(run_name: str, sentinel: str) -> str:
    return f"{remote_run_root(run_name)}/{sentinel}"


def tmux_train_session(run_name: str) -> str:
    return f"aitk-train-{run_name}"


def tmux_timer_session(run_name: str) -> str:
    return f"aitk-timer-{run_name}"


def trainer_pkill_pattern(run_name: str) -> str:
    """Pattern matching the trainer process for this run only.

    The [r] bracket trick keeps the pattern from matching the very shell that
    pgrep/pkill runs inside over ssh (whose command line contains the pattern
    verbatim) while still matching the real `python run.py ...<run>` process.
    """
    return f"[r]un.py.*{run_name}"


# ---------------------------------------------------------------------------
# Local layout
# ---------------------------------------------------------------------------

LOCAL_RUNS_DIR = "runs"      # runs/<run>/manifest.json, remote_config.yaml, mirrors
LOCAL_OUTPUT_DIR = "output"  # pulled artifacts mirror trainer layout: output/<run>/

# Local filename of the derived remote config written by preflight (R2/R19).
# The path shape is pinned by the plan ("runs/<run>/remote_config.yaml").
DERIVED_CONFIG_FILE = "remote_config.yaml"


def local_run_dir(run_name: str, base_dir: str = ".") -> str:
    return os.path.join(base_dir, LOCAL_RUNS_DIR, run_name)


def local_output_dir(run_name: str, base_dir: str = ".") -> str:
    return os.path.join(base_dir, LOCAL_OUTPUT_DIR, run_name)


# ---------------------------------------------------------------------------
# Disk sizing (R6) — constants adjustable here, not in pod.py
# ---------------------------------------------------------------------------

MODEL_CACHE_GB = 60      # Flux.2-class weights + HF cache
CHECKPOINT_GB = 0.6      # per-save LoRA checkpoint upper bound
HEADROOM_GB = 20         # samples, optimizer.pt, logs, slack


def disk_size_gb(total_steps: int, save_every: int) -> int:
    """volumeInGb = model cache + ceil(steps/save_every) * ckpt + headroom."""
    saves = -(-int(total_steps) // max(1, int(save_every)))  # ceil division
    return int(MODEL_CACHE_GB + saves * CHECKPOINT_GB + HEADROOM_GB + 0.999)


# ---------------------------------------------------------------------------
# Run states
# ---------------------------------------------------------------------------

class RunState(str, Enum):
    PREFLIGHTED = "PREFLIGHTED"
    PROVISIONING = "PROVISIONING"
    PROVISION_FAILED = "PROVISION_FAILED"
    POD_READY = "POD_READY"
    SYNCED = "SYNCED"
    RUNNING = "RUNNING"
    SAMPLING = "SAMPLING"      # sample-generation stall, not hung
    DEGRADED = "DEGRADED"      # recurring OOM-skip lines in the log
    COMPLETED = "COMPLETED"
    CRASHED = "CRASHED"
    STOPPED = "STOPPED"
    TIMED_OUT = "TIMED_OUT"    # max-runtime timer fired / pod stopped w/o sentinel
    UNKNOWN = "UNKNOWN"        # pod running, tmux gone, no sentinel
    POD_LOST = "POD_LOST"
    PULLED = "PULLED"
    TERMINATED = "TERMINATED"
    # DETACHED is computed at read time (pod alive, local watcher gone) and is
    # deliberately NOT a member: it must never be persisted to a manifest.


TERMINAL_STATES = {
    RunState.COMPLETED, RunState.CRASHED, RunState.STOPPED,
    RunState.TIMED_OUT, RunState.POD_LOST, RunState.PULLED,
    RunState.TERMINATED, RunState.PROVISION_FAILED,
}

# States that mean "training looks alive on the pod" (monitor/cli), and the
# superset lifecycle uses to hold STOPPED across the post-kill settle window
# (UNKNOWN = pod up, tmux gone, sentinel not yet written).
LIVE_STATES = {
    RunState.RUNNING.value,
    RunState.SAMPLING.value,
    RunState.DEGRADED.value,
}
SETTLE_STATES = LIVE_STATES | {RunState.UNKNOWN.value}

# watch exit codes (R17) — distinct codes so agent loops can branch.
EXIT_COMPLETED = 0
EXIT_RUNNING = 3           # watch --once: still running, nothing new
EXIT_NEW_SAMPLES = 10      # watch --once: new reviewable sample steps exist
EXIT_CRASHED = 20
EXIT_STOPPED = 21
EXIT_TIMED_OUT = 22
EXIT_UNKNOWN = 23
EXIT_POD_LOST = 24

WATCH_EXIT_CODES = {
    RunState.COMPLETED: EXIT_COMPLETED,
    RunState.CRASHED: EXIT_CRASHED,
    RunState.STOPPED: EXIT_STOPPED,
    RunState.TIMED_OUT: EXIT_TIMED_OUT,
    RunState.UNKNOWN: EXIT_UNKNOWN,
    RunState.POD_LOST: EXIT_POD_LOST,
}


# ---------------------------------------------------------------------------
# Filename grammars
# ---------------------------------------------------------------------------

SAMPLE_RE = re.compile(r"^(?P<time>\d+)__(?P<step>\d{9})_(?P<count>\d+)\.(?P<ext>jpg|jpeg|png|webp)$")


def parse_sample_filename(filename: str):
    """Return (step, count) for a trainer sample filename, else None."""
    m = SAMPLE_RE.match(os.path.basename(filename))
    if not m:
        return None
    return int(m.group("step")), int(m.group("count"))


def checkpoint_re(job_name: str):
    return re.compile(rf"^{re.escape(job_name)}_(?P<step>\d{{9}})\.safetensors$")


def parse_checkpoint_filename(filename: str, job_name: str):
    """Return the step int for a stepped checkpoint, else None."""
    m = checkpoint_re(job_name).match(os.path.basename(filename))
    if not m:
        return None
    return int(m.group("step"))


def is_final_checkpoint(filename: str, job_name: str) -> bool:
    """The terminal no-step save written when training completes."""
    return os.path.basename(filename) == f"{job_name}.safetensors"


# ---------------------------------------------------------------------------
# Log matchers
# ---------------------------------------------------------------------------
# The trainer's tqdm progress bar writes \r-separated updates that --log tees
# verbatim, so log text must be split on BOTH \r and \n before matching, and
# remote tails should use byte offsets (tail -c), never line counts.

OOM_SKIP_RE = re.compile(r"# OOM during training step, skipping batch")
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
# tqdm progress records look like: "name:  22%|##  | 865/4000 [3:21:26<12:10:03, 13.97s/it]"
PROGRESS_RE = re.compile(r"\d+/\d+\s*\[[^\]]*(?:s/it|it/s)")
# pre-training lifecycle output (model download / load / cache) — first runs
# cold-download Flux.2-class weights for 20-60 minutes before step 1.
WARMING_RE = re.compile(
    r"(?i)(downloading|fetching \d+ files|loading (?:model|checkpoint|pipeline|transformer|vae|te|text encoder)"
    r"|load(?:ing)? safetensors|caching latents|caching text embeddings|quantizing|fetching files)"
)


def split_log_records(text: str):
    """Split teed log text into records on both \\n and \\r."""
    return [r for r in re.split(r"[\r\n]+", text) if r.strip()]


def classify_log_tail(text: str) -> str:
    """Classify a log tail into 'traceback' | 'progress' | 'warming' | 'silent'.

    Precedence: a traceback anywhere wins (crash is terminal even if earlier
    records showed progress); otherwise the LAST matching record decides.
    """
    records = split_log_records(text)
    if any(TRACEBACK_RE.search(r) for r in records):
        return "traceback"
    for record in reversed(records):
        if PROGRESS_RE.search(record):
            return "progress"
        if WARMING_RE.search(record):
            return "warming"
    return "silent"


def count_oom_skips(text: str) -> int:
    return sum(1 for r in split_log_records(text) if OOM_SKIP_RE.search(r))
