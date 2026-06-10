"""Run manifest: the durable contract every pipeline subcommand reads/writes.

runs/<run>/manifest.json makes the pipeline re-entrant: a fresh session with
zero context can attach, watch, pull, and tear down from the manifest plus the
RunPod API alone.

Field classes:
  - authoritative: facts captured at the moment they happened (pod id, image
    tag, rates, timestamps, hashes, cadences, last-pulled steps).
  - cached observations: SSH endpoint and run state. RunPod reassigns IP/port
    across stop/start, so consumers must re-resolve the endpoint via pod.info
    before SSH; state is re-derived by monitor, the stored value is a hint.

Unknown fields survive round-trips (forward compatibility). Cost is derived,
never stored. DETACHED is computed at read time and never persisted.
"""

import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass, field

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.remote import contract

SCHEMA_VERSION = 1


class ManifestNotFoundError(FileNotFoundError):
    pass


@dataclass
class RunManifest:
    run_name: str
    schema_version: int = SCHEMA_VERSION

    # cached observations (re-derive before trusting)
    state: str = contract.RunState.PREFLIGHTED.value
    ssh_host: str = None  # cache only — re-resolve via pod.info before SSH
    ssh_port: int = None  # cache only

    # pod / provisioning (authoritative)
    pod_id: str = None
    image_tag: str = None
    image_repo_commit: str = None
    gpu_requested: str = None
    gpu_provisioned: str = None
    hourly_rate: float = None
    disk_gb: int = None
    provisioned_at: float = None      # epoch seconds
    terminated_at: float = None

    # launch (authoritative)
    launched_at: float = None
    max_runtime_deadline: float = None  # absolute epoch seconds
    max_grace_seconds: int = None
    tmux_session: str = None
    timer_session: str = None

    # config / code identity (authoritative)
    job_name: str = None              # config.name — checkpoint prefix + output folder
    config_hash: str = None           # sha256 of runs/<run>/remote_config.yaml
    overlay_git_sha: str = None
    overlay_dirty_hash: str = None

    # dataset fingerprint (authoritative)
    dataset_file_count: int = None
    dataset_total_bytes: int = None

    # cadences parsed from the config (authoritative)
    total_steps: int = None
    save_every: int = None
    sample_every: int = None
    prompt_count: int = None

    # pull / review progress (authoritative)
    last_pulled_sample_step: int = 0
    last_pulled_checkpoint_step: int = 0
    optimizer_pairing_step: int = None
    last_reviewed_step: int = 0
    last_sentinel: str = None         # mirrored exit_code value or "timed_out"
    # durable stop intent (authoritative): set by lifecycle.stop_run BEFORE
    # the pkill so monitor maps the eventual nonzero exit_code to STOPPED even
    # if a concurrent watch cycle clobbered state back to RUNNING (B5).
    stop_requested_at: float = None   # epoch seconds

    # forward compatibility: unknown keys survive round-trips
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    _FIELDS = None  # populated after class definition

    def to_dict(self) -> dict:
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "extra"}
        data.update(self.extra)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunManifest":
        known = {f.name for f in dataclasses.fields(cls) if f.name != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        if "run_name" not in kwargs:
            raise ValueError("manifest missing required field 'run_name'")
        return cls(extra=extra, **kwargs)

    # ------------------------------------------------------------------
    # derived values
    # ------------------------------------------------------------------

    def estimated_cost(self, now: float = None) -> float:
        """elapsed (provisioned→terminated|now) × hourly rate. None if unknown."""
        if self.hourly_rate is None or self.provisioned_at is None:
            return None
        end = self.terminated_at if self.terminated_at is not None else (now if now is not None else time.time())
        hours = max(0.0, (end - self.provisioned_at) / 3600.0)
        return round(hours * self.hourly_rate, 2)

    def expected_checkpoint_count(self, observed_max_step: int = None) -> int:
        """floor(max_step / save_every) — stepped saves land ON multiples.

        Uses the observed max trained step so early-stopped runs verify
        cleanly. The trainer loop is `for step in range(start, total_steps)`,
        so a COMPLETED run's last loop step is total_steps - 1: callers must
        pass total_steps - 1 (NOT total_steps) for completed runs — the
        no-suffix final save written after the loop is checked separately (R9).
        """
        if not self.save_every:
            return 0
        step = observed_max_step if observed_max_step is not None else 0
        return int(step) // int(self.save_every)


def manifest_path(run_name: str, base_dir: str = ".") -> str:
    return os.path.join(contract.local_run_dir(run_name, base_dir), "manifest.json")


def load(run_name: str, base_dir: str = ".") -> RunManifest:
    path = manifest_path(run_name, base_dir)
    if not os.path.exists(path):
        raise ManifestNotFoundError(
            f"no manifest for run '{run_name}' (expected {path}); "
            "run preflight first, or check the run name"
        )
    with open(path, "r") as f:
        return RunManifest.from_dict(json.load(f))


def save(m: RunManifest, base_dir: str = ".") -> str:
    """Atomic write: temp file in the same directory, then os.replace."""
    path = manifest_path(m.run_name, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(m.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path
