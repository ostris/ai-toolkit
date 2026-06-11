"""SSH exec and rsync up/down transport for the remote-GPU training pipeline (U4).

Every ssh/rsync invocation is built by a PURE command-builder function that
returns a list[str] (unit-testable without a network) and executed by a thin
runner (default subprocess.run, injectable in tests).

Covers:
  R5  — upload filters exclude AppleDouble junk, .DS_Store, latent caches.
  R15 — incremental pull, never --delete on downloads, mtime stability vs the
        POD's clock (not the laptop's), optimizer.pt snapshot paired to the
        newest pulled checkpoint only.
  R19 — config drift detection against the manifest's recorded hash.
  R23 — pulled artifacts mirror the trainer's own layout under
        output/<run_name>/ so existing review tooling works unchanged.
  R25 — every pull mirrors the exit-code/timed-out sentinels and the log tail
        (plus loss_log.db on terminal pulls) into runs/<run>/.

The SSH endpoint is always passed in explicitly (host, port). Callers
re-resolve it via pod.get_pod_info before every operation — this module must
NOT import scripts.remote.pod (it owns provider-specific strings and is built
in parallel as U3).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.remote import contract
from scripts.remote.manifest import RunManifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Stability thresholds, evaluated against the POD's clock (R15). A file whose
# pod-mtime is younger than its threshold may still be mid-write; skip it this
# cycle and pick it up on the next pull.
SAMPLE_STABILITY_SECONDS = 120
CHECKPOINT_STABILITY_SECONDS = 60

SSH_TIMEOUT = 60            # seconds, for short remote commands
RSYNC_TIMEOUT = 3600        # seconds — checkpoint transfers are large (#18)
LOG_TAIL_BYTES = 16384      # ~16KB of log mirrored on every pull (R25)
LOG_TAIL_FILE = "log_tail.txt"
LOSS_DB_FILE = "loss_log.db"

# Local filename of the derived remote config (canonical home: contract.py).
DERIVED_CONFIG_FILE = contract.DERIVED_CONFIG_FILE


# ---------------------------------------------------------------------------
# Endpoint and runners
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    host: str
    port: int
    user: str = "root"


def ssh_base_args(ep: Endpoint) -> list:
    """Base argv for ssh to the pod. Pure."""
    return [
        "ssh",
        "-p", str(ep.port),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={contract.SSH_KNOWN_HOSTS_FILE}",
        *contract.ssh_identity_args(),
        f"{ep.user}@{ep.host}",
    ]


def _ssh_transport_arg(ep: Endpoint) -> str:
    """The -e value for rsync: the ssh command without the destination."""
    return " ".join(ssh_base_args(ep)[:-1])


def _remote_spec(ep: Endpoint, remote_path: str) -> str:
    return f"{ep.user}@{ep.host}:{remote_path}"


def ssh_run(ep: Endpoint, command: str, *, timeout: float = SSH_TIMEOUT,
            runner=subprocess.run) -> subprocess.CompletedProcess:
    """Run a single remote command; returns CompletedProcess (caller checks)."""
    return runner(ssh_base_args(ep) + [command], text=True,
                  capture_output=True, timeout=timeout)


def _run(cmd: list, runner, timeout: float = RSYNC_TIMEOUT) -> subprocess.CompletedProcess:
    """Execute a built command (rsync legs, local git). Default timeout is
    RSYNC_TIMEOUT — checkpoint transfers are large; ssh_run keeps SSH_TIMEOUT."""
    return runner(cmd, text=True, capture_output=True, timeout=timeout)


def _check(proc: subprocess.CompletedProcess, what: str) -> subprocess.CompletedProcess:
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"{what} failed (exit {proc.returncode}): {stderr[:2000]}")
    return proc


def _warn(message: str):
    print(f"[transport] warning: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# rsync command builders (pure)
# ---------------------------------------------------------------------------

def build_rsync_up(local_dir: str, ep: Endpoint, remote_dir: str, *,
                   excludes=None, delete_within: bool = False) -> list:
    """Build (never run) an upload rsync command.

    Paths are used verbatim: pass a trailing slash on both sides to copy
    directory CONTENTS (the overlay/dataset convention). delete_within=True
    adds --delete — only valid when the destination is wholly owned by the
    source (overlay allowlist dirs); never used for repo root or downloads.
    """
    if excludes is None:
        excludes = contract.UPLOAD_EXCLUDES
    cmd = ["rsync", "-rlptz"]  # no -o/-g: chown EPERM on /workspace volumes
    for pattern in excludes:
        cmd.append(f"--exclude={pattern}")
    if delete_within:
        cmd.append("--delete")
    cmd += ["-e", _ssh_transport_arg(ep), str(local_dir), _remote_spec(ep, remote_dir)]
    return cmd


def build_rsync_down(ep: Endpoint, remote_path: str, local_dir: str, *,
                     excludes=(), files_from: str = None) -> list:
    """Build (never run) a download rsync command. NEVER includes --delete."""
    cmd = ["rsync", "-rlptz"]  # no -o/-g: chown EPERM on /workspace volumes
    for pattern in excludes:
        cmd.append(f"--exclude={pattern}")
    if files_from:
        cmd.append(f"--files-from={files_from}")
    cmd += ["-e", _ssh_transport_arg(ep), _remote_spec(ep, remote_path), str(local_dir)]
    # R15 invariant: downloads must never delete local artifacts.
    assert not any(a == "--delete" or a.startswith("--delete") for a in cmd), \
        "download rsync must never contain --delete"
    return cmd


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

def upload_dataset(ep: Endpoint, local_dataset_dir: str, run_name: str, *,
                   excludes=None, runner=subprocess.run) -> str:
    """Upload a dataset folder to /workspace/runs/<run>/dataset/ (R5 filters)."""
    remote_dir = contract.remote_dataset_dir(run_name)
    _check(ssh_run(ep, f"mkdir -p {contract.shell_quote(remote_dir)}", runner=runner),
           f"mkdir {remote_dir}")
    cmd = build_rsync_up(local_dataset_dir.rstrip("/") + "/", ep, remote_dir + "/",
                         excludes=excludes)
    _check(_run(cmd, runner), f"dataset upload to {remote_dir}")
    return remote_dir


def upload_file(ep: Endpoint, local_path: str, remote_path: str, *,
                runner=subprocess.run) -> str:
    """Upload a single file to an exact remote path (parent dirs created)."""
    parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
    _check(ssh_run(ep, f"mkdir -p {contract.shell_quote(parent)}", runner=runner),
           f"mkdir {parent}")
    cmd = build_rsync_up(local_path, ep, remote_path, excludes=())
    _check(_run(cmd, runner), f"file upload to {remote_path}")
    return remote_path


def upload_overlay(ep: Endpoint, repo_root: str, *, runner=subprocess.run) -> dict:
    """rsync the local-fork overlay onto the image repo (key decision: overlay).

    Each contract.OVERLAY_ALLOWLIST entry is rsynced onto
    contract.REMOTE_REPO_DIR with --delete scoped INSIDE that directory
    (rsync local/dir/ root@h:/app/ai-toolkit/dir/ --delete) so each overlaid
    dir is wholly local-fork-owned. --delete is NEVER applied at repo root.

    Returns code-identity facts for the manifest (caller records them):
      image_repo_commit  — `git -C /app/ai-toolkit rev-parse HEAD`, read FIRST
                           (before the overlay rewrites the tree)
      overlay_git_sha    — local `git rev-parse HEAD`
      overlay_dirty_hash — sha256 of local `git diff` output
    """
    # 1. read the image repo's commit BEFORE overlaying
    res = ssh_run(ep, f"git -C {contract.REMOTE_REPO_DIR} rev-parse HEAD", runner=runner)
    image_repo_commit = res.stdout.strip() if (res.returncode == 0 and res.stdout.strip()) else None
    if image_repo_commit is None:
        _warn("could not read image repo commit from the pod")

    # 2. local code identity
    res = _run(["git", "-C", repo_root, "rev-parse", "HEAD"], runner)
    overlay_git_sha = res.stdout.strip() if res.returncode == 0 else None
    res = _run(["git", "-C", repo_root, "diff"], runner)
    overlay_dirty_hash = (
        hashlib.sha256((res.stdout or "").encode("utf-8")).hexdigest()
        if res.returncode == 0 else None
    )

    # 3. overlay each allowlisted entry
    for entry in contract.OVERLAY_ALLOWLIST:
        local_path = os.path.join(repo_root, entry)
        remote_path = f"{contract.REMOTE_REPO_DIR}/{entry}"
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"overlay entry missing locally: {local_path}")
        if os.path.isdir(local_path):
            # contents-onto-contents with --delete scoped inside this dir only
            cmd = build_rsync_up(local_path.rstrip("/") + "/", ep, remote_path + "/",
                                 delete_within=True)
        else:
            cmd = build_rsync_up(local_path, ep, remote_path, excludes=())
        _check(_run(cmd, runner), f"overlay upload of {entry}")

    return {
        "overlay_git_sha": overlay_git_sha,
        "overlay_dirty_hash": overlay_dirty_hash,
        "image_repo_commit": image_repo_commit,
    }


# ---------------------------------------------------------------------------
# Pod clock
# ---------------------------------------------------------------------------

def pod_clock_offset(ep: Endpoint, *, runner=subprocess.run, now: float = None) -> float:
    """pod_time - local_time. mtime stability is evaluated vs the POD's clock."""
    res = _check(ssh_run(ep, "date +%s", runner=runner), "pod clock read")
    pod_time = float(res.stdout.strip())
    local = now if now is not None else time.time()
    return pod_time - local


# ---------------------------------------------------------------------------
# Remote artifact inventory
# ---------------------------------------------------------------------------

@dataclass
class ArtifactEntry:
    kind: str        # 'sample' | 'checkpoint' | 'final_checkpoint' | 'optimizer'
    name: str        # path relative to the remote job dir
    step: int        # None for final_checkpoint / optimizer
    size: int
    mtime: float     # pod-clock epoch seconds


@dataclass
class ArtifactInventory:
    entries: list = field(default_factory=list)

    def samples_by_step(self) -> dict:
        out = {}
        for e in self.entries:
            if e.kind == "sample":
                out.setdefault(e.step, []).append(e)
        return out

    def checkpoints(self) -> list:
        return sorted((e for e in self.entries if e.kind == "checkpoint"),
                      key=lambda e: e.step)

    def final_checkpoint(self):
        for e in self.entries:
            if e.kind == "final_checkpoint":
                return e
        return None

    def optimizer(self):
        for e in self.entries:
            if e.kind == "optimizer":
                return e
        return None

    def max_checkpoint_step(self) -> int:
        ckpts = self.checkpoints()
        return ckpts[-1].step if ckpts else 0


# Emitted by _find_command when the job dir does not exist yet, so callers can
# tell "no artifacts yet" (truthful empty inventory) from a transport failure.
DIR_ABSENT_MARKER = "__ABSENT__"


def _find_command(job_dir: str) -> str:
    q = contract.shell_quote(job_dir)
    find = (
        f"find {q} -maxdepth 2 -type f "
        "\\( -path '*/samples/*' -o -name '*.safetensors' -o -name 'optimizer.pt' \\) "
        "-printf '%P\\t%s\\t%T@\\n'"
    )
    return f"if [ -d {q} ]; then {find}; else echo {DIR_ABSENT_MARKER}; fi"


def parse_find_output(text: str, job_name: str) -> ArtifactInventory:
    """Parse `find -printf '%P\\t%s\\t%T@\\n'` output via contract grammars. Pure."""
    inv = ArtifactInventory()
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        rel, size_s, mtime_s = parts
        try:
            size = int(size_s)
            mtime = float(mtime_s)
        except ValueError:
            continue
        base = os.path.basename(rel)
        if rel.startswith("samples/"):
            parsed = contract.parse_sample_filename(base)
            if parsed is None:
                continue  # AppleDouble junk etc.
            step, _count = parsed
            inv.entries.append(ArtifactEntry("sample", rel, step, size, mtime))
        elif base == "optimizer.pt":
            inv.entries.append(ArtifactEntry("optimizer", rel, None, size, mtime))
        else:
            step = contract.parse_checkpoint_filename(base, job_name)
            if step is not None:
                inv.entries.append(ArtifactEntry("checkpoint", rel, step, size, mtime))
            elif contract.is_final_checkpoint(base, job_name):
                inv.entries.append(ArtifactEntry("final_checkpoint", rel, None, size, mtime))
    return inv


def list_remote_artifacts(ep: Endpoint, m: RunManifest, *,
                          runner=subprocess.run) -> ArtifactInventory:
    """One ssh `find` over the remote job dir, parsed with contract grammars.

    A missing job dir (pre-first-save) is a truthful empty inventory, signalled
    by the explicit DIR_ABSENT_MARKER. Any nonzero exit (ssh 255, find errors)
    RAISES — silently returning empty would let a transient transport failure
    masquerade as "no artifacts" and stall watermark-driven pulls.
    """
    job_name = m.job_name or m.run_name
    job_dir = contract.remote_job_dir(m.run_name, job_name)
    res = ssh_run(ep, _find_command(job_dir), runner=runner)
    if res.returncode != 0:
        raise RuntimeError(
            f"artifact listing failed (ssh): exit {res.returncode}: "
            f"{(res.stderr or '').strip()[:2000]}")
    if DIR_ABSENT_MARKER in (res.stdout or ""):
        return ArtifactInventory()  # job dir not created yet (pre-first-save)
    return parse_find_output(res.stdout, job_name)


def _stat_sizes(ep: Endpoint, job_dir: str, names: list, *, runner) -> dict:
    """Second stat for candidate checkpoints: {rel_name: size}. Missing = unstable."""
    paths = " ".join(contract.shell_quote(f"{job_dir}/{n}") for n in names)
    res = ssh_run(ep, f"stat --printf '%n\\t%s\\n' {paths}", runner=runner)
    abs_sizes = {}
    if res.returncode == 0:
        for line in res.stdout.splitlines():
            if "\t" not in line:
                continue
            path, size_s = line.rsplit("\t", 1)
            try:
                abs_sizes[path] = int(size_s)
            except ValueError:
                continue
    out = {}
    for n in names:
        abs_path = f"{job_dir}/{n}"
        if abs_path in abs_sizes:
            out[n] = abs_sizes[abs_path]
    return out


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

SAFETENSORS_HEADER_MAX = 100 * 1024 * 1024  # sanity cap on the JSON header


def validate_safetensors_container(path: str) -> bool:
    """Cheap container integrity check for a pulled .safetensors file.

    Reads the first 8 bytes as a little-endian uint64 header length, then
    json-parses the header. Catches truncated/garbage transfers without
    touching tensor data. Returns False on any structural problem; never
    raises. Deliberately NOT applied to optimizer.pt (torch pickle).
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) != 8:
                return False
            header_len = int.from_bytes(head, "little")
            if not 0 < header_len <= SAFETENSORS_HEADER_MAX:
                return False
            header = f.read(header_len)
            if len(header) != header_len:
                return False
            json.loads(header.decode("utf-8"))
        return True
    except (OSError, ValueError, UnicodeDecodeError):
        return False


@dataclass
class PullResult:
    new_sample_steps: list = field(default_factory=list)      # complete batches, sorted
    new_checkpoint_steps: list = field(default_factory=list)  # sorted
    pulled_files: int = 0
    partial_steps: list = field(default_factory=list)         # < prompt_count but old mtime; pulled with warning
    skipped_unstable: list = field(default_factory=list)      # names skipped this cycle
    optimizer_pairing_step: int = None


def pull_artifacts(ep: Endpoint, m: RunManifest, *, base_dir: str = ".",
                   now: float = None, runner=subprocess.run) -> PullResult:
    """Incrementally pull new samples/checkpoints into output/<run>/ (R15/R23).

    Mirrors the trainer's own layout (samples/ subdir, checkpoints at the
    root) so ai-toolkit-sample-reviewer works unchanged. Mutates the manifest
    watermarks in memory; the CALLER saves the manifest after success.
    Sentinels are mirrored on every pull (R25).
    """
    local_now = time.time() if now is None else float(now)
    offset = pod_clock_offset(ep, runner=runner, now=local_now)
    pod_now = local_now + offset

    inv = list_remote_artifacts(ep, m, runner=runner)
    result = PullResult()
    files_to_pull = []

    # --- samples (grouped per step; R16 batch semantics) ------------------
    by_step = inv.samples_by_step()
    for step in sorted(by_step):
        if step <= (m.last_pulled_sample_step or 0):
            continue
        entries = by_step[step]
        newest_age = pod_now - max(e.mtime for e in entries)
        stable = newest_age >= SAMPLE_STABILITY_SECONDS
        if m.prompt_count:
            complete = len(entries) >= m.prompt_count
        else:
            complete = stable  # no count contract; stability is the only signal
        if complete:
            files_to_pull.extend(e.name for e in entries)
            result.new_sample_steps.append(step)
        elif stable:
            files_to_pull.extend(e.name for e in entries)
            result.partial_steps.append(step)
            _warn(f"step {step}: partial sample batch "
                  f"({len(entries)}/{m.prompt_count}) with old mtime — pulled anyway")
        else:
            result.skipped_unstable.extend(e.name for e in entries)

    # --- checkpoints (mtime threshold + second-stat size stability) -------
    job_name = m.job_name or m.run_name
    job_dir = contract.remote_job_dir(m.run_name, job_name)
    candidates = []
    for e in inv.checkpoints():
        if e.step <= (m.last_pulled_checkpoint_step or 0):
            continue
        if pod_now - e.mtime < CHECKPOINT_STABILITY_SECONDS:
            result.skipped_unstable.append(e.name)
            continue
        candidates.append(e)
    final_candidate = inv.final_checkpoint()
    if final_candidate is not None and pod_now - final_candidate.mtime < CHECKPOINT_STABILITY_SECONDS:
        result.skipped_unstable.append(final_candidate.name)
        final_candidate = None

    stat_targets = candidates + ([final_candidate] if final_candidate else [])
    stable_ckpts = []
    if stat_targets:
        sizes = _stat_sizes(ep, job_dir, [e.name for e in stat_targets], runner=runner)
        for e in candidates:
            if sizes.get(e.name) == e.size:
                stable_ckpts.append(e)
            else:
                result.skipped_unstable.append(e.name)
        if final_candidate is not None and sizes.get(final_candidate.name) != final_candidate.size:
            result.skipped_unstable.append(final_candidate.name)
            final_candidate = None
        for e in stable_ckpts:
            files_to_pull.append(e.name)
            result.new_checkpoint_steps.append(e.step)
        if final_candidate is not None:
            files_to_pull.append(final_candidate.name)

    # --- execute the download (one rsync via --files-from) ----------------
    out_dir = contract.local_output_dir(m.run_name, base_dir)
    if files_to_pull:
        os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
        fd, list_path = tempfile.mkstemp(prefix=".aitk-pull-", suffix=".txt")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(files_to_pull) + "\n")
            cmd = build_rsync_down(ep, job_dir + "/", out_dir + os.sep,
                                   files_from=list_path)
            _check(_run(cmd, runner), "artifact pull")
        finally:
            os.unlink(list_path)
        result.pulled_files += len(files_to_pull)

        # --- integrity-validate every newly pulled checkpoint BEFORE it is
        # counted or the watermark advances (#10). A corrupt copy is deleted
        # locally and left re-pullable next cycle. optimizer.pt is a torch
        # pickle — deliberately NOT validated here.
        for e in stable_ckpts + ([final_candidate] if final_candidate else []):
            local_path = os.path.join(out_dir, e.name)
            if validate_safetensors_container(local_path):
                continue
            try:
                os.unlink(local_path)
            except OSError:
                pass
            if e.step is not None and e.step in result.new_checkpoint_steps:
                result.new_checkpoint_steps.remove(e.step)
            result.skipped_unstable.append(f"{e.name} (failed integrity check)")
            result.pulled_files -= 1
            _warn(f"{e.name}: failed integrity check after pull — local copy "
                  "deleted; will re-pull next cycle")

    # --- optimizer.pt snapshot, paired to the NEWEST pulled checkpoint ----
    # The trainer overwrites one optimizer.pt per save: only the newest
    # checkpoint can carry optimizer state. Pair only when the newest pulled
    # checkpoint is the newest checkpoint, period — if a fresher one was
    # skipped as unstable, optimizer.pt likely belongs to IT, not ours.
    opt = inv.optimizer()
    if opt is not None and result.new_checkpoint_steps:
        newest = max(result.new_checkpoint_steps)
        if newest == inv.max_checkpoint_step():
            os.makedirs(out_dir, exist_ok=True)
            opt_local = os.path.join(out_dir, f"optimizer_{newest:09d}.pt")
            cmd = build_rsync_down(ep, f"{job_dir}/{opt.name}", opt_local)
            res = _run(cmd, runner)
            if res.returncode == 0:
                result.optimizer_pairing_step = newest
                m.optimizer_pairing_step = newest
                result.pulled_files += 1
            else:
                _warn(f"optimizer.pt snapshot failed: {(res.stderr or '').strip()[:500]}")

    # --- watermarks (caller saves the manifest) ----------------------------
    # Partial steps are deliberately EXCLUDED from the sample watermark (#11):
    # they must stay below it so the next cycle re-pulls them, and they report
    # in new_sample_steps exactly once — on the cycle the batch completes.
    if result.new_sample_steps:
        m.last_pulled_sample_step = max(
            [m.last_pulled_sample_step or 0] + result.new_sample_steps)
    if result.new_checkpoint_steps:
        m.last_pulled_checkpoint_step = max(
            [m.last_pulled_checkpoint_step or 0] + result.new_checkpoint_steps)

    # --- sentinels, on EVERY pull (R25) ------------------------------------
    mirror_sentinels(ep, m, base_dir=base_dir, runner=runner)

    result.new_sample_steps.sort()
    result.new_checkpoint_steps.sort()
    result.partial_steps.sort()
    return result


# ---------------------------------------------------------------------------
# Sentinel mirroring (R25)
# ---------------------------------------------------------------------------

def mirror_sentinels(ep: Endpoint, m: RunManifest, *, base_dir: str = ".",
                     runner=subprocess.run, terminal: bool = None) -> dict:
    """Mirror exit_code/timed_out + log tail into runs/<run>/; tolerate absence.

    On terminal pulls (a sentinel found, m.state terminal, or terminal=True)
    also mirrors loss_log.db so the run stays diagnosable after teardown.
    Updates m.last_sentinel in memory; the caller saves the manifest.
    """
    run_dir = contract.local_run_dir(m.run_name, base_dir)
    os.makedirs(run_dir, exist_ok=True)
    found = {}

    for sentinel in (contract.EXIT_CODE_FILE, contract.TIMED_OUT_FILE):
        remote = contract.remote_sentinel_path(m.run_name, sentinel)
        res = ssh_run(ep, f"cat {contract.shell_quote(remote)} 2>/dev/null", runner=runner)
        # present = readable with content (sentinels are written with content)
        if res.returncode == 0 and (res.stdout or "").strip() != "":
            with open(os.path.join(run_dir, sentinel), "w") as f:
                f.write(res.stdout)
            found[sentinel] = res.stdout.strip()

    if contract.TIMED_OUT_FILE in found:
        m.last_sentinel = "timed_out"
    elif contract.EXIT_CODE_FILE in found:
        m.last_sentinel = found[contract.EXIT_CODE_FILE]

    log_remote = contract.remote_log_path(m.run_name)
    res = ssh_run(ep, f"tail -c {LOG_TAIL_BYTES} {contract.shell_quote(log_remote)} 2>/dev/null",
                  runner=runner)
    log_mirrored = False
    if res.returncode == 0 and res.stdout:
        with open(os.path.join(run_dir, LOG_TAIL_FILE), "w") as f:
            f.write(res.stdout)
        log_mirrored = True

    if terminal is None:
        terminal_values = {s.value for s in contract.TERMINAL_STATES}
        terminal = bool(found) or m.state in terminal_values
    loss_db_mirrored = False
    if terminal:
        job_name = m.job_name or m.run_name
        cmd = build_rsync_down(ep, contract.remote_loss_db_path(m.run_name, job_name),
                               os.path.join(run_dir, LOSS_DB_FILE))
        res = _run(cmd, runner)
        loss_db_mirrored = res.returncode == 0
        if not loss_db_mirrored:
            _warn("loss_log.db mirror failed (absent or unreadable) — tolerated")

    return {
        "exit_code": found.get(contract.EXIT_CODE_FILE),
        "timed_out": contract.TIMED_OUT_FILE in found,
        "log_tail_mirrored": log_mirrored,
        "loss_db_mirrored": loss_db_mirrored,
    }


# ---------------------------------------------------------------------------
# Config drift (R19)
# ---------------------------------------------------------------------------

def check_config_drift(m: RunManifest, *, base_dir: str = ".") -> bool:
    """True when runs/<run>/remote_config.yaml no longer matches m.config_hash.

    Call sites warn when the run is RUNNING-ish: an edited derived config no
    longer describes what the pod is actually training.
    """
    if not m.config_hash:
        return False  # nothing recorded yet; drift is undefined
    path = os.path.join(contract.local_run_dir(m.run_name, base_dir),
                        contract.DERIVED_CONFIG_FILE)
    if not os.path.exists(path):
        return True  # recorded hash but the derived config vanished: drifted
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return digest != m.config_hash
