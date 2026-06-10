"""Pod lifecycle: the RunPod control plane wrapper (U3).

Thin wrapper over the `runpod` Python SDK. This module owns EVERY
provider-specific string in the pipeline (R21, KTD "self-stop is a designed
mechanism"): the create request shape, the container start-command override
(R27), the pod-side self-stop script, and the GPU type ids. Other units only
see PodInfo and the manifest.

SDK handling:
  - `runpod` is imported lazily inside `_get_sdk()`; every public function
    takes `sdk=None` and uses `sdk or _get_sdk()` so tests inject a mock
    module and never import the real SDK.
  - Each SDK call is isolated in a tiny `_sdk_*` adapter function so a
    live-run signature fix touches exactly one place. The runpod-python
    surface used (per its docs, 1.x): create_pod, get_pod, stop_pod,
    resume_pod, terminate_pod. Signatures are flagged UNCERTAIN where the
    docs are thin; verify in U5's gating live validation.

Failure behavior (R6/R7):
  - Out-of-stock GPU types fail fast with OutOfStockError naming the type;
    substitution only via an explicit `gpu_fallback` list.
  - wait_for_ready never leaves a half-provisioned pod billing: on SSH or
    probe timeout it terminates the pod, then raises SshTimeoutError.
"""

import dataclasses
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.remote import contract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RunPod gpu_type_id-style display name for the default GPU (KTD: A100 80GB
# Secure Cloud ~$1.89/hr is the cheapest non-preemptible confirmed path).
DEFAULT_GPU = "NVIDIA A100 80GB PCIe"

# Short names accepted anywhere a gpu_type is passed; values are RunPod
# gpu_type_id display names. Unknown names pass through unchanged so any
# exact RunPod id keeps working.
GPU_ALIASES = {
    "A100": "NVIDIA A100 80GB PCIe",
    "A100 80GB": "NVIDIA A100 80GB PCIe",
    "A100 SXM": "NVIDIA A100-SXM4-80GB",
    "H100": "NVIDIA H100 80GB HBM3",
    "H100 PCIE": "NVIDIA H100 PCIe",
    "4090": "NVIDIA GeForce RTX 4090",
    "5090": "NVIDIA GeForce RTX 5090",
    "L40S": "NVIDIA L40S",
    # 96GB Blackwell workstation/server cards (verified live ids, June 2026).
    # MaxQ is routinely the cheapest 80GB+ option on Secure Cloud (~$0.50/hr).
    "MAXQ": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "PRO 6000 MAXQ": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "RTX PRO 6000 MAXQ": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "PRO 6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "RTX PRO 6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "PRO 6000 WK": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
}

# Pinned-tag policy (KTD): callers should pass an explicit pinned tag chosen
# at implementation time and record it in the manifest — never rely on
# :latest for real runs. This default is kept overridable via the
# AITK_REMOTE_IMAGE env var; create_pod resolves image_tag param > env > this.
DEFAULT_IMAGE_TAG = "ostris/aitoolkit:latest"

# Container disk holds the image layers + tmp; the big stuff (dataset,
# checkpoints, HF cache if redirected) lives on the /workspace volume whose
# size is the explicit `disk_gb` (contract.disk_size_gb formula, R6).
CONTAINER_DISK_GB = 50

# Env var names that hold secrets — redacted in dry-run output.
_SECRET_ENV_KEYS = ("RUNPOD_STOP_KEY", "HF_TOKEN")

_INSTALL_HINT = "pip install -r scripts/remote/requirements.txt"

# Substrings that mark a create failure as out-of-stock rather than a bug.
# RunPod's GraphQL error reads like: "There are no longer any instances
# available with the requested specifications. Please refresh and try again."
_OUT_OF_STOCK_MARKERS = (
    "no longer any instances available",
    "no instances available",
    "no gpu available",
    "not currently available",
    "out of stock",
)

# Substrings that mark a pod-API error as "pod is gone" (GONE / idempotent
# terminate). runpod-python's get_pod returns None for a missing pod via the
# GraphQL data path; REST-side errors surface as 404s.
_GONE_MARKERS = (
    "404",
    "not found",
    "does not exist",
    "doesn't exist",
    "no longer exists",
)


class PodError(Exception):
    pass


class OutOfStockError(PodError):
    pass


class SshTimeoutError(PodError):
    pass


@dataclass
class PodInfo:
    pod_id: str
    status: str               # 'RUNNING' | 'EXITED' | 'GONE' | raw desiredStatus
    ssh_host: str = None      # direct public IP (proxy SSH can't carry rsync)
    ssh_port: int = None
    gpu_type: str = None      # provisioned GPU (display name)
    hourly_rate: float = None


def _warn(msg: str):
    print(f"[pod] WARNING: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# SDK access (lazy import; injectable for tests)
# ---------------------------------------------------------------------------

def _get_sdk():
    try:
        import runpod
    except ImportError as e:
        raise PodError(
            "the 'runpod' SDK is not installed; install the laptop-side "
            f"dependencies with: {_INSTALL_HINT}"
        ) from e
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise PodError(
            "RUNPOD_API_KEY is not set; export it (or add it to .env) "
            "before running pod operations"
        )
    runpod.api_key = api_key
    return runpod


# --- one adapter per SDK call: a live-run signature fix touches one place ---

def _sdk_create_pod(sdk, request: dict):
    # runpod.create_pod(name, image_name, gpu_type_id, cloud_type=...,
    #   support_public_ip=..., start_ssh=..., volume_in_gb=...,
    #   container_disk_in_gb=..., volume_mount_path=..., ports=...,
    #   env=..., docker_args=...) — per runpod-python docs. UNCERTAIN:
    # whether docker_args is the exact start-command override param name.
    return sdk.create_pod(**request)


def _sdk_get_pod(sdk, pod_id: str):
    # Returns the pod dict (desiredStatus, costPerHr, machine.gpuDisplayName,
    # runtime.ports[...]) or None when the pod no longer exists.
    return sdk.get_pod(pod_id)


def _sdk_stop_pod(sdk, pod_id: str):
    return sdk.stop_pod(pod_id)


def _sdk_resume_pod(sdk, pod_id: str, gpu_count: int):
    # UNCERTAIN: runpod-python documents resume_pod(pod_id, gpu_count);
    # gpu_count=0 is the rescue path (start a stopped pod CPU-only to pull
    # artifacts off its volume).
    return sdk.resume_pod(pod_id, gpu_count=gpu_count)


def _sdk_terminate_pod(sdk, pod_id: str):
    return sdk.terminate_pod(pod_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_gpu_type(name: str) -> str:
    """Map a short alias ('A100', 'H100', '4090') to a RunPod gpu type id;
    unknown names pass through so exact RunPod ids keep working."""
    if not name:
        return DEFAULT_GPU
    key = str(name).strip()
    if key in GPU_ALIASES:
        return GPU_ALIASES[key]
    upper = {k.upper(): v for k, v in GPU_ALIASES.items()}
    return upper.get(key.upper(), key)


def read_public_key(ssh_dir: str = None) -> str:
    """Read the user's SSH public key for the pod's PUBLIC_KEY env
    (docker/start.sh appends it to root's authorized_keys)."""
    ssh_dir = ssh_dir or os.path.expanduser("~/.ssh")
    for name in ("aitk_remote_ed25519.pub", "id_ed25519.pub", "id_rsa.pub"):
        path = os.path.join(ssh_dir, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read().strip()
    raise PodError(
        f"no SSH public key found in {ssh_dir} (looked for aitk_remote_ed25519.pub, id_ed25519.pub, "
        "id_rsa.pub); generate one with: ssh-keygen -t ed25519"
    )


def stop_key_env() -> dict:
    """Env injected into the pod for self-stop auth (R21).

    Prefers a dedicated restricted key (RUNPOD_STOP_API_KEY) so the
    pod-resident credential is scoped to the minimum RunPod's key model
    allows — never name or default to the full account key without warning.
    """
    scoped = os.environ.get("RUNPOD_STOP_API_KEY")
    if scoped:
        return {"RUNPOD_STOP_KEY": scoped}
    fallback = os.environ.get("RUNPOD_API_KEY")
    if fallback:
        _warn(
            "RUNPOD_STOP_API_KEY not set; falling back to the full account "
            "API key for pod-side self-stop. Create a restricted key in the "
            "RunPod console and export RUNPOD_STOP_API_KEY (R21)."
        )
        return {"RUNPOD_STOP_KEY": fallback}
    _warn(
        "no RunPod API key available for pod-side self-stop; self-stop will "
        "be disabled (the max-runtime timer and laptop-side `down` are the "
        "only cost backstops)"
    )
    return {}


def validate_stop_key() -> bool:
    """Pre-spend check: can the key the pod will receive actually stop pods?

    Live-validated failure mode: a Read-Only restricted key authenticates
    nothing on the pods REST API (401), so self-stop fails OPEN and a
    finished pod idle-bills until the laptop notices. Probing the stop
    endpoint with a sentinel pod id distinguishes auth (401 -> bad key)
    from existence (4xx/5xx 'does not exist' -> key fine). Warn-only.
    """
    env = stop_key_env()
    key = env.get("RUNPOD_STOP_KEY")
    if not key:
        return False  # already warned by stop_key_env
    import urllib.request
    import urllib.error
    req = urllib.request.Request(
        "https://rest.runpod.io/v1/pods/aitk0sentinel0pod/stop",
        method="POST", headers={"Authorization": f"Bearer {key}"})
    try:
        urllib.request.urlopen(req, timeout=15)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _warn(
                "the pod-side stop key CANNOT stop pods (HTTP 401). "
                "Self-stop will fail-open and a finished pod will idle-bill. "
                "Fix: RunPod console -> Settings -> API Keys -> edit the "
                "RUNPOD_STOP_API_KEY key -> permission 'Restricted' with "
                "Pods: Read & Write (or remove RUNPOD_STOP_API_KEY from .env "
                "to fall back to the account key)."
            )
            return False
        return True  # non-401 (e.g. 'pod does not exist') means auth passed
    except Exception:
        return True  # network hiccup: don't block provisioning on the probe


def container_start_command() -> str:
    """R27: start-command override decoupling container lifetime from the
    image's Node UI keep-alive.

    docker/start.sh runs `set -e` and holds the container open with
    `npm run start`; an unmodified UI exit would kill the container — and the
    tmux server carrying the trainer, wrapper, and timer — while the pod
    keeps billing. Wrapping with `|| true; sleep infinity` keeps the
    container (and tmux) alive regardless of the UI process.
    """
    return "bash -c '/start.sh || true; sleep infinity'"


def self_stop_script() -> str:
    """Shell script content for contract.SELF_STOP_SCRIPT (uploaded by U4,
    invoked by U5's wrapper after the pulled.ok ack or max-grace deadline).

    The image ships no runpodctl (verified against docker/Dockerfile), so
    self-stop curls the RunPod REST stop endpoint, falling back to the
    GraphQL mutation. Authenticates with $RUNPOD_STOP_KEY (injected by
    stop_key_env()) — NEVER the full account key env var. $RUNPOD_POD_ID is
    exported into shell sessions by the image's start.sh.
    """
    return """#!/bin/bash
# Self-stop this RunPod pod. Generated by scripts/remote/pod.py (U3).
# Auth: $RUNPOD_STOP_KEY (restricted key injected at provision time).
set -u
# tmux/ssh shells do NOT inherit the container's env (live-validated failure:
# both backstops failed-open and a completed pod idle-billed for hours).
# Bridge the two needed vars from PID 1's environment when unset.
for var in RUNPOD_POD_ID RUNPOD_STOP_KEY; do
    if [ -z "$(eval echo \\"\\${$var:-}\\")" ]; then
        val=$(tr '\\0' '\\n' < /proc/1/environ | grep "^$var=" | head -1 | cut -d= -f2-)
        [ -n "$val" ] && export "$var=$val"
    fi
done
if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_STOP_KEY:-}" ]; then
    echo "self-stop: RUNPOD_POD_ID or RUNPOD_STOP_KEY missing (checked shell env and /proc/1/environ); cannot stop pod" >&2
    exit 1
fi
# Primary: REST stop endpoint.
if curl -fsS -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \\
        -H "Authorization: Bearer $RUNPOD_STOP_KEY"; then
    echo "self-stop: pod stop requested (REST)"
    exit 0
fi
# Fallback: GraphQL podStop mutation.
curl -fsS "https://api.runpod.io/graphql" \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer $RUNPOD_STOP_KEY" \\
    --data "{\\"query\\": \\"mutation { podStop(input: {podId: \\\\\\"$RUNPOD_POD_ID\\\\\\"}) { id desiredStatus } }\\"}" \\
    && echo "self-stop: pod stop requested (GraphQL)"
"""


def _build_pod_env() -> dict:
    env = {
        "PUBLIC_KEY": read_public_key(),
        # Model cache lands on the volume (contract.disk_size_gb budgets it
        # there), so a pod stop/start never re-downloads weights and the
        # small container disk stays out of the picture.
        "HF_HOME": "/workspace/.hf_home",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        # Provider-visible in pod env; README documents using read-scoped
        # tokens for non-push runs (R21).
        env["HF_TOKEN"] = hf_token
    env.update(stop_key_env())
    return env


def _build_create_request(run_name: str, gpu_type_id: str, image: str,
                          disk_gb: int, env: dict, cloud_type: str) -> dict:
    return dict(
        name=f"aitk-{run_name}",
        image_name=image,
        gpu_type_id=gpu_type_id,
        cloud_type=cloud_type,
        # Direct public-IP SSH is required: RunPod's proxy SSH cannot carry
        # rsync (R6).
        support_public_ip=True,
        start_ssh=True,
        ports="22/tcp",
        volume_in_gb=int(disk_gb),
        volume_mount_path="/workspace",
        container_disk_in_gb=CONTAINER_DISK_GB,
        env=dict(env),
        docker_args=container_start_command(),
    )


def _redacted_request(request: dict) -> dict:
    out = dict(request)
    env = dict(out.get("env") or {})
    for key in env:
        if key in _SECRET_ENV_KEYS:
            env[key] = "<redacted>"
    out["env"] = env
    return out


def _is_out_of_stock(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _OUT_OF_STOCK_MARKERS)


def _looks_gone(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _GONE_MARKERS)


def _pod_info_from_raw(pod_id: str, raw) -> PodInfo:
    """Parse the runpod pod dict the way the API returns it: desiredStatus +
    runtime.ports list with isIpPublic/privatePort 22 → ip + publicPort."""
    if not raw:
        return PodInfo(pod_id=pod_id, status="GONE")
    status = raw.get("desiredStatus") or raw.get("status") or "UNKNOWN"
    ssh_host = None
    ssh_port = None
    runtime = raw.get("runtime") or {}
    for port in (runtime.get("ports") or []):
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            ssh_host = port.get("ip")
            ssh_port = int(port["publicPort"]) if port.get("publicPort") is not None else None
            break
    gpu_type = (raw.get("machine") or {}).get("gpuDisplayName")
    rate = raw.get("costPerHr")
    return PodInfo(
        pod_id=raw.get("id") or pod_id,
        status=status,
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        gpu_type=gpu_type,
        hourly_rate=float(rate) if rate is not None else None,
    )


# ---------------------------------------------------------------------------
# Public lifecycle API
# ---------------------------------------------------------------------------

def create_pod(run_name, *, gpu_type=DEFAULT_GPU, image_tag=None, disk_gb,
               env=None, gpu_fallback=(), cloud_type="SECURE", dry_run=False,
               sdk=None) -> PodInfo:
    """Provision a pod (R6). Out-of-stock fails fast naming the GPU type;
    substitution only via the explicit `gpu_fallback` list (R7) — the
    returned PodInfo.gpu_type reflects what was actually provisioned."""
    contract.validate_run_name(run_name)
    image = image_tag or os.environ.get("AITK_REMOTE_IMAGE") or DEFAULT_IMAGE_TAG
    pod_env = _build_pod_env()
    if env:
        pod_env.update(env)

    # Dedupe the attempt list after alias resolution, preserving order.
    attempts = []
    for name in (gpu_type,) + tuple(gpu_fallback):
        resolved = resolve_gpu_type(name)
        if resolved not in attempts:
            attempts.append(resolved)

    if dry_run:
        request = _build_create_request(run_name, attempts[0], image, disk_gb,
                                        pod_env, cloud_type)
        print("create_pod --dry-run: request that WOULD be sent (no API call):")
        print(json.dumps(_redacted_request(request), indent=2, sort_keys=True))
        if len(attempts) > 1:
            print(f"gpu fallback order: {attempts}")
        return PodInfo(pod_id="DRY_RUN", status="DRY_RUN", gpu_type=attempts[0])

    sdk = sdk or _get_sdk()
    out_of_stock = []
    for gpu_type_id in attempts:
        request = _build_create_request(run_name, gpu_type_id, image, disk_gb,
                                        pod_env, cloud_type)
        try:
            created = _sdk_create_pod(sdk, request)
        except Exception as e:  # noqa: BLE001 — SDK raises plain exceptions
            if _is_out_of_stock(e):
                out_of_stock.append(gpu_type_id)
                _warn(f"out of stock: {gpu_type_id}")
                continue
            raise PodError(f"create_pod failed for {gpu_type_id!r}: {e}") from e
        if not created or not created.get("id"):
            # SDK returned no instance — treat like out-of-stock.
            out_of_stock.append(gpu_type_id)
            _warn(f"no instance returned for: {gpu_type_id}")
            continue

        pod_id = created["id"]
        # Enrich with the live pod record (host/port/rate/provisioned GPU).
        try:
            info = get_pod_info(pod_id, sdk=sdk)
        except PodError:
            info = _pod_info_from_raw(pod_id, created)
        if info.status == "GONE":
            info = _pod_info_from_raw(pod_id, created)
        if info.gpu_type is None:
            info = dataclasses.replace(info, gpu_type=gpu_type_id)
        if info.hourly_rate is None and created.get("costPerHr") is not None:
            info = dataclasses.replace(info, hourly_rate=float(created["costPerHr"]))
        return info

    raise OutOfStockError(
        f"no instances available for GPU type(s) {out_of_stock}; "
        "try again later, pass a different --gpu, or provide --gpu-fallback"
    )


def get_pod_info(pod_id, sdk=None) -> PodInfo:
    """Resolve current pod state. Status 'GONE' when the API 404s or the
    pod is absent — endpoint values are cache-only, re-resolve before SSH."""
    sdk = sdk or _get_sdk()
    try:
        raw = _sdk_get_pod(sdk, pod_id)
    except Exception as e:  # noqa: BLE001
        if _looks_gone(e):
            return PodInfo(pod_id=pod_id, status="GONE")
        raise PodError(f"get_pod failed for {pod_id!r}: {e}") from e
    return _pod_info_from_raw(pod_id, raw)


def default_ssh_probe(host: str, port: int) -> bool:
    """Readiness probe: SSH reachable AND rsync present over the direct path.

    `ssh ... rsync --version` proves sshd is up, the injected key works, and
    the rsync binary exists on the pod — i.e. the direct path can actually
    carry rsync (RunPod's proxy SSH cannot). Kept separate so tests and
    callers can inject a fake probe.
    """
    cmd = [
        "ssh",
        "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={contract.SSH_KNOWN_HOSTS_FILE}",
        *contract.ssh_identity_args(),
        "-o", "ConnectTimeout=10",
        f"root@{host}",
        "rsync --version",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def wait_for_ready(pod_id, *, timeout_s=2400, poll_s=15, probe=None, sdk=None,
                   _sleep=time.sleep) -> PodInfo:
    """Poll until the pod exposes direct SSH and passes the rsync probe.

    On timeout the half-provisioned pod is terminated before raising
    SshTimeoutError — never leave a pod billing that nobody can reach (R6).
    Elapsed time is accumulated from poll_s (not wall clock) so tests can
    inject a no-op _sleep and stay deterministic.
    """
    probe = probe or default_ssh_probe
    elapsed = 0.0
    last_status = None
    while True:
        info = get_pod_info(pod_id, sdk=sdk)
        last_status = info.status
        if info.status == "GONE":
            raise PodError(
                f"pod {pod_id} disappeared while waiting for readiness "
                "(terminated externally?)"
            )
        if info.ssh_host and info.ssh_port:
            if probe(info.ssh_host, info.ssh_port):
                return info
        if elapsed >= timeout_s:
            break
        _sleep(poll_s)
        elapsed += poll_s

    try:
        terminate_pod(pod_id, sdk=sdk)
        cleanup = "pod terminated"
    except Exception as e:  # noqa: BLE001 — cleanup must not mask the timeout
        cleanup = f"CLEANUP FAILED ({e}) — terminate pod {pod_id} manually"
    raise SshTimeoutError(
        f"pod {pod_id} not SSH-ready within {timeout_s}s "
        f"(last status: {last_status}; {cleanup})"
    )


def start_zero_gpu(pod_id, sdk=None) -> PodInfo:
    """Rescue path (R24): start a stopped pod with gpu_count=0 to pull
    artifacts off its volume without paying for a GPU."""
    sdk = sdk or _get_sdk()
    try:
        _sdk_resume_pod(sdk, pod_id, gpu_count=0)
    except Exception as e:  # noqa: BLE001
        raise PodError(f"resume_pod (gpu_count=0) failed for {pod_id!r}: {e}") from e
    return get_pod_info(pod_id, sdk=sdk)


def stop_pod(pod_id, sdk=None) -> None:
    sdk = sdk or _get_sdk()
    try:
        _sdk_stop_pod(sdk, pod_id)
    except Exception as e:  # noqa: BLE001
        raise PodError(f"stop_pod failed for {pod_id!r}: {e}") from e


def terminate_pod(pod_id, sdk=None) -> None:
    """Idempotent: terminating an already-gone pod succeeds quietly (R9
    `down` and timeout cleanup both rely on this)."""
    sdk = sdk or _get_sdk()
    try:
        _sdk_terminate_pod(sdk, pod_id)
    except Exception as e:  # noqa: BLE001
        if _looks_gone(e):
            return
        raise PodError(f"terminate_pod failed for {pod_id!r}: {e}") from e
