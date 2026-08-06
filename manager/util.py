"""Shared helpers for the AI Toolkit manager. Stdlib only."""

import json
import os
import platform
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# When --json is used, human output goes to stderr so stdout stays machine-readable
_json_mode = False


def set_json_mode(enabled):
    global _json_mode
    _json_mode = enabled


def _supports_color(stream):
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def _emit(prefix, msg, color):
    stream = sys.stderr if _json_mode else sys.stdout
    if _supports_color(stream):
        stream.write("\033[%sm%s\033[0m %s\n" % (color, prefix, msg))
    else:
        stream.write("%s %s\n" % (prefix, msg))
    stream.flush()


def info(msg):
    _emit("[*]", msg, "36")


def ok(msg):
    _emit("[+]", msg, "32")


def warn(msg):
    _emit("[!]", msg, "33")


def error(msg):
    _emit("[x]", msg, "31")


def die(msg, code=1):
    error(msg)
    sys.exit(code)


def print_json(data):
    sys.stdout.write(json.dumps(data, indent=2) + "\n")
    sys.stdout.flush()


def run(cmd, cwd=None, capture=False, check=True, env=None, stream=False):
    """Run a command. capture=True returns stdout text (stripped).

    stream=True inherits stdio so the user sees live output.
    Returns (returncode, stdout_or_None).
    """
    kwargs = {"cwd": cwd or REPO_ROOT}
    if env is not None:
        kwargs["env"] = env
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    elif _json_mode and not stream:
        # keep stdout clean in json mode
        kwargs["stdout"] = sys.stderr

    try:
        proc = subprocess.run(cmd, **kwargs)
    except FileNotFoundError:
        if check:
            die("Command not found: %s" % cmd[0])
        return 127, None

    out = None
    if capture:
        out = proc.stdout.decode("utf-8", errors="replace").strip()
    if check and proc.returncode != 0:
        detail = ""
        if capture and proc.stderr:
            detail = "\n" + proc.stderr.decode("utf-8", errors="replace").strip()
        die("Command failed (%d): %s%s" % (proc.returncode, " ".join(cmd), detail))
    return proc.returncode, out


def which(name):
    return shutil.which(name)


def find_uv():
    """Find uv: repo-local .uv/ first, then PATH, then common install dirs."""
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(REPO_ROOT, ".uv", "uv.exe" if IS_WINDOWS else "uv"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    uv = shutil.which("uv")
    if uv:
        return uv
    candidates = [
        os.path.join(home, ".local", "bin", "uv"),
        os.path.join(home, ".cargo", "bin", "uv"),
    ]
    if IS_WINDOWS:
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            candidates.append(os.path.join(local, "uv", "uv.exe"))
        candidates.append(os.path.join(home, ".local", "bin", "uv.exe"))
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def venv_dir():
    """Existing venv dir (.venv preferred, matching ui/cron/pythonPath.ts), else default target."""
    for name in (".venv", "venv"):
        d = os.path.join(REPO_ROOT, name)
        if os.path.isdir(d):
            return d
    return os.path.join(REPO_ROOT, ".venv")


def venv_python(venv=None):
    venv = venv or venv_dir()
    if IS_WINDOWS:
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python3")


# Env vars that let a system/conda/pyenv Python leak into our subprocesses.
# Scrubbed from every python/pip/node invocation (mirrors what the community
# Windows installer learned the hard way).
_SCRUB_VARS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHON",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONEXECUTABLE",
    "PIP_CONFIG_FILE",
    "PIP_REQUIRE_VIRTUALENV",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYENV_ROOT",
    "PYENV_VERSION",
)


def clean_env(extra=None):
    """os.environ copy with Python-hijacking vars removed.

    Also points uv's managed-python store into the repo (.uv/python) so
    interpreter downloads never land outside the checkout.
    """
    env = os.environ.copy()
    for var in _SCRUB_VARS:
        env.pop(var, None)
    env.setdefault("UV_PYTHON_INSTALL_DIR", os.path.join(REPO_ROOT, ".uv", "python"))
    if extra:
        env.update(extra)
    return env


def download(url, dest, label=None):
    """Download url to dest (stdlib only), logging progress every ~10%.

    Sends a browser User-Agent: some mirrors (ffmpeg.martin-riedl.de) return
    403 for the default "Python-urllib/x.y" agent.
    """
    import urllib.request

    label = label or os.path.basename(dest)
    info("Downloading %s ..." % label)
    last = [-1]

    def report(read, total):
        if total <= 0:
            return
        pct = min(100, int(read * 100 / total))
        if pct >= last[0] + 10:
            last[0] = pct
            info("  %s: %d%%" % (label, pct))

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            read = 0
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
                read += len(chunk)
                report(read, total)
    except Exception as e:  # noqa: BLE001 - surface any network failure clearly
        if os.path.exists(tmp):
            os.remove(tmp)
        # Windows: python's ssl verifies against the OS cert store but never
        # triggers Windows' on-demand intermediate-CA fetching, so on a fresh
        # machine github downloads can fail CERTIFICATE_VERIFY_FAILED even
        # though the chain is fine. curl (bundled since Win10, schannel-based)
        # does fetch intermediates — fall back to it before giving up.
        if not _download_with_curl(url, tmp, label):
            die("Download failed for %s: %s" % (url, e))
    os.replace(tmp, dest)


def _download_with_curl(url, tmp, label):
    curl = shutil.which("curl")
    if not curl:
        return False
    info("  %s: retrying with curl..." % label)
    try:
        code = subprocess.call(
            [curl, "-fSL", "--retry", "3", "-o", tmp, url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    if code != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    return True


def extract_archive(archive, dest_dir):
    """Extract .zip / .tar.* into dest_dir (created if needed)."""
    import tarfile
    import zipfile

    os.makedirs(dest_dir, exist_ok=True)
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest_dir)


def file_hash(paths):
    import hashlib

    h = hashlib.sha256()
    for p in sorted(paths):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()
