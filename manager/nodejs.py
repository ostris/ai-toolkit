"""Local (never global) Node.js provisioning into <repo>/.node/.

A system Node >= 20 is used when present; otherwise an official portable
build is downloaded next to the repo (same approach run_mac.zsh already uses).
Nothing is ever installed system-wide.
"""

import os
import shutil
import subprocess
import tempfile

from .util import (
    IS_MAC,
    IS_WINDOWS,
    REPO_ROOT,
    clean_env,
    download,
    extract_archive,
    file_hash,
    info,
    ok,
    run,
    warn,
    which,
)

NODE_DIR = os.path.join(REPO_ROOT, ".node")
# Node 24 is the current LTS line and matches the dgx_instructions.md guidance.
NODE_VERSION = "24.11.1"
MIN_NODE_MAJOR = 20

UI_DIR = os.path.join(REPO_ROOT, "ui")
UI_STATE_KEY = "ui_deps_hash"


def node_bin_dir():
    # windows zips have node.exe/npm.cmd at the archive root; unix under bin/
    return NODE_DIR if IS_WINDOWS else os.path.join(NODE_DIR, "bin")


def local_node_exe():
    return os.path.join(node_bin_dir(), "node.exe" if IS_WINDOWS else "node")


def _node_major(exe):
    try:
        out = subprocess.run(
            [exe, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        text = out.stdout.decode().strip()  # v24.11.1
        return int(text.lstrip("v").split(".")[0])
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _dist_url(detection):
    arch = detection["arch"]
    if detection["os"] == "linux":
        plat = "linux-arm64" if arch == "aarch64" else "linux-x64"
        ext = "tar.xz"
    elif detection["os"] == "mac":
        plat = "darwin-arm64" if arch == "arm64" else "darwin-x64"
        ext = "tar.gz"
    elif detection["os"] == "windows":
        plat = "win-x64"
        ext = "zip"
    else:
        return None, None
    name = "node-v%s-%s" % (NODE_VERSION, plat)
    return "https://nodejs.org/dist/v%s/%s.%s" % (NODE_VERSION, name, ext), name


def have_usable_node():
    """(exe, major) for the best available node: local .node/ first, then system."""
    local = local_node_exe()
    if os.path.isfile(local):
        major = _node_major(local)
        if major is not None and major >= MIN_NODE_MAJOR:
            return local, major
    system = which("node")
    if system:
        major = _node_major(system)
        if major is not None and major >= MIN_NODE_MAJOR:
            return system, major
    return None, None


def ensure_node(detection, dry_run=False):
    exe, major = have_usable_node()
    if exe:
        ok("Node.js v%d found (%s)." % (major, exe))
        return False
    url, inner_name = _dist_url(detection)
    if url is None:
        warn(
            "No portable Node.js build for this platform — install Node >= %d manually."
            % MIN_NODE_MAJOR
        )
        return False
    if dry_run:
        info(
            "[dry-run] would install portable Node.js v%s into %s"
            % (NODE_VERSION, NODE_DIR)
        )
        return False
    tmp = tempfile.mkdtemp(prefix="aitk_node_")
    try:
        archive = os.path.join(tmp, os.path.basename(url))
        download(url, archive, label="node v%s" % NODE_VERSION)
        extract_archive(archive, tmp)
        inner = os.path.join(tmp, inner_name)
        if not os.path.isdir(inner):
            warn("Unexpected Node.js archive layout — skipping node install.")
            return False
        if os.path.isdir(NODE_DIR):
            shutil.rmtree(NODE_DIR)
        shutil.move(inner, NODE_DIR)
        ok("Portable Node.js v%s installed at %s" % (NODE_VERSION, NODE_DIR))
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- ui deps


def npm_env():
    """Scrubbed env with the local .node/ ahead of PATH."""
    env = clean_env()
    if os.path.isdir(node_bin_dir()):
        env["PATH"] = node_bin_dir() + os.pathsep + env.get("PATH", "")
    return env


def find_npm(env=None):
    """npm from the local .node/ copy when we installed one, else the system."""
    path = (env or npm_env()).get("PATH")
    return shutil.which("npm.cmd" if IS_WINDOWS else "npm", path=path)


def _ui_deps_hash():
    return file_hash(
        [
            os.path.join(UI_DIR, "package-lock.json"),
            os.path.join(UI_DIR, "package.json"),
        ]
    )


def ensure_ui_deps(env=None, dry_run=False):
    """Install ui/node_modules without ever rewriting ui/package-lock.json.

    A plain `npm install` treats the lockfile as writable and re-derives it for
    the machine doing the install: on Windows it strips the `libc` fields off
    the Linux-only optional binaries (@next/swc-linux-*, rollup, lightningcss),
    on Linux it puts them back. Since the UI installs on every launch, everyone
    ends up with a modified lockfile they never asked for — which then blocks
    `manager update`, because a dirty tree aborts the pull.

    `--no-save` keeps the resolution but stops the write-back, and stays an
    incremental install (seconds) rather than the full wipe-and-refetch of
    `npm ci`. The lockfile is snapshotted and restored regardless, so this
    holds even if a future npm changes what `--no-save` covers.

    Gated on a hash of the manifests so steady-state launches skip npm.
    """
    from . import env as env_mod

    lock = os.path.join(UI_DIR, "package-lock.json")
    if not os.path.isfile(lock):
        warn("ui/package-lock.json is missing — skipping UI dependency install.")
        return False
    want = _ui_deps_hash()
    if env_mod.venv_exists() and env_mod.load_state().get(UI_STATE_KEY) == want:
        if os.path.isdir(os.path.join(UI_DIR, "node_modules")):
            ok("UI dependencies already installed.")
            return False
    if dry_run:
        info("[dry-run] would run: npm install --no-save (in %s)" % UI_DIR)
        return False
    env = env or npm_env()
    npm = find_npm(env)
    if npm is None:
        warn("npm not found — skipping UI dependency install.")
        return False

    info("Installing UI dependencies...")
    with open(lock, "rb") as f:
        before = f.read()
    code, _ = run(
        [npm, "install", "--no-save", "--no-audit", "--no-fund"],
        cwd=UI_DIR,
        env=env,
        stream=True,
        check=False,
    )
    with open(lock, "rb") as f:
        after = f.read()
    if after != before:
        with open(lock, "wb") as f:
            f.write(before)
        info("Reverted npm's edit to ui/package-lock.json (managed by the repo).")
    if code != 0:
        warn("UI dependency install failed — the UI may not start.")
        return False
    if env_mod.venv_exists():
        state = env_mod.load_state()
        state[UI_STATE_KEY] = want
        env_mod.save_state(state)
    ok("UI dependencies ready.")
    return True
