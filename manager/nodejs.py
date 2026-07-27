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
    download,
    extract_archive,
    info,
    ok,
    warn,
    which,
)

NODE_DIR = os.path.join(REPO_ROOT, ".node")
# Node 24 is the current LTS line and matches the dgx_instructions.md guidance.
NODE_VERSION = "24.11.1"
MIN_NODE_MAJOR = 20


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
