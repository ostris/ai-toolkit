"""Local (never system) uv provisioning into <repo>/.uv/.

uv is a single static binary (no deps, no Python needed). Keeping it inside
the repo means fast installs + exact-version Python provisioning with zero
footprint outside the checkout. The run scripts also install into this same
dir (via the astral installer with UV_INSTALL_DIR) for the no-python
bootstrap case; util.find_uv() checks here first either way.
"""

import os
import platform
import shutil
import stat
import tempfile

from .util import IS_MAC, IS_WINDOWS, REPO_ROOT, download, extract_archive, ok, warn

UV_DIR = os.path.join(REPO_ROOT, ".uv")

_RELEASE = "https://github.com/astral-sh/uv/releases/latest/download/"


def _asset():
    arch = platform.machine().lower()
    if arch == "amd64":
        arch = "x86_64"
    if IS_WINDOWS:
        return "uv-x86_64-pc-windows-msvc.zip"
    if IS_MAC:
        return "uv-%s-apple-darwin.tar.gz" % (
            "aarch64" if arch == "arm64" else "x86_64"
        )
    return "uv-%s-unknown-linux-gnu.tar.gz" % (
        "aarch64" if arch in ("aarch64", "arm64") else "x86_64"
    )


def local_uv_exe():
    return os.path.join(UV_DIR, "uv.exe" if IS_WINDOWS else "uv")


def ensure_uv(dry_run=False):
    """Download the uv binary into .uv/ if it isn't available anywhere."""
    from .util import find_uv

    if find_uv():
        return False
    if dry_run:
        from .util import info

        info("[dry-run] would download uv into %s" % UV_DIR)
        return False
    tmp = tempfile.mkdtemp(prefix="aitk_uv_")
    try:
        asset = _asset()
        archive = os.path.join(tmp, asset)
        download(_RELEASE + asset, archive, label="uv")
        extract_archive(archive, tmp)
        # zip: uv.exe at root; tarballs: uv-<triple>/uv — search the tree
        exe_name = "uv.exe" if IS_WINDOWS else "uv"
        found = None
        for root, _dirs, files in os.walk(tmp):
            if exe_name in files:
                found = os.path.join(root, exe_name)
                break
        if not found:
            warn("Unexpected uv archive layout — continuing without uv.")
            return False
        os.makedirs(UV_DIR, exist_ok=True)
        dest = local_uv_exe()
        shutil.move(found, dest)
        os.chmod(
            dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        ok("uv installed at %s" % dest)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
