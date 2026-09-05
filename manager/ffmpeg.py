"""Local (never global) FFmpeg provisioning into <repo>/.ffmpeg/.

Why: the toolkit shells out to ffmpeg/ffprobe for video work, and torchcodec
dlopens the FFmpeg *shared libraries* at runtime. Installing FFmpeg system-wide
(apt/winget/brew) is exactly what we want to avoid, so we download a portable
build next to the repo:

- Linux / Windows: BtbN shared builds (bin/ + lib/ with .so/.dll) — the shared
  libs are what torchcodec needs. The FFmpeg major version must be one the
  pinned torchcodec supports.
- macOS: Martin Riedl static ffmpeg/ffprobe executables (no shared libs
  published; torchcodec on mac keeps whatever it uses today).

Exposure to the rest of the system:
- `manager launch` prepends .ffmpeg/bin to PATH (and .ffmpeg/lib to
  LD_LIBRARY_PATH on Linux) so the UI and every training job it spawns see it.
- env.py writes a sitecustomize.py into the venv that prepends .ffmpeg/bin to
  PATH and (on Windows) calls os.add_dll_directory — so torchcodec finds the
  DLLs in ANY use of the venv python, not just via `manager launch`.
"""

import os
import shutil
import stat
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
)

FFMPEG_DIR = os.path.join(REPO_ROOT, ".ffmpeg")

# FFmpeg 8 on all BtbN platforms — torchcodec 0.15 (pinned in spec.py)
# supports ffmpeg up to 8. Bump these together with the torchcodec pin.
_BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
_RIEDL = (
    "https://ffmpeg.martin-riedl.de/redirect/latest/macos/{arch}/release/{tool}.zip"
)

_SOURCES = {
    ("linux", "x86_64"): _BTBN + "ffmpeg-n8.1-latest-linux64-gpl-shared-8.1.tar.xz",
    ("linux", "aarch64"): _BTBN + "ffmpeg-n8.1-latest-linuxarm64-gpl-shared-8.1.tar.xz",
    ("windows", "x86_64"): _BTBN + "ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip",
    # Windows-on-ARM in the emulated-x64 stack gets the x64 build, NOT BtbN's
    # winarm64 one: an x64 torchcodec can only dlopen x64 FFmpeg DLLs, and the
    # exes run fine under emulation.
    ("windows", "aarch64"): _BTBN + "ffmpeg-n8.1-latest-win64-gpl-shared-8.1.zip",
}

# Native Spark stack: the self-built win_arm64 torchcodec is linked against
# (and dlopens) arm64 FFmpeg 8 — LGPL to keep the distributed torchcodec wheel
# clean, matching C:\Dev spark build scripts / the wheel-set build recipe.
_SPARK_NATIVE_SOURCE = _BTBN + "ffmpeg-n8.1-latest-winarm64-lgpl-shared-8.1.zip"


def bin_dir():
    return os.path.join(FFMPEG_DIR, "bin")


def lib_dir():
    return os.path.join(FFMPEG_DIR, "lib")


def ffmpeg_exe():
    return os.path.join(bin_dir(), "ffmpeg.exe" if IS_WINDOWS else "ffmpeg")


def is_installed(source_url):
    marker = os.path.join(FFMPEG_DIR, ".source")
    if not os.path.isfile(ffmpeg_exe()) or not os.path.isfile(marker):
        return False
    with open(marker) as f:
        return f.read().strip() == source_url


def _mark_installed(source_url):
    with open(os.path.join(FFMPEG_DIR, ".source"), "w") as f:
        f.write(source_url)


def _install_btbn(url):
    tmp = tempfile.mkdtemp(prefix="aitk_ffmpeg_")
    try:
        archive = os.path.join(tmp, os.path.basename(url))
        download(url, archive, label="ffmpeg")
        extract_archive(archive, tmp)
        # archives contain a single top-level dir with bin/ lib/ include/
        inner = None
        for name in os.listdir(tmp):
            path = os.path.join(tmp, name)
            if os.path.isdir(path) and os.path.isdir(os.path.join(path, "bin")):
                inner = path
                break
        if inner is None:
            warn("Unexpected ffmpeg archive layout — skipping ffmpeg install.")
            return False
        if os.path.isdir(FFMPEG_DIR):
            shutil.rmtree(FFMPEG_DIR)
        shutil.move(inner, FFMPEG_DIR)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _install_mac(detection):
    arch = "arm64" if detection["arch"] == "arm64" else "amd64"
    tmp = tempfile.mkdtemp(prefix="aitk_ffmpeg_")
    try:
        os.makedirs(bin_dir(), exist_ok=True)
        for tool in ("ffmpeg", "ffprobe"):
            url = _RIEDL.format(arch=arch, tool=tool)
            archive = os.path.join(tmp, tool + ".zip")
            download(url, archive, label=tool)
            extract_archive(archive, tmp)
            src = os.path.join(tmp, tool)
            if not os.path.isfile(src):
                warn("Unexpected %s archive layout — skipping." % tool)
                return False
            dest = os.path.join(bin_dir(), tool)
            shutil.move(src, dest)
            os.chmod(
                dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def source_url(detection, spec=None):
    if detection["os"] == "mac":
        arch = "arm64" if detection["arch"] == "arm64" else "amd64"
        return _RIEDL.format(arch=arch, tool="ffmpeg")
    if (
        getattr(spec, "backend", None) == "cu134"
        and (detection["os"], detection["arch"]) == ("windows", "aarch64")
    ):
        return _SPARK_NATIVE_SOURCE
    return _SOURCES.get((detection["os"], detection["arch"]))


def ensure_ffmpeg(detection, dry_run=False, spec=None):
    url = source_url(detection, spec=spec)
    if url is None:
        warn(
            "No portable FFmpeg source for %s/%s — skipping local ffmpeg."
            % (detection["os"], detection["arch"])
        )
        return False
    if is_installed(url):
        ok("Local FFmpeg already installed (.ffmpeg/).")
        return False
    if dry_run:
        info("[dry-run] would install local FFmpeg from %s into %s" % (url, FFMPEG_DIR))
        return False
    installed = (
        _install_mac(detection) if detection["os"] == "mac" else _install_btbn(url)
    )
    if installed:
        _mark_installed(url)
        ok("Local FFmpeg installed at %s" % FFMPEG_DIR)
    return installed


def env_additions():
    """(path_dirs, ld_library_dirs) to prepend when launching anything."""
    paths = []
    lib_paths = []
    if os.path.isdir(bin_dir()):
        paths.append(bin_dir())
    if not IS_WINDOWS and not IS_MAC and os.path.isdir(lib_dir()):
        lib_paths.append(lib_dir())
    return paths, lib_paths
