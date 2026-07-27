"""Local (never global) Git provisioning for Windows into <repo>/.mingit/.

MinGit is the official minimal, portable Git for Windows — a plain zip meant
for embedding in tools (no installer, no registry, no PATH changes). On
Windows, if the system has no git, `manager sync` drops one here so updates
keep working from any terminal. The first clone (before this repo exists) is
handled the same way by the bootstrap layer (install.ps1 / desktop launcher),
which then moves its MinGit into the fresh checkout as .mingit/.

Linux and macOS have no sane portable git (glibc / Xcode CLT entanglement),
and git is effectively always available there — so this is Windows-only and
the manager just errors with install instructions elsewhere.
"""

import os
import shutil
import tempfile

from .util import IS_WINDOWS, REPO_ROOT, download, extract_archive, info, ok, warn

MINGIT_DIR = os.path.join(REPO_ROOT, ".mingit")
# Update this pin together with nothing else — it's independent of torch etc.
MINGIT_URL = (
    "https://github.com/git-for-windows/git/releases/download/"
    "v2.55.0.windows.3/MinGit-2.55.0.3-64-bit.zip"
)


def local_git_exe():
    return os.path.join(MINGIT_DIR, "cmd", "git.exe")


def find_git():
    """Path/command for git: repo-local MinGit first (Windows), then system."""
    if IS_WINDOWS and os.path.isfile(local_git_exe()):
        return local_git_exe()
    return shutil.which("git")


def ensure_git(dry_run=False):
    """Windows-only: provision .mingit/ when the system has no git."""
    if not IS_WINDOWS:
        return False
    if find_git():
        return False
    if dry_run:
        info("[dry-run] would install MinGit into %s" % MINGIT_DIR)
        return False
    tmp = tempfile.mkdtemp(prefix="aitk_mingit_")
    try:
        archive = os.path.join(tmp, "mingit.zip")
        download(MINGIT_URL, archive, label="MinGit")
        # MinGit zips have no top-level folder: cmd/, mingw64/, etc at the root
        extracted = os.path.join(tmp, "mingit")
        extract_archive(archive, extracted)
        if not os.path.isfile(os.path.join(extracted, "cmd", "git.exe")):
            warn("Unexpected MinGit archive layout — skipping local git install.")
            return False
        if os.path.isdir(MINGIT_DIR):
            shutil.rmtree(MINGIT_DIR)
        shutil.move(extracted, MINGIT_DIR)
        ok("Local Git (MinGit) installed at %s" % MINGIT_DIR)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
