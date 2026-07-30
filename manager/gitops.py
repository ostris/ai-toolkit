"""Git operations for update checking and pulling. Stdlib only."""

import os

from . import gitwin
from .util import run, REPO_ROOT, die


def _git(args, capture=True, check=True):
    git = gitwin.find_git()
    if not git:
        die(
            "git was not found. On Windows run `python -m manager sync` to "
            "install a local copy; elsewhere install git with your package "
            "manager (or xcode-select --install on macOS)."
        )
    # skip LFS payloads — the repo may carry LFS files not needed at runtime
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    return run([git] + args, cwd=REPO_ROOT, capture=capture, check=check, env=env)


def current_branch():
    _, out = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    return out


def current_commit():
    _, out = _git(["rev-parse", "--short", "HEAD"])
    return out


def is_dirty():
    _, out = _git(["status", "--porcelain"])
    # untracked files don't block updates; modified/staged tracked files do
    for line in (out or "").splitlines():
        if not line.startswith("??"):
            return True
    return False


def fetch():
    code, _ = _git(["fetch", "--quiet"], capture=False, check=False)
    return code == 0


def upstream():
    code, out = _git(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], check=False
    )
    if code != 0:
        return None
    return out


def behind_count():
    """Commits the local branch is behind its upstream. None if no upstream."""
    up = upstream()
    if not up:
        return None
    code, out = _git(["rev-list", "--count", "HEAD..@{u}"], check=False)
    if code != 0 or out is None:
        return None
    try:
        return int(out)
    except ValueError:
        return None


def remote_commit():
    code, out = _git(["rev-parse", "--short", "@{u}"], check=False)
    return out if code == 0 else None


def incoming_log(limit=15):
    code, out = _git(
        ["log", "--oneline", "HEAD..@{u}", "--max-count=%d" % limit], check=False
    )
    if code != 0 or not out:
        return []
    return out.splitlines()


def pull_ff():
    """Fast-forward pull. Dies with guidance on failure."""
    code, _ = _git(["pull", "--ff-only"], capture=False, check=False)
    if code != 0:
        die(
            "git pull --ff-only failed. Your branch has local commits or "
            "conflicts with the remote. Resolve manually (git stash / git rebase) "
            "and re-run the update."
        )
