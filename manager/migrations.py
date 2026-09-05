"""One-time migration steps that run after an update.

Add a migration when an update needs more than a dependency sync (moving
files, converting configs, clearing caches, etc). Each runs at most once per
environment; applied ids are recorded in the venv state file.

    def _example(dry_run):
        ...

    MIGRATIONS = [
        {"id": "2026-07-example-cache-move", "run": _example},
    ]
"""

from .util import info
from . import env

MIGRATIONS = []


def run_pending(dry_run=False):
    if not MIGRATIONS:
        return
    state = env.load_state()
    applied = set(state.get("migrations", []))
    for migration in MIGRATIONS:
        if migration["id"] in applied:
            continue
        info("Running migration: %s" % migration["id"])
        if not dry_run:
            migration["run"](dry_run)
            applied.add(migration["id"])
            state["migrations"] = sorted(applied)
            env.save_state(state)
