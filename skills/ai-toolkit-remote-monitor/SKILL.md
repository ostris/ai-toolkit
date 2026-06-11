---
name: ai-toolkit-remote-monitor
description: >
  Follow a running AI Toolkit training run on a RunPod GPU and drive the
  mid-run checkpoint-review loop from the Mac CLI. Use when: (1) a remote run
  is already launched and the user says "check on my run", "how's training
  going", "watch the balfua run", "any new samples", "is it done yet",
  (2) a fresh Claude session needs to re-attach to a run started earlier
  ("attach to <run>"), (3) samples have come back and it's time to pick a
  checkpoint mid-run. Drives scripts/remote/cli.py watch/status/pull/attach/
  mark-reviewed in the ai-toolkit repo, pulls samples+checkpoints into
  output/<run>/, and invokes ai-toolkit-sample-reviewer on each new batch.
  For STARTING a run use ai-toolkit-remote-launch; for shutting down use
  ai-toolkit-remote-teardown.
---

# AI Toolkit remote monitor (RunPod)

Follow a live run and drive the review loop. Everything runs from the Mac
CLI via `scripts/remote/cli.py` in `~/repos/ai-toolkit`. This skill assumes
the run is already launched (see `ai-toolkit-remote-launch`).

The whole pipeline is re-entrant from `runs/<run>/manifest.json` + the
RunPod API, so a **fresh session with zero context** can pick up any run by
name — start with `attach`.

## Re-attach first (fresh session)

```bash
cd ~/repos/ai-toolkit
python scripts/remote/cli.py attach <run>     # rebuild watcher state from manifest + API
```

Don't know the run name? It's the `config.name` from the YAML, and lives at
`runs/<run>/manifest.json` — `ls runs/` lists them.

## The polling pattern (this is the core loop)

Do **not** hold a blocking `watch` open across turns. Poll on a schedule
with `--once --json` — one cycle, machine-readable, distinct exit code:

```bash
python scripts/remote/cli.py watch <run> --once --json
```

Returns one JSON object and an exit code:

| exit | meaning | what to do |
|---|---|---|
| 0 | COMPLETED | final pull + teardown (teardown skill) |
| 3 | running, nothing new | sleep, poll again later |
| 10 | new reviewable sample steps | review them (below), then `mark-reviewed` |
| 20 | CRASHED | read `detail` / `log_tail_path`; pod left up for diagnosis |
| 21 | STOPPED | user/early stop; pull + down |
| 22 | TIMED_OUT | max-runtime timer fired |
| 23 | UNKNOWN | pod up but tmux/sentinel gone (container restart) — investigate |
| 24 | POD_LOST | pod gone from the API |

The JSON carries: `state`, `step`, `total_steps`, `loss`, `oom_skips`,
`disk_used_pct`, `drift`, `reviewable_steps`, `last_reviewed_step`,
`cost_estimate`, `pulled_checkpoint_steps`, `detail`, `log_tail_path`.

`watch --once` also **pulls** new samples/checkpoints into `output/<run>/`
each cycle, so the artifacts are local by the time you see exit 10.

**Cadence:** every ~10 min for an active run. First runs spend 20-60 min in
`warming` (model download) before step 1 — that's `state: RUNNING`,
`step: null`. Sample/checkpoint cadence follows the config's
`sample_every`/`save_every` (usually 250).

### Self-paced loop (recommended for long runs)

Use `/loop` (or re-invoke yourself on a schedule) to call `watch --once
--json`, branch on the exit code, and only surface to the user on a state
change or a new reviewable batch. Don't spam status when nothing changed
(exit 3).

## Reviewing samples mid-run

On exit 10 (new reviewable steps), the samples are already in
`output/<run>/samples/`. Hand them to **`ai-toolkit-sample-reviewer`** —
**once per new-steps batch, not one call per step** (serial montage review;
parallel per-step image-review subagents are a known crash mode on the
laptop). After reviewing, advance the watermark so the same steps don't
re-report forever:

```bash
python scripts/remote/cli.py mark-reviewed <run> <highest-step-just-reviewed>
```

Remember **LoRA checkpoint trajectories are non-monotonic** — review 3+ late
checkpoints, never auto-assume the last save is best. `pulled_checkpoint_steps`
in the JSON tells you which are local to compare.

## Health signals to watch

- **DEGRADED** (`oom_skips > 0`, state still RUNNING) — recurring OOM-skips.
  The trainer aborts on 3 *consecutive*; scattered skips just reduce
  effective coverage. Surface it; the run continues.
- **disk_warning / `disk_used_pct` ≥ 85** — the keep-all checkpoint policy
  can fill `/workspace`. Flag it before a save dies.
- **drift: true** — the local config was edited after launch; the running
  pod still trains the original. The new config only applies on next launch.
- **SAMPLING** — step stall with fresh sample-file mtimes is normal (12-16
  images × ~45s), not a hang.

## Stopping early

If the reviewer's verdict is "overcooked / a mid checkpoint is the winner",
stop the run (it preserves everything through the last save):

```bash
python scripts/remote/cli.py stop <run>     # kills the trainer, sentinel/self-stop still run
```

`stop` does NOT pick a checkpoint — it just ends training; you choose the
checkpoint from what's already pulled. After stopping, go to
`ai-toolkit-remote-teardown` for the verified final pull + terminate.

## When it reaches a terminal state

- **COMPLETED (exit 0)** — hand off to `ai-toolkit-remote-teardown` for the
  final pull, verify, terminate, cost report.
- **CRASHED (exit 20)** — read `detail` and `runs/<run>/log_tail.txt`
  (`log_tail_path`). The pod is left up, **billing full GPU rate**, so
  decide immediately which mode you're in:
  - *Diagnosing right now, in this session* (and the fix might need the
    pod — e.g. inspecting the remote filesystem): leave it up, diagnose,
    tear down the moment you're done. `ai-toolkit-dataset-diagnostics`
    maps most trainer startup failures.
  - *Anything else* — the user is stepping away, the session is ending, or
    this is a first-time trainer: pull what's needed (the log tail is
    already local; `pull` grabs any artifacts), **tear down first, diagnose
    from the saved log**. Almost every crash is diagnosable from
    `log_tail.txt` alone, and a fix means a fresh launch anyway. Never park
    a crashed pod "to look at later" — that's the most expensive way to
    store a text file.
- **POD_LOST / TIMED_OUT** — the pod may be stopped (not gone); the teardown
  skill's `rescue` retrieves artifacts from a stopped pod.

## Common mistakes

- **Holding `watch` open across turns** — use `--once --json` and re-poll;
  a blocking watch wastes the turn and dies on the first transient SSH blip
  in older builds (current build tolerates it but the polling pattern is
  still the contract).
- **Reviewing every step in parallel** — montage, serial, one batch at a
  time.
- **Forgetting `mark-reviewed`** — without it, `watch --once` returns exit
  10 for the same steps forever.
- **Trusting the last checkpoint** — non-monotonic; review several.
- **Parking a crashed pod** — CRASHED leaves the pod up at full GPU rate;
  if diagnosis isn't happening right now, tear down and diagnose from the
  saved log tail.
- **Re-resolving SSH from the manifest** — never needed; every CLI command
  re-resolves the endpoint via the API (RunPod recycles IP:port).

## Related skills

- `ai-toolkit-remote-launch` — start a run.
- `ai-toolkit-remote-teardown` — `down`/`rescue`, cost, orphan-pod scan.
- `ai-toolkit-sample-reviewer` — the actual checkpoint-picking review.
- `ai-toolkit-dataset-diagnostics` — when a run CRASHED at startup.

Full reference: `scripts/remote/README.md` in the ai-toolkit repo.
