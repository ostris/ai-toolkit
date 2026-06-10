---
name: ai-toolkit-remote-teardown
description: >
  Tear down a RunPod training pod safely and confirm nothing is still
  billing. Use when: (1) a run finished or was stopped and the user says
  "tear it down", "shut down the pod", "we're done, clean up", (2) the user
  asks "is anything still running", "am I still being billed", "any pods
  left on", "check for orphaned pods", (3) a pod self-stopped or was stopped
  and artifacts still need retrieving ("rescue my checkpoints", "the pod
  stopped before I pulled"), (4) cost concern at any time. Drives
  scripts/remote/cli.py down / rescue in the ai-toolkit repo and the RunPod
  API for an account-wide live-pod scan. Verified teardown never destroys
  un-pulled checkpoints. For following a live run use ai-toolkit-remote-monitor.
---

# AI Toolkit remote teardown (RunPod)

Shut a run's pod down with verified artifact retrieval, rescue artifacts
off an already-stopped pod, and confirm the account has nothing billing.
All via `scripts/remote/cli.py` in `~/repos/ai-toolkit` + the RunPod API.

## Verified teardown (the normal path)

```bash
cd ~/repos/ai-toolkit
python scripts/remote/cli.py down <run>
```

`down` does, in order: final incremental pull -> **verify** the expected
checkpoints landed locally -> touch the `pulled.ok` ack -> terminate the
pod -> print the cost. It **never terminates unverified data**: if the pull
or verification fails it *stops* (not terminates) the pod, names exactly
what's missing, and returns a non-zero exit — your checkpoints stay on the
pod's volume for a `rescue`.

- `down <run> --force` terminates anyway despite missing artifacts (exit 2,
  "forced incomplete") — only when you've accepted the loss.
- `down` on an already-terminated/gone pod succeeds quietly and still
  reports cost from the manifest.

Expected-checkpoint math accounts for early stops: a run stopped at step
1500 of 5000 verifies against the checkpoints it actually produced, not the
full-run count.

## Rescue (stopped pod, artifacts still on it)

If a pod **self-stopped** (cost dead-man fired after the run ended) or was
stopped on a failed-pull, its volume still holds the checkpoints but it has
no running container to SSH into. `rescue` brings them back:

```bash
python scripts/remote/cli.py rescue <run>
```

It zero-GPU-starts the stopped pod (cheap — no GPU charge, just the brief
CPU+volume), re-resolves the SSH endpoint, pulls, verifies, then terminates.
This is the expected endgame after any post-completion self-stop, because
**stopped pods keep billing volume storage** until terminated. Live-validated
2026-06-10.

`rescue` refuses a RUNNING pod (use `down`) and a GONE pod (nothing to
rescue — reports the local state).

## Account-wide "is anything billing?" scan

When the user asks whether anything is still on — or to be safe after any
teardown — check the whole account, not just one run's manifest (a manifest
can go stale, and `--force-new` runs or manual console pods won't be in the
run you're looking at):

```bash
cd ~/repos/ai-toolkit && python3 - <<'PY'
import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.append("."); from dotenv import load_dotenv; load_dotenv(".env")
import runpod; runpod.api_key = os.environ["RUNPOD_API_KEY"]
pods = runpod.get_pods() or []
if not pods:
    print("CLEAN — no pods on the account (nothing billing).")
for p in pods:
    print(f'{p["id"]}  {p.get("name")}  {p.get("desiredStatus")}  ${p.get("costPerHr","?")}/hr')
PY
```

`desiredStatus: RUNNING` = billing GPU + volume. `EXITED` = billing volume
storage only (still costs a little — terminate it). Empty list = fully
clean. To kill a stray pod by id:

```bash
python3 -c "import os,sys,warnings;warnings.filterwarnings('ignore');sys.path.append('.');from dotenv import load_dotenv;load_dotenv('.env');import runpod;runpod.api_key=os.environ['RUNPOD_API_KEY'];runpod.terminate_pod('<POD_ID>');print('terminated <POD_ID>')"
```

(Or `scripts/remote/pod.terminate_pod` — it's idempotent.)

## Cost context

- MaxQ 96GB ~$0.50/hr, A100 80GB ~$1.39/hr. A typical 3000-step Klein run is
  ~$2.50. An EXITED pod's volume is pennies/hr but non-zero — terminate
  promptly.
- `down`/`rescue`/`status` all print a cost estimate (elapsed x rate from
  provision time).

## Common mistakes

- **`terminate` instead of `down` on a finished run you haven't pulled** —
  destroys the checkpoints. Use `down` (it pulls + verifies first). Only the
  account-scan kill-by-id is for confirmed-empty or already-pulled pods.
- **Assuming self-stop means terminated** — self-stop *stops* the pod
  (volume still billing); `rescue` then terminates it. Run the account scan
  to confirm nothing's left EXITED.
- **Trusting one manifest for "am I billing?"** — always run the
  account-wide scan; `--force-new` pods and console-created pods won't be in
  the run you're inspecting.
- **`--force` out of habit** — it terminates with missing artifacts. Only
  use it when you've decided the missing checkpoints don't matter.

## Related skills

- `ai-toolkit-remote-monitor` — follow a live run (it routes here at the end).
- `ai-toolkit-remote-launch` — start a run.

Full reference: `scripts/remote/README.md` in the ai-toolkit repo.
