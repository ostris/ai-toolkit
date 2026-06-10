# Remote GPU training on RunPod

Fully agentic LoRA training from the Mac CLI. This pipeline replaces the
Colab+Drive loop: preflight a config locally, provision a RunPod Secure Cloud
pod, sync the dataset and a derived config, launch a sentinel-wrapped
training run inside tmux, watch it on a schedule, pull samples and
checkpoints incrementally into `output/<run>/`, and tear the pod down with
verified artifacts. Every command is re-entrant from
`runs/<run>/manifest.json` + the RunPod API alone — a fresh Claude Code
session with zero context can `attach <run>` and keep going.

This guide is written for both a human and a Claude Code agent driving the
CLI. Sections 5 (the agentic loop) and the exit-code table are the machine
contract.

```
python scripts/remote/cli.py <subcommand>     # or: python -m scripts.remote.cli
```

| Subcommand | What it does |
|---|---|
| `preflight <config>` | Validate config + dataset, write `runs/<run>/remote_config.yaml`. Local only. |
| `provision <run>` | Create a pod (`--gpu/--gpu-fallback/--image/--disk-gb/--dry-run`), wait for SSH. Refuses when the manifest already records a pod that is not GONE — use `status`/`down`/`rescue` on the existing pod, or `--force-new` to provision anyway. |
| `sync <config>` | Upload dataset, ctrl images, derived config, self-stop tool, repo overlay. |
| `launch <run>` | Guarded start (`--resume/--fresh/--max-hours/--max-grace`). |
| `up <config>` | preflight → provision → sync → launch; tears the pod down on post-provision failure. |
| `status <run>` | State, step, loss, OOM health, disk, drift, cost. |
| `pull <run>` | One incremental artifact pull (never deletes local files). |
| `watch <run>` | status+pull loop; `--once --json` is the agent polling contract. |
| `stop <run>` | Kill the trainer early; sentinel/self-stop mechanics still run. |
| `down <run>` | Final pull → verify → ack → terminate (`--force` to skip verify). |
| `attach <run>` | Rebuild watcher state in a fresh session. |
| `rescue <run>` | Zero-GPU start a stopped pod, pull, verify, terminate. |
| `mark-reviewed <run> <step>` | Advance the review watermark after reviewing samples. |

---

## 1. Setup — RunPod API key (USER-REQUIRED)

You need a RunPod account, an API key in `.env`, and billing credit.

1. Create an account at [runpod.io](https://runpod.io).
2. Open the **Console → Settings → API Keys**.
3. Click **"+ API Key"**, name it (e.g. `ai-toolkit-laptop`), give it
   **Read/Write** permission, and create it.
4. **Copy the key immediately** — RunPod shows it only once.
5. Add it to the repo's `.env` file (gitignored; never commit keys):

   ```
   RUNPOD_API_KEY=rpa_XXXXXXXXXXXXXXXX
   ```

6. Add billing credit: **Console → Billing → Add funds**. $10 covers roughly
   5 A100-hours — one to two full training runs.

### Recommended: a second, restricted key for pod-side self-stop

The pod stops *itself* when training ends (the cost dead-man, section 7).
That requires an API key living **on the pod**, so scope it down:

1. Create a second key (**Settings → API Keys → "+ API Key"**).
2. Give it the **minimum permission RunPod's console offers**. If
   per-permission scoping is available, grant **Pods Read/Write only** —
   nothing else. If the console only offers full-scope keys, understand the
   residual risk: a pod-resident key can act on your whole account (create
   pods, spend credit). The key exists only for the pod's lifetime and the
   pod is yours alone, but it is provider-visible.
3. Add it to `.env`:

   ```
   RUNPOD_STOP_API_KEY=rpa_YYYYYYYYYYYYYYYY
   ```

   If unset, the CLI falls back to `RUNPOD_API_KEY` **with a warning** — the
   full account key then rides on the pod.

### HF_TOKEN

Gated models (Flux.2, Klein) need a HuggingFace token, forwarded to the pod:

1. [huggingface.co](https://huggingface.co) → **Settings → Access Tokens →
   Create new token**.
2. Scope it **READ**. Only use a write-scoped token if your config sets
   `save.push_to_hub` (preflight warns about this — the token is
   provider-visible on the pod, so never forward more scope than needed).
3. Add `HF_TOKEN=hf_...` to `.env`.

### SSH key

Transport is plain ssh/rsync as root. You need `~/.ssh/id_ed25519.pub` or
`~/.ssh/id_rsa.pub`. If absent:

```
ssh-keygen -t ed25519
```

---

## 2. Install

```
pip install -r scripts/remote/requirements.txt
```

Install into your **laptop venv** (the one you run the CLI from), NOT the
training venv, and never merge these pins into the repo's training
requirements — the pod's training environment is the pinned
`ostris/aitoolkit` image.

macOS note: Apple ships `openrsync`. The CLI warns if `rsync --version`
doesn't report GNU rsync 3.x; transport restricts itself to portable flags
so openrsync works, but `brew install rsync` is recommended.

---

## 3. Quickstart

```bash
# 1. Validate config + dataset locally (free, no network)
python scripts/remote/cli.py preflight config/examples/train_lora_flux2_klein_9b_balfua_style_v3.yaml

# 2. Provision + sync + launch in one shot
python scripts/remote/cli.py up config/examples/train_lora_flux2_klein_9b_balfua_style_v3.yaml

# 3. Follow the run (blocks; exits with a distinct code on terminal states)
python scripts/remote/cli.py watch balfua_v3

# 4. Final pull, verify, terminate, cost report
python scripts/remote/cli.py down balfua_v3
```

Artifacts land in `output/<run>/` mirroring the trainer's own layout, so all
existing review tooling works unchanged. Run state lives in
`runs/<run>/manifest.json` (gitignored).

---

## 4. The agentic loop (how Claude Code drives it)

The full pipeline, with the existing skill at each stage:

1. **`ai-toolkit-lora-config`** — produce the training YAML from reference
   images and a goal.
2. **Caption scripts** (`ai-toolkit-gemini-captioner`) — caption the dataset.
3. **`style-vs-content-caption-auditor`** — audit captions for leakage
   before committing to a multi-hour run.
4. **`preflight`** — catches every cheap failure before money is spent.
5. **`up`** — provision, sync, launch. Note the warming case: first runs
   download Klein/Flux.2 weights for 20-60 minutes before step 1.
6. **`watch --once --json` on a schedule** — see the polling pattern below.
7. **`ai-toolkit-sample-reviewer`** on `output/<run>/samples/` whenever new
   reviewable sample steps appear. Invoke it **once per new-steps batch**
   (not one call per step) — serial montage-based review; parallel per-step
   image-review subagents are a known failure mode. After each review, run
   **`mark-reviewed <run> <step>`** with the highest step just reviewed —
   this advances the watermark; without it `watch --once` re-reports the
   same steps (exit 10) forever.
8. **`stop <run>`** early if the reviewer's verdict is "done/overcooked" —
   the desired checkpoint is usually earlier than the final step.
9. **`down <run>`** — final pull, verify, terminate, cost report.
10. **Checkpoint pick** — non-monotonic trajectories: review 3+ late
    checkpoints, never auto-pick the last save. The JSON's
    `pulled_checkpoint_steps` lists exactly which checkpoint steps are
    already local to choose from.

### Polling pattern for multi-hour runs

Do **not** hold a `watch` process open for hours. Re-invoke on a schedule
(every ~10 minutes) and branch on the exit code:

```bash
python scripts/remote/cli.py watch <run> --once --json
```

This performs ONE status+pull cycle and prints one JSON object:

| JSON field | Meaning |
|---|---|
| `run` | run name |
| `state` | current run state (see exit codes below) |
| `step` / `total_steps` | progress from `loss_log.db`, queried in place |
| `loss` | recent loss value (null early on) |
| `oom_skips` | OOM-skipped batches seen in the log tail (DEGRADED signal) |
| `disk_used_pct` | `/workspace` usage — keep-all checkpoints make disk-full a real risk |
| `drift` | true when the local derived config no longer matches the launch hash |
| `reviewable_steps` | sample steps with complete batches, newer than `last_reviewed_step` |
| `last_reviewed_step` | watermark; advance with `mark-reviewed <run> <step>` after reviewing |
| `cost_estimate` | elapsed × hourly rate captured at provision |
| `detail` | human-readable state explanation (crash code, stall, OOM count, drift) |
| `log_tail_path` | absolute path to the mirrored log tail (`runs/<run>/log_tail.txt`); null until a pull mirrors it |
| `pulled_checkpoint_steps` | sorted checkpoint steps already pulled into `output/<run>/` |
| `exit_code` | mirrors the process exit code |

### Exit codes (the complete contract)

| Code | State | Agent action |
|---|---|---|
| `0` | COMPLETED | run `down <run>`. TERMINATED also resolves exit 0 after a clean `down`/`rescue` teardown — the run is finished and fully accounted for. |
| `3` | running, nothing new | sleep ~10 min, re-invoke. Also covers DEGRADED (inspect `oom_skips`/`state` in the JSON) and SAMPLING (normal during sample generation). |
| `10` | new reviewable sample steps | run `ai-toolkit-sample-reviewer`, then `mark-reviewed <run> <step>`, then re-invoke |
| `20` | CRASHED | pod left up — read `detail` and the file at `log_tail_path` in the JSON, diagnose, then `down` |
| `21` | STOPPED | early stop landed; `down <run>` |
| `22` | TIMED_OUT | max-runtime timer fired (or pod stopped with no sentinel); `rescue` then inspect |
| `23` | UNKNOWN | pod running but tmux session gone, no sentinel — container restart or tmux death; ssh in to inspect |
| `24` | POD_LOST | pod gone from the RunPod API; local mirrors under `runs/<run>/` are all that remain |
| `1` | — | CLI error (bad args, missing manifest, API failure) on any command |
| `2` | DOWN_FORCED_INCOMPLETE | `down --force` terminated with missing artifacts — the missing files were named; nothing more is pullable |

Non-watch commands exit `0` on success, `1` on failure (`down` and `rescue`
exit `1` specifically when artifact verification fails — the pod is stopped,
not terminated, and the missing artifacts are named).

---

## 5. First-run live validation (REQUIRED once)

Before trusting unattended runs, execute the plan's U5 gate once against a
real pod (~$1-2 on the cheapest GPU):

1. Provision the cheapest available GPU (`provision <run> --gpu 4090` or
   similar) and run a **~100-step job** (copy a config, set
   `train.steps: 100`, `save.save_every: 50`, `sample.sample_every: 50`).
2. Verify, in order:
   - the **exit_code sentinel** appears at
     `/workspace/runs/<run>/exit_code` when training ends (content `0`);
   - the **ack handshake**: `down` touches `pulled.ok` and the pod
     self-stops promptly instead of waiting out max-grace;
   - **self-stop** fires on its own when you skip the ack (wait out
     max-grace once);
   - **UI-kill tmux survival**: `kill` the image's Node UI process over ssh
     and confirm the `aitk-train-*` and `aitk-timer-*` tmux sessions
     survive (the R27 start-command override working);
   - **`pgrep` exists** on the pod (`stop` and the timer depend on it; fall
     back to tmux pane-PID kill if a future image drops it).
3. Confirm the SDK call shapes that were flagged UNCERTAIN at
   implementation time (`scripts/remote/pod.py` adapters — a fix touches
   one function each):
   - `create_pod(..., docker_args=...)` is the correct start-command
     override parameter;
   - `resume_pod(pod_id, gpu_count=0)` performs the zero-GPU rescue start;
   - `get_pod(pod_id)` returns `None` for a gone pod vs raising — both are
     handled, but confirm which actually happens.

---

## 6. Cost

| GPU (Secure Cloud, on-demand) | ~Rate | 6h run |
|---|---|---|
| A100 80GB PCIe (default) | ~$1.89/hr | ~$11 |
| H100 80GB | ~$2.99/hr | ~$18 |
| RTX 4090 24GB (validation runs) | ~$0.69/hr | ~$4 |

Rates drift; the actual rate is captured at provision time into the manifest
and `status`/`down` report estimated cost from it.

**The cost dead-man design** — laptop sleep can never cause unbounded spend:

- The launch wrapper writes the exit-code **sentinel** when training ends,
  waits (bounded by `--max-grace`, default 30 min) for the laptop's verified
  final pull to **ack** with `pulled.ok`, then **self-stops** the pod.
- A hard **max-runtime timer** (default 24h, `--max-hours`) runs in its own
  tmux session: pkill the trainer (sentinel path still runs), then
  force-stop the pod if it is still up minutes later.
- **Stopped pods still bill volume storage.** Stop is never the endgame —
  `rescue` (zero-GPU start → pull → verify → **terminate**) is. After any
  self-stop, run `rescue <run>`.

---

## 7. Trust model

- **Everything remote runs as root on a provider pod.** The dataset sits on
  provider disk from `sync` until `down`/`rescue` terminates the pod.
- **HF_TOKEN is provider-visible** (pod env var). Use a READ-scoped token
  unless the config pushes to the Hub.
- **The self-stop key lives on the pod.** That's why a second, restricted
  key (section 1) matters: if it leaks, the blast radius is pod operations,
  not your whole account — to the extent RunPod's key scoping allows.
- No secrets ever enter configs, manifests, or git; they flow only through
  `.env` → environment → pod env.

---

## 8. The optimizer gap

The trainer overwrites a **single `optimizer.pt`** — only the **newest**
checkpoint carries optimizer state. `pull` snapshots it as
`output/<run>/optimizer_<step>.pt` paired (by mtime) with the newest pulled
checkpoint. Consequence: picking an *earlier* checkpoint as your winner means
any future resume/continuation from it starts the optimizer **cold**.
Acceptable for inference-only picks; relevant if you plan to train onward.

---

## 9. Manual fallback runbook

When the CLI can't tell you what's going on, go look:

```bash
# endpoint (re-resolve first: RunPod reassigns IP/port across stop/start)
python scripts/remote/cli.py status <run>          # refreshes the cache
cat runs/<run>/manifest.json                       # ssh_host / ssh_port

ssh -p <ssh_port> root@<ssh_host>

# on the pod:
tmux ls                          # expect aitk-train-<run> and aitk-timer-<run>
tmux attach -t aitk-train-<run>  # live trainer output (Ctrl-b d to detach)
```

Where everything lives on the pod:

```
/workspace/runs/<run>/
├── dataset/<folder>/        # uploaded dataset(s)
├── ctrl/                    # ctrl/mask/reference images
├── remote_config.yaml       # the derived config the trainer actually runs
├── log.txt                  # full trainer log (tail -c, it's \r-heavy)
├── exit_code                # sentinel: run.py's exit code, written by wrapper
├── timed_out                # sentinel: max-runtime timer fired
├── pulled.ok                # ack: laptop's verified final pull landed
├── train_wrapper.sh         # the generated sentinel wrapper
├── timer.sh                 # the generated max-runtime timer
├── self_stop.sh             # provider self-stop tool (uploaded at sync)
└── output/<job_name>/       # checkpoints, optimizer.pt, samples/, loss_log.db
```

Local mirrors that survive pod teardown: `runs/<run>/exit_code`,
`runs/<run>/log_tail.txt`, `runs/<run>/loss_log.db` (terminal pulls), plus
all pulled artifacts in `output/<run>/`.

Rescue flow for a stopped pod (self-stopped or stop-on-failed-pull):

```bash
python scripts/remote/cli.py rescue <run>   # zero-GPU start → pull → verify → terminate
```

If even that fails: the RunPod console can start/stop/terminate the pod
manually; the volume keeps your artifacts until **terminate**.

