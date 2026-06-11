---
name: ai-toolkit-remote-launch
description: >
  Launch an AI Toolkit LoRA training run on a hosted RunPod GPU from the Mac
  CLI (replaces the Colab+Drive workflow). Use when: (1) the user has a
  training config + captioned dataset ready and says "train this on RunPod",
  "launch a remote run", "kick off training in the cloud", "run this on a
  hosted GPU", "spin up a pod and train", (2) the user wants to start a run
  whose samples/checkpoints stream back to the Mac for review, (3) any time a
  local config is ready and the user does NOT want to train on their own
  machine. Drives scripts/remote/cli.py preflight -> up (provision + sync +
  launch) in the ai-toolkit repo. Hands off to ai-toolkit-remote-monitor for
  the watch/review loop and ai-toolkit-remote-teardown for shutdown. This is
  the KICKOFF skill only — for an already-running pod use the monitor skill.
---

# AI Toolkit remote launch (RunPod)

Kick off a LoRA training run on a RunPod Secure Cloud pod. Everything runs
from the Mac CLI via `scripts/remote/cli.py` in the **ai-toolkit repo**
(`~/repos/ai-toolkit`); the pod runs the pinned `ostris/aitoolkit` image.

This skill covers **launch only**: validate the config, provision a pod,
sync the dataset + config up, and start training under the cost-safe
sentinel wrapper. Once it's running, switch to
**`ai-toolkit-remote-monitor`** to follow it and **`ai-toolkit-remote-teardown`**
to shut down.

## Preconditions (check, don't assume)

The remote pipeline is config-and-dataset-in, trained-LoRA-out. Before
launching, the upstream prep should already be done (these have their own
skills): config generated (`ai-toolkit-lora-config` /
`flux2-klein-lora-config`), dataset captioned (`ai-toolkit-gemini-captioner`),
captions audited (`style-vs-content-caption-auditor`). If the dataset looks
unprepped, route there first — a remote run costs real money, so don't
launch a config you haven't validated locally.

`.env` in the repo must carry `RUNPOD_API_KEY`, `RUNPOD_STOP_API_KEY`, and
`HF_TOKEN`. If `RUNPOD_API_KEY` is missing the CLI says so; the README
§1 documents setup.

## The launch sequence

```bash
cd ~/repos/ai-toolkit

# 1. Preflight — free, no network. Validates YAML + dataset, rewrites all
#    local paths to pod paths, writes runs/<run>/remote_config.yaml.
python scripts/remote/cli.py preflight config/examples/<config>.yaml

# 2. up — provision + sync + launch in one shot.
python scripts/remote/cli.py up config/examples/<config>.yaml \
    --gpu MAXQ --gpu-fallback "A100" --gpu-fallback "A100 SXM"
```

Always run `preflight` first and read its output — it reports every path
remap, the dataset file/caption counts, and any warnings (e.g.
`save.push_to_hub` forwarding a write-scoped HF token). A preflight failure
is free; a bad config discovered after provisioning is not.

`up` chains preflight -> provision -> sync -> launch and **tears the pod
down if any post-provision step fails**, so a half-provisioned pod never
idle-bills. If you want the steps separately (e.g. to inspect the pod
before launching), run `provision`, `sync`, then `launch` individually —
same flags.

### GPU choice — ASK, don't assume

**Always ask the user which GPU to provision before launching. Do NOT
silently default to the cheapest.** Present a short speed-vs-cost choice and
let them pick. For a first-time trainer, recommend the cheapest card that
fits the model's VRAM floor; honor a known speed-over-price preference
(e.g. from memory) when the user has one. Offer roughly:

- **Fastest** — `H100` (SXM, 80G, ~$3.29/hr), `H100 NVL` (94G, ~$3.19/hr),
  or `H200` (141G, ~$3.79/hr). ~1.5–2× A100 training throughput.
- **Balanced** — `A100` (80G, ~$1.39/hr) / `A100 SXM` (~$1.49/hr).
- **Cheapest** — `MAXQ` (RTX PRO 6000 Blackwell, 96G, ~$0.50/hr) — cheapest
  80GB+ card, but the slowest of these.

Cost is ~neutral for a fixed run (a faster GPU finishes in fewer hours), so
speed is usually the better axis. Always pass `--gpu-fallback` to a
same-tier alternate since any single type is frequently out of stock — e.g.
`--gpu H100 --gpu-fallback "H100 NVL" --gpu-fallback "H100 PCIE" --gpu-fallback "A100"`.

| GPU (alias) | VRAM | Secure $/hr | speed tier |
|---|---|---|---|
| `H200` | 141G | ~3.79 | fastest |
| `H100` (SXM) | 80G | ~3.29 | fastest |
| `H100 NVL` | 94G | ~3.19 | fastest |
| `A100 SXM` | 80G | ~1.49 | balanced |
| `A100` (PCIe) | 80G | ~1.39 | balanced |
| `MAXQ` | 96G | ~0.50 | cheapest (slowest) |

Full alias list in `scripts/remote/pod.py` (`H100`, `H100 NVL`, `H200`,
`A100`, `A100 SXM`, `MAXQ`, `PRO 6000`, `A6000`, `RTX 6000 ADA`, `L40`,
`L40S`, `4090`, `5090`, ...); unknown names pass through as exact RunPod ids.
A 2500-step Klein run is ~$3–5 on MaxQ and faster (often similar total $) on
an H100.

### Fit the GPU to the model's VRAM need (don't over-provision)

Pick the smallest VRAM tier that clears the model's floor at the chosen
precision, THEN apply the speed preference within tiers that fit. Forcing
80GB+ on a model that trains fine in 48GB wastes money and often *speed* —
the 48GB cards (A6000 ~$0.49, RTX 6000 Ada ~$0.77, L40/L40S ~$0.82) are
cheaper and can be faster than MaxQ. Conversely, never recommend a card
below the floor — it OOMs before step 1.

Starting fit guidance (LoRA training; **verify per model**, quantization
shifts these down a tier):

| Model (`arch`) | Unquantized floor | With `quantize: true` | Notes |
|---|---|---|---|
| Flux.2 dev | 80G (can still OOM) | ~48–80G | batch 1 + grad-accum; OOM-prone even at 80G |
| Flux.2 Klein 9B | 80G | ~24–48G | repo configs run it unquantized on 80G+ |
| Qwen-Image-Edit (2511) | ~80G | ~48G (`quantize_te`) | quantized fits 48G |
| Wan2.2 14B (video) | 80G+ | — | `gradient_checkpointing` required even at 95G |
| SDXL | 24G | — | fits anywhere |
| Z-Image Turbo / Base | ~24G | — | small; 24–48G is plenty |

So: a quantized Qwen-Image-Edit or SDXL/Z-Image run should be offered the
48GB (or 24GB) tier — fast AND cheap — not an 80GB card. An unquantized
Klein/Flux.2/Wan run needs 80GB+, so the choice there is which 80GB+ card
(speed vs cost). Read the config's `model.quantize` / `quantize_te` and
`arch` to decide the floor before presenting GPU options.

### Multiple GPUs (`--gpus N`)

`--gpus N` (on `provision` / `up`) provisions an N-GPU pod and trains
data-parallel via `accelerate launch --multi_gpu --num_processes N` instead
of plain `python run.py` (ai-toolkit's run.py is built for this — model,
LoRA network, and optimizer are `accelerator.prepare()`'d, and
sampling/saving/logging are rank-0-guarded, so there's still one
`loss_log.db`, one samples dir, one set of checkpoints). Default is 1 GPU
(the fully-validated path).

Three things to tell the user before using it:

- **Effective batch scales ~Nx** — the dataset is seen ~N× faster per step,
  so step counts/LR tuned for 1 GPU will OVER-train. Cut `train.steps`
  roughly N× (launch prints this warning when `gpu_count > 1`).
- **No per-GPU VRAM savings** — each GPU holds a full frozen base model, so
  all N GPUs must be 80GB+ (MaxQ/A100 fine).
- **Sublinear speedup** — the frozen-base forward dominates, so 4 GPUs is
  faster but not 4×; it costs ~N× the hourly rate. Worth it for long runs or
  when wall-clock matters, not for a quick 100-step job.

The exact `accelerate launch` flag set is conservative (`--mixed_precision
no --dynamo_backend no`, letting ai-toolkit own dtype/compile) and is the
**one piece validated only on the first real multi-GPU run** — override via
the `AITK_ACCELERATE_ARGS` env var if a pod needs different flags. Single-GPU
(the default) is fully live-validated.

## What "up" actually does (so you can narrate it / debug it)

1. **preflight** (again, idempotent) — derived config + hash into the manifest.
2. **provision** — creates the pod, filters for direct public-IP SSH (proxy
   SSH can't carry rsync), sizes the volume from steps/save_every, injects
   `PUBLIC_KEY` / `HF_TOKEN` / `RUNPOD_STOP_KEY` / `HF_HOME`, overrides the
   container start command so the pod survives the image's UI process
   exiting, and **probes the stop key before spending** — if it warns the
   key can't stop pods, fix it (teardown skill / README §1) before relying
   on unattended self-stop.
3. **sync** — rsyncs dataset + ctrl images + derived config + self-stop tool
   + the local repo overlay (allowlisted dirs) up.
4. **launch** — uploads the sentinel wrapper + max-runtime timer, starts
   `run.py` inside tmux, then block-tails the log and reports one of:
   `warming` (model download, 20-60 min on first run — expected),
   `running` (already training), or `crashed` (prints the traceback, leaves
   the pod up for diagnosis).

When `up` returns with the run RUNNING or warming, **hand off**: tell the
user the run name and that you'll follow it, then use
`ai-toolkit-remote-monitor`.

## Re-launch / resume guards

If the manifest already records a live pod, `provision`/`up` **refuse**
(so you can't double-bill) — use `status`/`down`/`rescue` on the existing
pod, or `--force-new` to deliberately provision a second one.

`launch` refuses if checkpoints already exist remotely unless you pass
`--resume` (continue from the latest checkpoint — ai-toolkit auto-resumes)
or `--fresh` (move the old output aside and start clean; this also resets
the local pull watermarks).

## Cost safety (what protects you)

The pod stops *itself* when training ends: the wrapper writes an exit-code
sentinel, waits for the laptop's `pulled.ok` ack (bounded by `--max-grace`,
default 30 min), then self-stops via the RunPod API. A separate
max-runtime timer (`--max-hours`, default 24) is the hard backstop. **Laptop
sleep cannot leave a pod billing.** This was live-validated 2026-06-10.

**Right-size `--max-hours` on every launch — don't ride the 24h default.**
The backstop only limits damage if it's near the real run length: estimate
wall-clock (steps × s/step for the model+GPU, plus up to 1h warming) and set
`--max-hours` to ~2× that estimate. A crashed-then-abandoned pod bills the
full GPU rate until this timer fires — 24h of H100 is ~$80, a right-sized
6h cap is ~$20. This matters most for a first-time trainer who won't
recognize a crash or know to tear down. Short runs: `--max-hours 2
--max-grace 240`.

## Common mistakes

- **Launching without preflight** — you discover a bad path after paying for
  a pod. Preflight is free; always run it.
- **No `--gpu-fallback`** — single GPU types go out of stock constantly;
  the run fails fast instead of falling back.
- **Forgetting this is the launch skill** — if the pod is already running,
  don't re-`up` (it refuses anyway); use `ai-toolkit-remote-monitor`.
- **Assuming `warming` is a hang** — first runs download ~18GB of Klein/
  Flux.2 weights before step 1. 20-60 min of silence is normal.
- **Passphrase-protected SSH key** — the pipeline uses a dedicated
  passphrase-less `~/.ssh/aitk_remote_ed25519` (BatchMode can't prompt);
  if SSH "times out" during provision, that key is missing (README §1).

## Related skills

- `ai-toolkit-remote-monitor` — follow the run, pull artifacts, drive the
  sample-review loop, decide when to stop.
- `ai-toolkit-remote-teardown` — `down`/`rescue`, cost report, orphan-pod
  scan.
- `ai-toolkit-lora-config` / `flux2-klein-lora-config` — generate the config.
- `style-vs-content-caption-auditor` — audit captions before the run.

Full reference: `scripts/remote/README.md` in the ai-toolkit repo.
