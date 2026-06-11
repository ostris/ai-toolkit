---
name: ai-toolkit-train
description: >
  Orchestrate an end-to-end AI Toolkit LoRA training run, walking the user
  step by step from dataset to finished checkpoint and coordinating the
  individual stage skills. Use when: (1) the user says "walk me through
  training", "train a LoRA end to end", "start a new training project",
  "manage the whole training process", "I want to train X, guide me",
  (2) the user has a dataset and a goal but isn't sure of the sequence,
  (3) the user wants the full lifecycle (config -> caption -> audit ->
  train on RunPod -> review -> pick checkpoint -> teardown) run as one
  guided flow with checkpoints between stages. This is the CONDUCTOR: it
  holds the sequence, the go/no-go gates, and the handoffs, and invokes the
  stage skills (ai-toolkit-lora-config, ai-toolkit-gemini-captioner,
  style-vs-content-caption-auditor, ai-toolkit-dataset-diagnostics,
  ai-toolkit-remote-launch/monitor/teardown, ai-toolkit-sample-reviewer) at
  the right moments. For a single stage (just a config, just a caption
  audit, just checking on a running pod), invoke that stage's skill directly
  instead.
---

# AI Toolkit training orchestrator

You are conducting an end-to-end LoRA training run. Your job is **sequence,
gates, and handoffs** — not doing each stage's work yourself. Each stage has
a dedicated skill that owns the how; you own the order, the go/no-go
decisions, and keeping the user oriented ("here's where we are, here's
what's next, here's the call you need to make").

**Core discipline: one stage at a time, stop at every gate.** Never run two
stages silently. After each stage, report the outcome, then pause at the
next gate for the user's call before proceeding — especially the gates that
cost money or are hard to undo (config approval, the caption-audit gate, the
launch decision, the checkpoint pick).

## The stage machine

```
0. Frame      -> goal, dataset location, target model, compute (local vs RunPod)
1. Config     -> ai-toolkit-lora-config            [GATE: user approves the YAML]
2. Caption    -> ai-toolkit-gemini-captioner
3. Audit      -> style-vs-content-caption-auditor  [GATE: leak rate acceptable]
   (3b. Triage -> ai-toolkit-dataset-diagnostics, only if something looks off)
4. Launch     -> ai-toolkit-remote-launch          [GATE: preflight clean, $ ok]
                 (or local `python run.py` if the user chose local)
5. Monitor    -> ai-toolkit-remote-monitor  + ai-toolkit-sample-reviewer (loop)
6. Pick       -> ai-toolkit-sample-reviewer         [GATE: user picks checkpoint(s)]
7. Teardown   -> ai-toolkit-remote-teardown        [confirm: nothing left billing]
```

Track which stage you're in and state it at every turn ("Stage 3 of 7:
caption audit"). The run is resumable — if the user comes back mid-flow,
re-orient from what exists on disk (config in `config/examples/`, captions
next to the dataset, `runs/<run>/manifest.json` for a live remote run) and
resume at the right stage.

## Stage 0 — Frame the run (do this first, always)

Establish, by asking only what you can't infer:

- **Goal**: style / character / subject / motion LoRA? What should the
  trigger do?
- **Dataset**: where are the images/videos? How many? Captioned yet?
- **Target model**: Flux.2 Klein, Qwen-Image-Edit, Wan2.2, SDXL, ...?
  (Derrick trains Qwen-Image-Edit most often and Klein 9B for stubborn
  styles — default questions there unless told otherwise.)
- **Compute**: **RunPod** (the hosted pipeline — default now; replaces
  Colab) or **local**? This decides Stage 4.

Confirm the frame in 2-3 lines before moving on. Don't over-interrogate —
infer model/params from the goal and let Stage 1 refine.

## Stage 1 — Config  ·  invoke `ai-toolkit-lora-config`

Hand the goal + dataset + target model to `ai-toolkit-lora-config` (or
`flux2-klein-lora-config` for Klein). It produces the YAML, captioning
strategy, and sample-prompt design.

**GATE:** show the user the config's key decisions (model/arch, rank/alpha,
steps, save/sample cadence, trigger, resolution buckets) and get approval
before captioning. A wrong config here wastes the whole run. Always confirm
`logging.use_ui_logger: true` and a high/keep-all `max_step_saves_to_keep`
are set (both are training-correctness rules, not preferences).

## Stage 2 — Caption  ·  invoke `ai-toolkit-gemini-captioner`

Generate captions with the per-dataset Gemini captioner matching the goal
(style / character / motion / etc.). Captions land as `.txt` next to each
image. Runs from the `.venv-captioning` venv, not the training venv.

## Stage 3 — Audit  ·  invoke `style-vs-content-caption-auditor`

**This is the most important gate before spending compute.** Audit the
captions for leakage — words describing what should be the trigger. Report
the per-category leak rate.

**GATE:** if the leak rate is high, do NOT proceed to training. Spot-fix or
recaption (the auditor routes back to the captioner), then re-audit. Only
advance when captions are clean. A leaky dataset produces a weak trigger and
a wasted multi-hour run.

### Stage 3b — Triage (conditional)  ·  invoke `ai-toolkit-dataset-diagnostics`

Only if something looks wrong — dataset feels mis-sized, you suspect
AppleDouble pollution, wrong extensions, bucket issues, or you want a
pre-flight sanity sweep. Skip when the dataset is known-good.

## Stage 4 — Launch

**Branch on the Stage 0 compute choice.**

**RunPod (default):** invoke `ai-toolkit-remote-launch`. It runs
`preflight` (free validation) then `up` (provision + sync + launch).

- **GATE:** read the preflight output together — path remaps, file/caption
  counts, warnings — before provisioning. Preflight failure is free; a bad
  config discovered after a pod is running is not.
- Confirm GPU + rough cost with the user (MaxQ 96GB ~$0.50/hr is the default;
  a 3000-step Klein run ≈ $2.50). Always use `--gpu-fallback`.
- First run spends 20-60 min in `warming` (model download) — that's normal,
  not a hang.

**Local:** the user runs `python run.py config/examples/<config>.yaml`
themselves (suggest the `! ` prefix so its output lands in the session).
Then skip the remote monitor and go straight to Stage 6 review against the
local `output/<run>/` folder.

## Stage 5 — Monitor + review loop  ·  `ai-toolkit-remote-monitor` (+ reviewer)

For a RunPod run, invoke `ai-toolkit-remote-monitor`. It polls
`watch --once --json` on a schedule, pulls samples/checkpoints into
`output/<run>/`, and surfaces new reviewable sample steps (exit code 10).

When a new batch is reviewable, invoke `ai-toolkit-sample-reviewer` on
`output/<run>/samples/` — **once per batch, serial montage review, never
parallel per-step subagents** (a known laptop crash mode). Then advance the
watermark with `mark-reviewed`. Surface health signals (DEGRADED / disk /
drift) as the monitor reports them.

If the reviewer's verdict mid-run is "a mid checkpoint is the winner /
overcooked", stop early (`stop`) — training preserves through the last save.

## Stage 6 — Pick the checkpoint  ·  `ai-toolkit-sample-reviewer`

When the run reaches a terminal state (or you stopped early), do the final
review across the saved checkpoints.

**GATE:** remember **LoRA checkpoint trajectories are non-monotonic** —
review 3+ late checkpoints, never auto-pick the last save. The user picks
the winner (and optionally a merge of a few). This is their call, not yours.

## Stage 7 — Teardown  ·  invoke `ai-toolkit-remote-teardown`

For a RunPod run, invoke `ai-toolkit-remote-teardown`: `down` does the final
verified pull + terminate + cost report; `rescue` retrieves artifacts from a
self-stopped pod. **Always close with the account-wide billing scan** so the
user knows nothing is left running. (Local runs have nothing to tear down.)

## Orientation & resume

At the start of every turn while orchestrating, state the current stage and
the immediate next action. If re-entering a run cold:

- Config exists in `config/examples/`? -> past Stage 1.
- Captions (`.txt`) next to the dataset? -> past Stage 2.
- `runs/<run>/manifest.json` exists? -> a remote run was launched; use
  `ai-toolkit-remote-monitor`'s `attach` to recover its state, then resume at
  Stage 5/6/7 based on the run state.
- Samples/checkpoints in `output/<run>/`? -> in or past Stage 5.

## What this skill does NOT do

- It does not re-implement any stage — it invokes the stage skill. If you
  catch yourself writing a captioner or a config inline, stop and invoke the
  dedicated skill.
- It does not skip gates to "save time". The gates are where the user's
  money and hours are protected.
- It does not pick the checkpoint for the user (Stage 6 is their call).
- For a single isolated stage, the user should invoke that stage's skill
  directly; this orchestrator is for the guided full lifecycle.

## Related skills (the stages)

`ai-toolkit-lora-config` · `flux2-klein-lora-config` · `ai-toolkit-gemini-captioner`
· `style-vs-content-caption-auditor` · `ai-toolkit-dataset-diagnostics` ·
`dop-class-advisor` · `ai-toolkit-remote-launch` · `ai-toolkit-remote-monitor`
· `ai-toolkit-remote-teardown` · `ai-toolkit-sample-reviewer`

Remote-pipeline reference: `scripts/remote/README.md`. Skill set + activation:
`skills/README.md`.
