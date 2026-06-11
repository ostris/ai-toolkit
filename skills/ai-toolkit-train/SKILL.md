---
name: ai-toolkit-train
description: >
  Orchestrate an end-to-end AI Toolkit LoRA training run, walking the user
  step by step from dataset to finished checkpoint and coordinating the
  individual stage skills. Use when: (1) the user says "walk me through
  training", "train a LoRA end to end", "start a new training project",
  "manage the whole training process", "I want to train X, guide me",
  (2) the user has a dataset and a goal but isn't sure of the sequence,
  (3) the user wants the full lifecycle (dataset check -> model brief ->
  config -> caption -> audit -> train on RunPod -> review -> pick
  checkpoint -> teardown) run as one
  guided flow with checkpoints between stages. This is the CONDUCTOR: it
  holds the sequence, the go/no-go gates, and the handoffs, and invokes the
  stage skills (ai-toolkit-model-brief, ai-toolkit-lora-config, ai-toolkit-gemini-captioner,
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
                 + expectations block (cost / time / walk-away)
0.5 Dataset   -> references/dataset-readiness.md   [GATE: GO / FIX FIRST / STOP]
0.75 Brief    -> ai-toolkit-model-brief            [GATE: user confirms the brief]
1. Config     -> ai-toolkit-lora-config            [GATE: user approves the YAML]
2. Caption    -> ai-toolkit-gemini-captioner
3. Audit      -> style-vs-content-caption-auditor  [GATE: leak rate acceptable]
   (3b. Triage -> ai-toolkit-dataset-diagnostics, only if something looks off)
4. Launch     -> ai-toolkit-remote-launch          [GATE: preflight clean, $ ok]
                 (or local `python run.py` if the user chose local)
5. Monitor    -> ai-toolkit-remote-monitor  + ai-toolkit-sample-reviewer (loop)
6. Pick       -> ai-toolkit-sample-reviewer         [GATE: user picks checkpoint(s)]
7. Teardown   -> ai-toolkit-remote-teardown        [confirm: nothing left billing]
8. v2 loop    -> if the verdict isn't a clean winner: diagnose, re-enter at
                 the stage the diagnosis points to (see "The v2 loop")
```

Track which stage you're in and state it at every turn ("Stage 3:
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
  Returning trainers usually have a go-to (check memory / ask); for a
  first-timer don't ask at all — let Stage 1 pick from the goal, licensing,
  and budget, and present the choice in outcome terms.
- **Compute**: **RunPod** (the hosted pipeline — default now; replaces
  Colab) or **local**? This decides Stage 4. If RunPod, also ASK which GPU —
  offer fastest (H100/H200) / balanced (A100) / cheapest-that-fits, and
  never silently pick. Recommend cheapest-that-fits for a first-timer;
  honor a known speed-over-price preference (e.g. from memory) when the
  user has one.

Confirm the frame in 2-3 lines before moving on. Don't over-interrogate —
infer model/params from the goal and let Stage 1 refine.

**Calibrate the user here, once.** If they're fluent in LoRA terms, run the
gates in shorthand. If this is their first training run (or they ask what a
LoRA/trigger/checkpoint is), switch to the first-timer gate protocol in
`references/first-timer-gates.md` for every gate that follows.

**Then set expectations before proceeding** — the three-line block from
`references/first-timer-gates.md`: rough total cost (~$3–10 GPU + <$1
captioning), rough wall-clock (prep ~15–30 min, training 2–6 h plus first-run
warming), and "you can walk away — the pod stops itself, your laptop can
sleep, we re-attach any time." Plus the draft framing: first runs usually
take 1–3 iterations. Cost and time surprises discovered mid-flow are how
first-time trainers get lost; surface them here, not at the launch gate.

## Stage 0.5 — Dataset readiness  ·  `references/dataset-readiness.md`

A clean run on a bad dataset is the most expensive failure in the pipeline —
every later gate passes and the result is still wrong. Before any config:
list the dataset folder, look at 5–8 representative images, and walk the
readiness checklist (count vs. goal, watermarks/logos/text, one aesthetic or
subject per folder, variety in what should stay promptable, duplicates,
quality floor, rights, mechanical hygiene). Watermarks are the
non-negotiable: they reproduce as garbled text in every output — crop them
before anything else.

Close with the **reflect-back**: describe in plain words what the model will
learn (trigger-bound) vs. what stays promptable, and ask "is that the thing
you want it to learn?" This is the one gate where a first-time trainer is
the expert — a goal mismatch caught here costs a sentence; the same mismatch
at Stage 6 costs the run.

**GATE:** GO / FIX FIRST (name each fix concretely, help apply, re-check) /
STOP (not trainable as-is — say what would make it trainable; don't soften
to a FIX). Mechanical issues (dotfiles, mixed media, loader traps) route to
`ai-toolkit-dataset-diagnostics` rather than hand-fixing here.

## Stage 0.75 — Model brief  ·  invoke `ai-toolkit-model-brief`

Brainstorm what the model needs to DO before anything gets configured:
dream prompts first ("write 3–5 prompts you wish you could type"), then
only the capability axes the prompts left ambiguous (text rendering,
generate-vs-edit, the fidelity↔flexibility dial, license, inference
destination). The skill writes `briefs/<project>-brief.md` — the artifact
Stage 1 consumes for model choice and sample prompts, Stage 6 scores
against, and the v2 loop carries forward.

Run it right after the dataset check on purpose: the reflect-back said what
the images *can teach*; the brief says what the artist *wants* — surface
any gap between the two now (collect more, or demote the requirement),
not at Stage 6. A requirement the dataset can't support routes back to 0.5.

**GATE:** read the brief back in plain words and get the user's "yes,
that's the model I want." Skippable only when the user is fluent AND the
goal is already unambiguous (e.g. a v2 rerun with an existing brief) — then
confirm the existing brief still holds instead of re-eliciting.

## Stage 1 — Config  ·  invoke `ai-toolkit-lora-config`

Hand the goal + dataset + target model + **the brief** to
`ai-toolkit-lora-config` (or `flux2-klein-lora-config` for Klein). It
produces the YAML, captioning strategy, and sample-prompt design — sample
prompts derived from the brief's dream prompts, model filtered through its
capability matrix by the brief's MUSTs.

**GATE:** show the user the config's key decisions (model/arch, rank/alpha,
steps, save/sample cadence, trigger, resolution buckets) and get approval
before captioning. For a first-time trainer, present at most 3-5 decisions
as a recommendation-first table in outcome language (the Stage 1 script in
`references/first-timer-gates.md`) — never the raw hyperparameter list; make
the remaining calls silently with defaults. A wrong config here wastes the
whole run. Always confirm
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
- Confirm the GPU + rough cost with the user — ASK, never assume. The
  launch skill carries the speed/balanced/cheapest options and the
  model→min-VRAM fit table. Always use `--gpu-fallback`.
  For a first-time trainer this is the money gate: lead with the cheapest
  card that fits the model, give the total-dollar estimate, and repeat the
  walk-away reassurance (Stage 4 script in `references/first-timer-gates.md`).
- Right-size the cost backstop: set `--max-hours` to ~2× the estimated
  wall-clock instead of riding the 24h default (the launch skill has the
  sizing rule). An abandoned crashed pod bills until this timer fires.
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

**On CRASHED: tear down first unless you are diagnosing right now.** The
crashed pod bills full GPU rate and the log tail is already local
(`runs/<run>/log_tail.txt`). Default — and the only mode for a first-time
trainer — is: pull, tear down (Stage 7), diagnose from the saved log, then
re-enter via the v2 loop. Only keep the pod up if you're actively debugging
in-session and the diagnosis genuinely needs the remote filesystem.

## Stage 6 — Pick the checkpoint  ·  `ai-toolkit-sample-reviewer`

When the run reaches a terminal state (or you stopped early), do the final
review across the saved checkpoints.

**GATE:** remember **LoRA checkpoint trajectories are non-monotonic** —
review 3+ late checkpoints, never auto-pick the last save. The user picks
the winner (and optionally a merge of a few). This is their call, not yours.
For a first-time trainer, present the reviewer's winner as the recommended
default with winner-vs-dataset images side by side and one runner-up — never
a bare list of checkpoint numbers (Stage 6 script in
`references/first-timer-gates.md`). Their eyes are the gate; their taste
outranks the rubric.

If the reviewer's verdict is "no checkpoint is acceptable", don't force a
pick — proceed to Stage 7 teardown and then "The v2 loop" below.

## Stage 7 — Teardown  ·  invoke `ai-toolkit-remote-teardown`

For a RunPod run, invoke `ai-toolkit-remote-teardown`: `down` does the final
verified pull + terminate + cost report; `rescue` retrieves artifacts from a
self-stopped pod. **Always close with the account-wide billing scan** so the
user knows nothing is left running. (Local runs have nothing to tear down.)

## The v2 loop (when the verdict isn't a clean winner)

"No acceptable checkpoint" or "winner with caveats" from Stage 6 — or a
crash — is the **normal loop, not failure**. Most LoRAs take 1–3 runs; say
so plainly (the Stage 0 draft framing set this up). Teardown always happens
first: the v2 decision never needs the pod.

Map the diagnosis to the re-entry stage, and **carry it forward explicitly**
— the stage skills are stateless, so tell them what v1 did and what failed.
The model brief (`briefs/<project>-brief.md`) is the persistent statement
of intent: v2 re-reads it instead of re-eliciting, and updates it only if
the diagnosis changed a requirement (note what changed and why):

| v1 symptom (reviewer's "Concerns" / crash log) | Re-enter at | v2 change |
|---|---|---|
| Weak trigger / style promptable instead of automatic / bleed | Stage 2–3 | Recaption with v1's leaked words in the avoid list; config usually unchanged |
| Memorization (identical compositions, dataset subjects recurring) | Stage 1 (± 0.5) | Lower rank and/or fewer steps; check dataset variety |
| Style/identity never appears (undertrained at every save) | Stage 1 | More steps and/or higher lr/rank |
| Gibberish text in outputs | Stage 0.5 | Crop the watermarks/text the readiness check missed |
| Crash at startup | per `ai-toolkit-dataset-diagnostics` | Usually config/dataset mechanical fix, then relaunch |

Two rules for the v2 config:

- **Name it `<name>-v2`** so configs, `runs/`, and `output/` don't collide
  with v1.
- **If v2's fix is cleaner captions, the trajectory compresses ~30–40%** —
  v1-tuned step counts and lr will overshoot; drop one or the other, not
  both.

Gate the relaunch like any other launch: state what v1 taught, what v2
changes, and the fresh cost/time estimate, then get the go/no-go.

## Orientation & resume

At the start of every turn while orchestrating, state the current stage and
the immediate next action. If re-entering a run cold:

- Brief exists in `briefs/`? -> past Stage 0.75.
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

`ai-toolkit-model-brief` · `ai-toolkit-lora-config` · `flux2-klein-lora-config`
· `ai-toolkit-gemini-captioner` · `style-vs-content-caption-auditor` ·
`ai-toolkit-dataset-diagnostics` · `dop-class-advisor` · `ai-toolkit-remote-launch`
· `ai-toolkit-remote-monitor` · `ai-toolkit-remote-teardown` · `ai-toolkit-sample-reviewer`

Remote-pipeline reference: `scripts/remote/README.md`. Skill set + activation:
`skills/README.md`.

## Reference files (this skill's own)

- `references/dataset-readiness.md` — the Stage 0.5 curation checklist,
  reflect-back gate, and GO / FIX FIRST / STOP verdicts.
- `references/first-timer-gates.md` — the gate protocol for users new to
  training: concept primers, the Stage 0 cost/time/walk-away expectations
  block, and per-gate scripts (config, audit, launch, pick, close-out).
