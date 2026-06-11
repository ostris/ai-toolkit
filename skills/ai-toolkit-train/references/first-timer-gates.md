# Running the gates with a first-time trainer

Most people entering this workflow have never trained a model and have no
ML background. The gates only protect them if they can actually evaluate
what they're approving. This file is the translation layer: the protocol
for how every gate is presented, the expectation numbers for Stage 0, and
a per-gate script.

For an experienced user (anyone fluent in rank/steps/leak rates), skip the
scripts and run the gates in shorthand as the stage sections describe. Calibrate once at Stage 0 — if they ask "what's a
LoRA?", they're a first-timer for every gate that follows.

## The gate protocol

Every gate, no exceptions:

1. **Lead with the recommendation.** "Recommended: X" comes first, then the
   explanation, then the question. Never open with a menu.
2. **Outcome language only.** Every number gets translated into what it
   means for *their result*. "Rank 32" is meaningless; "enough capacity to
   learn your texture, low enough not to memorize your 20 exact images" is
   evaluable.
3. **At most ~3 decisions per gate.** You make every other call silently
   with the defaults. A first-timer asked to approve ten things approves
   all ten without reading — which protects nothing.
4. **Define each concept once, in one or two sentences, at first use.**
   Then use it freely. The primers below are the canonical first-use
   definitions.
5. **A rubber stamp is a gate failure.** If the user says "uh, sure, I
   guess?" the gate didn't work — shrink it to one decision and re-ask.

## Concept primers (use at first mention, roughly verbatim)

- **Trigger word** — "A made-up word, like `k3llio`, that we attach your
  style to during training. After training, typing it in a prompt switches
  your style on; leaving it out gives you the normal model. It's deliberately
  a non-word so it can't collide with anything the model already knows."
- **Steps** — "How long the model studies your images. Too few and it
  hasn't learned the style; too many and it memorizes your exact images
  instead. We don't have to guess: we save snapshots along the way and pick
  the best one afterward."
- **Checkpoint / snapshot** — "A saved copy of the model partway through
  training. Quality doesn't just keep climbing — it peaks somewhere in the
  middle and then degrades — so we keep all snapshots and choose the peak."
- **Samples** — "Test images the run generates every few hundred steps from
  fixed prompts, so we can watch the style taking hold. The prompts are
  deliberately plain — if they described your style in words, we couldn't
  tell whether the model learned anything."
- **Caption leakage** — "Each image gets a text description. The rule:
  anything the captions *describe* stays a knob you have to type at prompt
  time; anything they *omit* gets baked into your trigger word. So if the
  captions mention your style, the style never binds to the trigger and the
  model feels like it didn't learn."
- **Warming** — "Before step 1, the cloud machine downloads the base model
  — 20–60 minutes of apparent silence on a first run. Normal, not stuck."

## Stage 0 — the expectations block (give this before asking to proceed)

Three lines, always, before the user commits to the flow:

- **Money**: "Captioning costs under $1. The GPU rental is the real cost:
  typically **$3–10 for a full run** (~$0.50–1.50/hr on the cards we
  usually use, for 2–6 hours). Nothing is billed until you approve the
  launch, and I'll show you the exact estimate at that gate."
- **Time**: "Prep (captioning + checks) is ~15–30 minutes of back-and-forth
  now. Training itself is **2–6 hours**, plus 20–60 minutes of model
  download on a first run. Reviewing the results takes a few minutes at the
  end."
- **You can walk away**: "Once training is launched you don't need to stay.
  Your laptop can sleep or close — the cloud machine shuts **itself** down
  when it finishes (with a hard time limit as a backstop), and we can
  reconnect to check on it any time."

And one line of framing that prevents the most common emotional faceplant:

- **First runs are drafts**: "A first training run usually teaches us
  something to adjust — captions, length, dataset. Plan on it taking 1–3
  runs to get the model you're imagining. That's the normal loop, not
  failure."

## Gate scripts

### Stage 0.5 — dataset readiness

Use the reflect-back from `references/dataset-readiness.md`: describe in
plain words what the model will learn vs. what stays promptable, list any
FIX items concretely, then ask: *"Is that the thing you want the model to
learn?"* This is the one gate where the first-timer is the expert — give it
room.

### Stage 1 — config approval

Present **a short table, not the YAML**. Three to five rows, each in
outcome terms, recommendation pre-filled:

| Decision | Recommended | What it means for your result |
|---|---|---|
| Base model | (name) | "The model we're teaching. Chosen because (one clause: quality / license lets you sell work made with it / fits the budget GPU)" |
| Trigger word | (token) | primer above, plus: "you'll type this in every prompt that should use your style" |
| Training length | (N steps) | "~X hours; we save a snapshot every (save_every) so we can pick the peak afterward" |
| Test prompts | (count) | "what the run generates along the way so we can watch it learn — deliberately plain" |

Rank, alpha, optimizer, buckets, DOP, quantization: decide silently with
the stage skill's defaults. Mention them only if asked, or if something
about this dataset forced a non-default choice worth flagging.

Close with: **"My recommendation is to run it exactly like this. Anything
you'd like different before we caption?"**

### Stage 3 — caption audit

Translate the leak rate before the verdict: *"X of your N captions
accidentally describe the style itself (words like '…'). Left alone, those
parts of your style stay manual — you'd have to type them every time and
the trigger word stays weak."* Then the recommendation straight from the
auditor's rubric: clean → "captions are clean, ready to train"; spot-fix →
"I'll fix the N flagged files, takes a minute"; recaption → "faster to
regenerate all captions with a stricter script than to hand-fix — I'll do
that and re-check." Ask only when there's a real choice (borderline
spot-fix vs. recaption); otherwise state the action and do it.

### Stage 4 — launch (the money gate)

This is the only gate where real money starts. Always include, in order:

1. **The number**: "Recommended: (cheapest GPU that fits this model) at
   ~$(rate)/hr — about **$(total) total** for this run, done in ~(hours)h.
   If the wait matters, (faster card) costs ~(2×rate)/hr and finishes
   ~1.5–2× sooner — similar total cost." For a first-timer the recommended
   default is the **cheapest card that fits the model's VRAM floor** (the
   launch skill has the fit table); offer the faster card as the
   alternative rather than the default.
2. **The control**: "Nothing is billed until you say go. If anything fails
   before training starts, the machine tears itself down."
3. **The walk-away reminder**: "After launch you can close your laptop —
   I'll check on it and bring back samples as they appear."
4. Preflight results in one line ("config and dataset validated, N images +
   N captions confirmed") — not the raw remap dump.

Then: **"Want me to launch?"** — and wait.

### Stage 6 — checkpoint pick

The reviewer's recommendation is the default; the user's job is to look at
images, not to weigh trajectories. Present:

1. One line of frame: "Training saved N snapshots. Quality peaks and then
   degrades, so I compared all of them against your originals."
2. **The winner, with images**: show 2–3 winner samples next to 2–3 dataset
   images. "Recommended: step X — (one sentence why, in visual terms)."
3. The runner-up with 1–2 images and when they'd prefer it ("step Y keeps
   more texture but drifts off your palette").
4. Ask: **"Go with step X?"** Their eyes are the final gate — if they like
   the runner-up better, that's the pick; their taste outranks the rubric.

Never hand a first-timer a bare list of checkpoint numbers and scores.

### Stage 7 — close-out

End with the two sentences they actually need: "The cloud machine is
terminated and an account-wide scan shows **nothing is still billing**.
Total cost of this run: $(X)." Then where the model file lives and what its
trigger word is — written down, because they will forget it.

## If something goes wrong

Two scripts, both leading with the money status — that's the first thing a
first-timer silently worries about:

- **Crash**: "Training hit an error. I've saved the error log and **shut
  the machine down — nothing is billing**. (One sentence on the cause in
  plain terms.) Here's what we change and what a retry costs." Never leave
  the pod running while a first-timer decides what to do; with them, the
  teardown-first rule in the monitor skill is absolute, not a judgment
  call.
- **"It didn't learn it" (no acceptable checkpoint)**: "The run finished
  and nothing is billing, but none of the snapshots are good enough to
  keep — this happens, and it's round one doing its job. Here's what this
  run taught us: (the reviewer's diagnosis in plain terms). For v2 we
  change (one thing), which costs another ~$(X) and (Y) hours. Want to go
  again?" Frame v2 as the plan working, not as starting over — the
  captions/dataset/learnings all carry forward.
