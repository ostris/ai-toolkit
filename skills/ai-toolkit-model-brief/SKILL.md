---
name: ai-toolkit-model-brief
description: >
  Brainstorm and capture what a LoRA needs to DO before any model or config
  decision is made, producing a written "model brief" that downstream stages
  consume. Use when: (1) the user is starting a LoRA project and hasn't
  settled what the model is for ("I want to train something on my work",
  "what model should I use", "help me figure out what my model should do"),
  (2) the user says "brainstorm my model", "requirements for my LoRA",
  "model brief", (3) the orchestrator (ai-toolkit-train) reaches Stage 0.75,
  after the dataset readiness check and before config, (4) the user has NO
  dataset yet and wants to know what to collect — the brief doubles as
  dataset-collection guidance. This skill captures the WHAT (capabilities,
  fidelity-vs-flexibility, constraints); it does NOT pick the model or write
  the config — that's ai-toolkit-lora-config, which reads the brief.
---

# AI Toolkit model brief — brainstorm before config

Model and config choices are currently made from three facts (goal type,
license, VRAM). The decisions actually depend on a wider set: does it need
to render text, edit existing images, generalize far beyond the dataset,
run on a specific platform, produce at volume? Nobody can answer those at
config time if nobody asked. This skill is the asking — a short divergent
conversation that converges into a half-page **brief file** the rest of the
pipeline reads.

The brief outlives the conversation. It is consumed at Stage 1 (model +
config + captioning variables), becomes the sample prompts (requirements →
the test suite the run generates evidence for), drives the Stage 6 rubric,
and carries intent into the v2 loop without re-eliciting.

## Entry states

- **Orchestrated (Stage 0.75)** — dataset exists, readiness check done. The
  reflect-back already established what the images *can teach*; this stage
  establishes what the artist *wants* — the gap between the two is the most
  valuable thing this skill can find.
- **Standalone, no dataset yet** — run the same flow; the brief's
  requirements then double as collection guidance ("you want full scenes,
  so the dataset needs scenes, not isolated objects on white"). Route to
  the dataset-readiness check once images exist.

## Step 1 — Dream prompts (the core exercise, always first)

Do NOT open with capability questions — a first-time artist can't answer
"how much prompt adherence do you want?" Instead ask:

> "Imagine the model is done and it's great. Write 3–5 prompts you wish
> you could type, and tell me what comes back for each."

If they struggle, scaffold: "finish this sentence — six months from now I
type ___ and get ___", or offer scenarios (posters for your shows? client
work? turning photos into your style? animations? variations on your
existing pieces?). Get the prompts **verbatim** — they go in the brief and
later become sample prompts.

Then ask for **2–3 stress prompts**: "what's the weirdest thing you (or
your audience) will actually type?" Past models died in production on
exactly these ("a dog riding a dolphin" is the recurring de facto stress
test) because nobody captured them before training — they become sample
prompts too, so generalization failure shows up at step 250, not after
deployment.

**If the user is training on another artist's behalf** (common), get the
artist's ask in the artist's own words — "what exactly did they say they
want people to be able to do?" Several past requirements arrived as
relayed artist complaints after deployment because the brief-equivalent
conversation only ever happened second-hand.

## Step 2 — Infer the axes from the prompts (don't ask what you can read)

Most capability axes are implicit in the dream prompts. Read
`references/capability-axes.md` for the full inference table; the headline
signals:

| Dream-prompt signal | Implication |
|---|---|
| Words/lettering in the imagined output (poster, label, title) | Text rendering is a MUST → hard model filter |
| "Take this photo / my friend's portrait and make it ..." | This is an **edit LoRA** — different model lane entirely |
| Anything moving | Motion LoRA (video models) — different lane |
| Subjects the dataset never showed | Flexibility-leaning dial; generalization matters |
| Their existing work with small variations | Fidelity-leaning dial; high faithfulness wanted |
| "50 of these for a client" / selling work | Commercial license + inference volume/cost matter |
| People/faces central to outputs | Face quality matters (model tier + DOP) |
| "every image should have ___" vs "I want to ask for ___" | Embedded vs promptable ruling for that trait |
| A feel-word keeps recurring ("delicate", "glowy") | Candidate make-or-break ingredient — capture verbatim |

## Step 3 — Ask only the residual axes

Whatever the prompts left ambiguous, ask — **for a first-timer, at most
2–3 questions**, in artist language (the per-axis phrasings are in
`references/capability-axes.md`). Two things must ALWAYS be established,
inferred or asked. The second is the **embedded-vs-promptable walk** (axis
10): take each recurring trait the dataset look surfaced and get a ruling —
automatic every time / only when asked / keep its variety. It is the
most-missed question in this workflow's history and it cannot be changed
after training without a recaption and rerun. The first is the
**fidelity↔flexibility dial**:

> "When you prompt something your images never showed, should the model
> follow the prompt and stretch your style around it — or pull the result
> back toward what your work actually looks like?"

Record the dial as a 1–5 position plus one sentence in the user's own
words. It resolves rank, steps, captioning aggressiveness, DOP, how much
bleed is acceptable, and which checkpoint wins at Stage 6.

Also always capture two constraints (one question each, if unknown):
**commercial use?** and **where will you run it?** (fal / titles.xyz /
local ComfyUI / don't know yet — an unhosted arch can only run locally).

## Step 4 — Reality-check against the dataset

If a dataset exists, hold each must-have against it:

- Text rendering required but no image contains lettering → the LoRA won't
  learn their typography; flag it (model prior carries text, dataset
  carries style — set expectations or collect text examples).
- Dream prompts are full scenes, dataset is isolated objects → route back
  to dataset readiness (Stage 0.5) to collect, or demote the requirement.
- Edit-LoRA intent but no paired/control images → the prep is different;
  flag before config.

A requirement the dataset can't support gets one of two fates, chosen by
the user: **collect more** (back to 0.5) or **demote it** (move to
nice-to-have and note why). Never leave it silently unsupported — that's
the mismatch that surfaces as disappointment at Stage 6.

If no dataset exists, invert: emit a "what to collect" list derived from
the must-haves.

## Step 5 — Write the brief

Fill `references/brief-template.md` and write it to
**`briefs/<project>-brief.md`** in the repo. Then confirm it back in plain
words ("here's what I heard the model needs to do — anything missing?").
The brief should be half a page; if it's longer, it's a transcript, not a
brief.

## Who reads the brief downstream (and what they take)

| Consumer | What it takes |
|---|---|
| `ai-toolkit-lora-config` (Stage 1) | Must-have capabilities as hard model filters (via the capability matrix in its `references/models.md`); the dial → rank/steps/DOP; "what the artist wants to control" → the captioner's ALWAYS-describe list; **dream prompts → sample prompts** |
| `style-vs-content-caption-auditor` (Stage 3) | What's deliberately promptable vs trigger-bound |
| `ai-toolkit-sample-reviewer` (Stage 6) | Scores checkpoints against the brief's must-haves; the dial position weights failure modes (fidelity-leaning tolerates bleed longer; flexibility-leaning treats bleed as disqualifying) |
| The v2 loop | The persistent statement of intent — v2 re-reads it instead of re-eliciting |

## What this skill does NOT do

- **It does not pick the model.** It captures requirements; the
  requirements→model mapping lives in `ai-toolkit-lora-config`'s
  `references/models.md` (capability matrix). Two skills must not fight
  over model selection.
- It does not write config YAML or captioning scripts.
- It is not a long interview. Dream prompts + 2–3 residual questions +
  the two constraint checks. For a fluent user it can compress to one
  turn: "here's the brief I inferred — edits?"
- It does not skip the file. A brief that only exists in chat evaporates;
  every downstream consumer expects `briefs/<project>-brief.md`.

## Related skills

`ai-toolkit-train` (invokes this as Stage 0.75) · `ai-toolkit-lora-config`
(primary consumer) · `ai-toolkit-sample-reviewer` (scores against it) ·
dataset readiness lives in `ai-toolkit-train/references/dataset-readiness.md`.

## Reference files

- `references/capability-axes.md` — every axis: the artist-language
  question, how to infer it from dream prompts, and what it maps to
  downstream (hard model filter vs training lever).
- `references/brief-template.md` — the half-page brief format to fill and
  write to `briefs/<project>-brief.md`.
