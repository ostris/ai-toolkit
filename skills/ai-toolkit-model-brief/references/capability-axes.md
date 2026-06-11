# Capability axes

Each axis: what it decides downstream, how to infer it from the dream
prompts, and the artist-language question for when inference fails. Axes
marked **HARD** filter the model choice outright; the rest are training
levers (rank/steps/captions/DOP) or constraints.

**Question budget:** for a first-timer, ask at most 2–3 of these
explicitly — the dream prompts should answer the rest. Two things must
ALWAYS be established: the dial (axis 1) and the embedded-vs-promptable
walk (axis 10 — it's one compound question driven by the dataset look, not
a questionnaire). Axes 6 and 7 are one-line constraint checks, not
brainstorm questions.

> Axes 10–13 were mined from every past run's post-mortems: each one is a
> question whose absence cost at least one full paid training run.

---

## 1. Fidelity ↔ flexibility dial (ALWAYS establish; the centerpiece)

- **Decides:** rank (fidelity → higher), steps tolerance, captioning
  aggressiveness, DOP on/off, acceptable control-prompt bleed, and the
  Stage 6 checkpoint policy (a fidelity-leaning artist's "overcooked" is a
  flexibility-leaning artist's "finally committed").
- **Infer:** dream prompts that are mostly the artist's existing work with
  small variations → fidelity-leaning (4–5). Prompts full of subjects the
  dataset never showed → flexibility-leaning (1–2).
- **Ask:** "When you prompt something your images never showed, should the
  model follow the prompt and stretch your style around it — or pull the
  result back toward what your work actually looks like?"
- **Record:** position 1 (follow the prompt) to 5 (stay close to my work),
  plus one sentence in the user's own words.

## 2. Text rendering — **HARD**

- **Decides:** model choice (strong-text models vs the rest); whether the
  dataset needs lettering examples; sample prompts must include a text
  test if MUST.
- **Infer:** any dream prompt whose output contains words — posters,
  titles, labels, lyrics, signage.
- **Ask:** "Will the images the model makes ever need readable words in
  them — titles, posters, lettering?"
- **Note:** if the artist's *own typography* is part of the style, that's
  a dataset + captioning question too (text content stays promptable,
  letterform treatment binds to the trigger).

## 3. Generate vs edit — **HARD** (lane decision)

- **Decides:** the entire model lane. "Type a prompt, get an image" →
  text-to-image LoRA. "Take an existing image and transform it" → edit
  LoRA (Qwen-Image-Edit, Kontext, Klein ctrl_img) with different dataset
  prep (control/target pairs) and different failure modes.
- **Infer:** dream prompts phrased as "take this photo / my sketch / a
  portrait and make it ..." are edit intent, even when the user calls it
  generation.
- **Ask:** "Will you mostly type prompts to create images from scratch, or
  hand it an existing image to transform?" (Both is possible — some bases
  do edit natively — but name the primary.)
- **If edit, ALWAYS nail the preservation contract** (a full t2i run has
  been wasted on this twice, and one edit project burned four runs on it):
  1. **Overlay or transform?** "Should the result be the person's photo
     with your effect added on top, or your full reinterpretation of it?"
     These are different training approaches, not strength settings.
  2. **What must stay exactly as uploaded?** Face, skin texture, hair,
     identity, background. Full-image diffusion edits re-render the whole
     frame — a hard untouched-region guarantee means planning region
     compositing from day one, not after the artist complains about
     plastic skin.
  3. **What will people actually feed it?** Tight selfies only, or full-
     body/action shots too? Generalization is framing-dependent; an
     all-tight-crop dataset fails on everything else.

## 4. Still vs motion — **HARD** (lane decision)

- **Decides:** video models (Wan family) vs image models; clip dataset
  prep; motion captioning discipline.
- **Infer:** any dream output that moves.
- **Ask:** only if ambiguous: "Still images, or moving?"

## 5. Subject breadth & people

- **Decides:** generalization expectations → rank (broad → lower,
  caption discipline stricter), dataset variety requirements, DOP class;
  if faces are central, model face quality matters and DOP is usually on.
- **Infer:** list the subjects across all dream prompts; compare to the
  dataset's subjects. Faces appearing in prompts but not the dataset is a
  flag.
- **Ask:** "What kinds of things will you actually ask it for — mostly
  the subjects in your images, or anything you can think of?"

## 6. Commercial use — **HARD** (one-line constraint check)

- **Decides:** license filter (rules out the non-commercial bases —
  LoRAs inherit the base license).
- **Ask:** "Will you sell or commercially publish what this model makes?"
  Don't explain license law; just record yes/no/unsure (treat unsure as
  yes — it only narrows the menu, never hurts quality much).

## 7. Inference destination — **HARD-ish** (one-line constraint check)

- **Decides:** the trained arch must be hosted where the user will run
  it — an unhosted arch can only run locally. Also platform conventions
  (e.g. hosted runtimes often need higher LoRA strength than the trainer's
  1.0 — note the destination in the brief so the use-your-model stage can
  calibrate).
- **Ask:** "Where do you picture running this — a website/service, or
  software on your own machine, or you don't know yet?" "Don't know" is a
  fine answer; record it and prefer widely-hosted archs.
- **If known, record the platform's runtime knobs** — negative prompts
  available? per-prompt editing, or a fixed assisted prompt around user
  input? strength slider exposed? Inference-side mitigations (gibberish-
  text negatives, strength tuning) get designed during training, and twice
  they turned out to be impossible on the actual platform (a fixed
  assisted-prompt system; a turbo model with no negative-prompt support).
  Knowing the knobs up front moves those fixes into the training plan.

## 8. Volume / speed / cost at inference

- **Decides:** turbo/distilled tier vs full models. Cross-reference the
  texture-fidelity tiers in `ai-toolkit-lora-config`'s
  `references/models.md` — distillation costs high-frequency texture, so
  this axis TRADES against axis 9. Never resolve this trade silently;
  if both matter, say so and let the user pick.
- **Infer:** "50 of these", "every product shot", client/volume language.
- **Ask:** only when volume language appeared: "Do you need lots of
  images fast and cheap, or is a slower, better one fine?"

## 9. Texture fidelity

- **Decides:** model tier (tier-1 base models vs turbo) — see the tier
  table in `references/models.md`.
- **Infer:** usually from the DATASET, not the prompts — the Stage 0.5
  look already saw whether grain/halftone/brushwork/weave is the point of
  the work. If surface texture is the style, this is a MUST and turbo
  models are out.
- **Ask:** rarely needed; if unsure: "If the images came back with your
  compositions and colors but smoother surfaces — less grain, less of the
  physical texture — would that still be your work?"

## 10. Embedded vs promptable — ALWAYS establish (the most-missed question in project history)

- **Decides:** the entire captioning strategy (omit-to-bind vs
  describe-to-control vs describe-to-preserve-variety), and it cannot be
  changed after training without a full recaption + rerun. This question
  arrived late in at least six past projects (recurring shapes, palette,
  collage panels, flowers, human figures, material variants) and each time
  cost a retrain.
- **How to run it:** take the recurring/uniform traits observed in the
  Stage 0.5 dataset look (signature motifs, palette, a layout system,
  material treatments, the fact that every image contains X) and walk them
  in ONE compound question: "Your work always/often has ⟨trait⟩ — when
  someone uses your model, should that show up **automatically every
  time**, appear **only when they ask for it**, or **keep its natural
  variety** (different every generation)?"
- **Three possible answers per trait, three different caption rules:**
  - **Automatic** → omit from captions (binds to trigger). Warn about the
    known cost: omitted nouns become back-door triggers.
  - **Only when asked** → name it consistently in the caption body.
  - **Keep the variety / user-pickable** → describe it per-image so it
    stays a conditioned variable; otherwise it AVERAGES to the dataset's
    dominant mode (the color-flattening failure).
- **Watch for the hidden second goal:** "is this just the look, or also a
  recurring character/motif that should come along with it?" — a style
  LoRA quietly carrying a character is a scope change, not a tweak.

## 11. Same prompt twice: consistent or surprising?

- **Decides:** whether variety is part of the product. "Trigger-alone
  variety" (same prompt, different variant each generation) inverts the
  normal training shape — variety peaks in the FIRST third of training and
  then mode-collapses to the dominant dataset mode, so checkpoint policy,
  step budget, and dataset balance all flip. One project trained v2 and v3
  against "variants are promptable," then redefined the goal to "variants
  happen automatically" — a fourth run.
- **Ask:** "If two people type the exact same prompt, should they get the
  same signature look — or different variants from your range each time?"
- Only ask when the dataset has visibly distinct modes/variants; for a
  single-look dataset the answer is trivially "consistent."

## 12. Bleed tolerance: does the style need an off switch?

- **Decides:** the checkpoint rubric (twice the entire evaluation changed
  when the user finally said "I don't care about trigger bleed"), the
  anchor strategy, and whether late strong checkpoints are eligible.
- **Ask:** "Will this model ever run alongside other styles or other
  prompts where it must stay completely OFF unless invoked — or is it fine
  (even good) if your look sneaks in everywhere?"
- **Map:** must-stay-off → bleed is disqualifying; earlier checkpoints,
  vocabulary-disjoint control prompts, document back-door words. Always-on
  (e.g. deployed behind a fixed assisted prompt that always includes the
  trigger) → bleed is irrelevant; pick on fidelity alone.

## 13. The make-or-break ingredient (the acceptance criterion)

- **Decides:** what the sample reviewer scores FIRST at every checkpoint,
  and a dataset check that has predicted failure before: omission only
  binds a consistently-present feature, so if the make-or-break ingredient
  appears in under ~half the dataset, it will NOT bind — fix the dataset
  before training, don't hope.
- **Ask:** "Name the one ingredient that, if the model loses it, makes the
  whole thing a failure — and give me 2–3 plain words for the feel of your
  work (delicate? bold? rough? glowing?)." Past acceptance criteria that
  went unstated until after training: "delicate, not big and bold," "the
  jewel-tone glow," "the digital glitch," "the multi-panel collage look."
- **Record verbatim** in the brief. If the user is training on another
  artist's behalf, get the *artist's* words, not a paraphrase — several
  past requirements arrived as relayed artist feedback after deployment,
  the most expensive possible time.
- **Cross-check against the dataset immediately:** is the ingredient
  consistently present? Is the named register (delicate/bold) actually
  what the images show? A model cannot produce a register that isn't in
  its training data.

---

## Inference quick-table (dream-prompt signal → axis)

| Signal in a dream prompt | Axis | Default reading |
|---|---|---|
| Output contains words/lettering | 2 | text = MUST |
| "take this image / photo of X and ..." | 3 | edit lane |
| motion verbs / "animation", "loop" | 4 | motion lane |
| subjects absent from dataset | 1, 5 | flexibility-leaning, broad |
| variations on existing pieces | 1 | fidelity-leaning |
| client / selling / volume language | 6, 8 | commercial + volume |
| people/faces central | 5 | face quality + DOP |
| named platform ("on titles", "on fal", "in ComfyUI") | 7 | record verbatim |
| "it should always look like ___" / "every image should have ___" | 10 | embedded |
| "I want to be able to ask for ___" | 10 | promptable |
| "sometimes ___, sometimes ___" about their own range | 11 | variety question |
| "stacking with other LoRAs" / "merged" / fixed assisted prompt | 12 | bleed tolerance |
| a feel-word repeated ("delicate", "glowy", "gritty") | 13 | make-or-break candidate |

## Severity language for the brief

Rank each captured capability as:

- **MUST** — the model is a failure without it. Hard filters + a dedicated
  sample prompt to test it.
- **NICE** — want it, wouldn't kill the project. Influences tie-breaks.
- **DON'T CARE** — explicitly out of scope. Recording these prevents
  re-litigating them at every later gate.
