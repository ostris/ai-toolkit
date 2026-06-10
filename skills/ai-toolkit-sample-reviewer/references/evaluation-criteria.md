# Evaluation Criteria by LoRA Type

The questions you ask of a checkpoint depend on what was trained. Use the rubric that matches.

## Common to all types

### Fidelity floor

Look at trigger prompts at every step. When does the trained behavior **first appear**? This is the floor.

- Style LoRA: first time the trained aesthetic shows up visibly on a triggered prompt (not just a hint — clearly the style).
- Character LoRA: first time the character's identity is recognizable.
- Combined: first time both style and identity coexist in one image.

Below the floor, the LoRA isn't doing anything yet. Above it, you have a real candidate.

### Overfitting ceiling

Look for these signs (they often appear together):

1. **Compositional repetition** — multiple different prompts produce nearly identical compositions. The LoRA has collapsed to "draw the dataset" instead of "render the prompt."
2. **Vocabulary memorization** — text-in-image (when the dataset has text) becomes specific recurring strings rather than appropriate-to-the-prompt strings.
3. **Subject leakage** — recurring objects from the dataset appear in prompts that didn't ask for them. (E.g., the bust from image 12 shows up in a "red bicycle" prompt.)
4. **Anatomy/structure collapse** — character LoRAs start producing the same pose, same face angle, same expression regardless of prompt.
5. **Generalization drop** — a prompt for a subject *not* in the dataset (e.g. "[trigger] a goldfish" when there are no fish in training) starts looking like a dataset subject rather than a goldfish in the style.

The ceiling is the step where these begin. Past the ceiling, every additional step makes things worse.

### Control-prompt bleed

Look at non-triggered prompts (no trigger word in the prompt) at every step. They should look like the base model.

Bleed signs:
- Subtle palette shift toward the dataset's palette
- Texture/grain showing up where it shouldn't
- Compositional habits of the dataset appearing
- Subjects from the dataset appearing unprompted

Bleed is the most reliable overfitting indicator on style LoRAs. A LoRA can look "great" on triggered prompts and be useless because it's also altering non-triggered prompts (i.e., it's just biasing the base model rather than binding a style to a trigger).

Note: if `train.diff_output_preservation: true` was set, bleed should be near-zero by design. If it isn't, DOP isn't helping and something else is wrong.

### Honoring artist intent

This is the highest-leverage criterion and the most-missed. Re-read the YAML comments and the dataset before scoring. If the artist or config author specified preferences:

- "Texture/grain/imperfection welcome" → score down clean-looking outputs even if technically faithful
- "Bright saturated color" → score down desaturated outputs
- "Realistic elements welcome" → score down purely abstract outputs
- "Character should be flexible across styles" → score up checkpoints where generalization holds

A "technically faithful but spiritually wrong" checkpoint is not the winner.

---

## Style LoRA rubric

Goal: trigger applies an aesthetic to any subject.

### Score on:

1. **Style transfer breadth** — does the style show up across diverse prompts (portrait / landscape / object / scene), or only on a narrow content type?
2. **Style fidelity to dataset** — does the visible aesthetic actually match the dataset, or is it the base model's idea of what the style words mean?
3. **Subject preservation** — does the trigger keep the prompted subject recognizable? "A bicycle in the style" must still be a bicycle.
4. **Texture/material fidelity** — for styles with strong texture (grain, halftone, brushwork, glitch), does the texture appear? This is usually what late checkpoints get right and early ones miss.
5. **Bleed (control prompts)** — see common section.

### Pattern to look for: the "feeling" gap

Sometimes a checkpoint reproduces every individual element of the dataset (colors, shapes, motifs) but the result doesn't *feel* like the dataset. This usually means the LoRA learned the vocabulary but missed the syntax — composition, weighting, restraint. A checkpoint that gets the feeling right with fewer literal matches is the winner.

---

## Character LoRA rubric

Goal: trigger generates the specific person across many styles/contexts.

### Score on:

1. **Identity recognizability** — would someone who knows the person recognize them in these outputs?
2. **Identity stability across styles** — try the character in different style prompts ("[trigger] in renaissance painting style", "[trigger] anime", etc.). Identity should survive style changes.
3. **Identity stability across angles/lighting** — same person from different angles, in different lighting, still the same person.
4. **Style flexibility** — does the LoRA *allow* the prompted style, or does it force one rendering?
5. **Bleed (control prompts)** — see common section.
6. **No memorized poses/clothing** — the LoRA should know the *person*, not the dataset's specific outfits. If the character keeps wearing the same shirt unprompted, the captioning under-described clothing.

### Watch for:

- **Style baking** — character only renders in the dataset's medium (e.g. only photographic). This is captioning leakage — clothing/setting/medium got memorized as identity.
- **Demographic drift** — character looks like a generic "person who looks vaguely like X" rather than the specific person. Usually means under-training; bump steps.

---

## Combined character + style rubric

Goal: trigger generates the character AND the style; different style descriptors in captions activate different aesthetics.

### Score on:

1. **Both modes activate** — the identity-set style descriptor produces one look, the target style descriptor produces a different look, and identity holds in both.
2. **Identity stability across both styles** — same person in both.
3. **Style separation** — the two styles are *visibly distinct*. If they look identical, the LoRA didn't learn the separation (usually because the two style descriptors in captions were too similar).
4. **Generalization to unseen styles** — prompting the character in a style never captioned (e.g. "[trigger] watercolor") should produce a recognizable character in that style.
5. **Bleed (control prompts)** — see common section.

### Watch for:

- **Style collision** — both descriptors produce the same look. Caption issue, not training issue.
- **Identity-only learning** — both descriptors produce the character but only in one style. The style descriptors didn't bind.
- **One-way bind** — style A works, style B doesn't (or vice versa). Often a dataset-balance issue; `num_repeats` may need adjusting on the smaller set.

---

## Scoring approach

For each candidate checkpoint, assign per-category scores: **Strong / Adequate / Weak / Broken**.

Don't use numbers. The point isn't precision — it's that "Strong on fidelity, Weak on bleed, Adequate on generalization" tells the user where the checkpoint fits in their workflow. They'll combine that with their own intent.

The **winner** is the checkpoint with the best combination given the user's intent — not the highest score across the board. If the artist said "weird artifacts welcome," a checkpoint that's Strong on fidelity + Adequate on bleed beats one that's Strong on both but looks too clean.
