# Avoid-words cookbook

The avoid list is the single highest-leverage decision in the captioner. It
controls what the LoRA learns implicitly (via omission) vs. what stays
user-promptable. This doc is the discipline for building a complete one.

## The core principle

> If a trait appears in EVERY training image, it is the trigger.
> If a trait varies across images, it is the variable.

Captions must describe variables and never describe trigger-bound traits.
Words in the avoid list are the words Gemini must not use. Every word that
slips through becomes promptable instead of automatic, and the trigger
weakens proportionally.

## Build the list in passes

### Pass 1 — name the style in 6–10 words

Force the user to articulate it. "Chemigram darkroom prints with iridescent
chemistry, leaf photograms on dark paper" → that one sentence already names
six avoid-list categories: medium, palette, technique, surface, substrate,
subject-as-style.

### Pass 2 — list every distinctive trait

For each trait, list 4–8 synonyms. Synonyms matter because Gemini will
substitute them when blocked. Examples:

| Trait | Synonyms (all must be in the avoid list) |
|-------|-----------------------------------------|
| iridescent | iridescent, pearlescent, opalescent, holographic, shimmery, prismatic, rainbow |
| sepia-toned | sepia, sepia-toned, toned, monochrome, duotone, brown-toned, warm |
| vintage feel | vintage, antique, aged, old, weathered, retro, period, historical |
| dreamy | dreamy, surreal, ethereal, otherworldly, atmospheric, moody, evocative |
| oily/wet surface | oily, oil-slick, glistening, glossy, wet, shiny |
| photographic medium | photograph, photo, photographic, image, picture, scan, scanned |

### Pass 3 — include color words if color is part of the style

If the style has a recognizable palette, list every color word:

```
purple, blue, green, teal, magenta, lavender, violet, indigo, brown, tan,
amber, ochre, warm, cool, muted, desaturated
```

If the user later wants to prompt "a red dog, <trigger>", Gemini must
not have associated colors with the trigger. The LoRA should preserve
trigger-style regardless of subject color, and that requires colors to
have stayed out of training captions entirely.

### Pass 4 — include style/medium meta-vocabulary

Always include:

```
stylized, aesthetic, vibe, mood, atmospheric, cinematic, beautiful,
striking, delicate, fragile, elegant, dramatic, evocative, haunting
```

Plus medium words that don't apply to the dataset:

```
photograph, photo, illustration, render, rendering, digital, painting,
artwork, art piece, sculpture
```

(Include only the ones you don't want — if the dataset IS photographs,
keep `photograph` describable as content; if it's chemigrams that look
photographic, exclude `photograph` to keep "photographic" out of captions.)

### Pass 5 — include hedging and quality words

```
appears to be, seems like, looks like, as if,
high quality, detailed, masterpiece, 4k, hyperdetailed, intricate
```

These don't hurt the LoRA directly but they pollute caption signal.

## Special cases

### Style-as-subject overlap (chemigram trap)

If the SUBJECT and the STYLE share visible features, you have to be
surgical. Chemigram example: leaves are the subject (variable), but
**leaf veins** are the style (binds to trigger — the user wants vein-
patterning to fire automatically on a "dog, 1ll6m3ns" prompt). The
captioner must describe leaves and never describe veins.

Add the shared-feature words to the avoid list explicitly:

```
vein, veins, leaf veins, vein pattern, vein structure,
veined, veining, branching veins, vascular pattern, network of veins,
translucent, x-ray, see-through, internal structure
```

### Composition-as-style trap (Klein edit-mode failure)

If every training image has the same composition (object centered on
white), the LoRA will overfit to that composition. Captions must describe
composition explicitly so it stays variable:

```
ALWAYS describe: centered, off-center, edge-cropped, full-frame,
                  asymmetric, two-subject layout, scattered, etc.
```

This is a positive instruction in the SYSTEM_PROMPT, not an avoid-list
entry. But the failure pattern is the same: trait shared across all
images = bound to trigger unless explicitly described.

### Motion LoRAs

Motion datasets need three avoid-list categories that style datasets don't:

1. **Motion verbs.** Every verb family the dataset exhibits, in every
   tense: `melt, melting, melted`. Cover the family — `slump, slumping,
   slumped, droop, drooping, drooped, sag, sagging, sagged`.
2. **Time-evolution words.** `gradually, slowly, eventually, throughout,
   over time, during, after, before, begin, end, progress, sequence`.
3. **Anticipatory phrases.** `about to, ready to, on the verge of,
   set to, poised to`. These are how Gemini sneaks motion past the
   verb filter.
4. **Video meta-language.** `video, clip, footage, scene, shot, take,
   frame, still`. Describe the SUBJECT, not the artifact.

### Character LoRAs

Character avoid lists are short and structural — not stylistic:

```
face, jawline, cheekbones, eye color, eye shape, hair color, hair type,
skin tone, complexion, freckles, moles, scars, body type, height, build,
young, middle-aged, old, woman, man, girl, boy, ethnicity
```

The trigger word carries identity. Don't describe identity, the LoRA
will learn the face-shape-and-features whenever the trigger fires.

## When prior runs leaked

If the user says "the v1 LoRA came out tinted purple even though I
prompted blue subjects," look at the v1 captions. You'll find Gemini
mentioned the chemistry color in 30%+ of them. The fix is:

1. Add the leaked words and their synonyms to the avoid list.
2. Add an explicit "the previous run leaked X — never describe color
   tone, palette, or hue" section to the SYSTEM_PROMPT.
3. Add 1–2 new BAD examples to the prompt that show the leakage and
   call it out.
4. Recaption with `--overwrite` and retrain.

Document the leakage in the script's docstring with `# leaked in v1`
comments so the next dataset's captioner inherits the lesson.

## Sanity check before training

Before training, grep the captions:

```bash
grep -i 'sepia\|iridescent\|<other-avoid-words>' "$DATASET"/*.txt
```

Any hits → re-edit the avoid list and recaption. The 5 minutes here
saves a 4-hour failed training run.
