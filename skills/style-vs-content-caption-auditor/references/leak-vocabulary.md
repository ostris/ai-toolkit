# Leak vocabulary — per-mode avoid-lists

The audit script's avoid-lists, organized by mode and category. This doc
is the source of truth for what each mode considers "leakage." If you
want to extend or override the script's lists, this is what to base
edits on.

The same families appear in `ai-toolkit-gemini-captioner`'s
`avoid-words-cookbook.md` — the captioner uses these to *prevent*
leakage; this skill uses them to *detect* it. They should stay in sync.

## How leakage works

> If a trait appears in EVERY training image, it is the trigger.
> If a trait varies across images, it is the variable.

Captions describe variables. Words in the avoid-list are words Gemini (or
a human captioner) shouldn't have used. Every word that slips through
becomes promptable instead of automatic, and the trigger weakens
proportionally.

A leak isn't fatal in isolation — one caption mentioning "iridescent"
out of 50 is barely noise. But aggregate leak rate above ~5% means the
trigger has competition for that concept at training time.

## Style mode

For style LoRAs (one visual aesthetic across all training images).
Captions describe content; trigger encodes the aesthetic.

### `medium`
Words for the photographic/illustrative/sculptural medium of the image.
If your dataset is chemigrams that look photographic, you DON'T want
the LoRA to associate "photograph" with the style.

```
photograph, photo, photographic, image, picture, scan, scanned,
painting, illustration, render, rendering, digital, artwork,
art piece, sculpture, drawing
```

### `palette`
Color words. The dataset's palette is part of the style — if every
training image is sepia and you want sepia to fire automatically with
the trigger, captions can't say "sepia."

```
purple, blue, green, teal, magenta, lavender, violet, indigo, brown,
tan, amber, ochre, red, yellow, orange, pink, warm, cool, muted,
desaturated, monochrome, duotone, sepia, iridescent, pearlescent,
opalescent, holographic, metallic, shimmery, shiny, glossy,
prismatic, rainbow, saturated
```

### `vibe`
Mood, era, aesthetic descriptors. Almost always trigger-bound.

```
vintage, antique, aged, old, weathered, retro, period, historical,
dreamy, surreal, ethereal, otherworldly, atmospheric, moody,
evocative, haunting, nostalgic, melancholic, timeless, mysterious,
stylized, aesthetic, vibe, mood, cinematic, whimsical
```

### `quality`
"High quality" / "masterpiece" / "4k" — pollute caption signal even
when the LoRA isn't trying to bind these. Inherited from older
training recipes that don't apply to modern diffusion models.

```
high quality, detailed, masterpiece, 4k, 8k, hyperdetailed,
intricate, ultra realistic, photorealistic, professional, beautiful,
striking, delicate, fragile, elegant, dramatic, stunning, gorgeous
```

### `hedging`
"Appears to be" / "looks like" — uncertainty leaks into Gemini-
generated captions. Doesn't directly hurt the LoRA but pollutes the
signal.

```
appears to be, seems like, looks like, as if, kind of, sort of,
appears, seems
```

## Character mode

For character LoRAs (one identity across all training images).
Captions describe what varies (clothing, pose, setting, style/medium);
trigger encodes the fixed identity (face, eyes, hair, skin, build,
demographics).

### `face`
Facial features. Identity-bound — never describe.

```
face, facial, jawline, cheekbones, cheeks, chin, nose, lips, mouth,
forehead, eyebrows, eyelashes, expression line
```

### `eyes`
Eye color and shape.

```
eye color, blue eyes, brown eyes, green eyes, hazel eyes, gray eyes,
dark eyes, almond shaped, round eyes, narrow eyes
```

### `hair`
Hair color and type. (Style — bun, ponytail, etc. — is OK if it
varies across the dataset.)

```
blonde, brunette, redhead, black hair, brown hair, blonde hair,
red hair, gray hair, white hair, long hair, short hair, curly hair,
straight hair, wavy hair
```

### `skin`
Skin tone, texture, marks.

```
skin tone, complexion, fair skin, pale skin, dark skin, tan skin,
olive skin, freckles, freckled, moles, scars
```

### `body`
Body type, height, build.

```
tall, short, slim, slender, petite, muscular, athletic, stocky,
thin, thick, heavyset, build
```

### `demographic`
Age and gender. The trigger word carries identity — describing
"woman" or "young" makes those promptable.

```
young, youthful, middle-aged, elderly, old, teenager, child, adult,
woman, man, girl, boy, lady, gentleman, female, male
```

### `ethnicity`
Demographic descriptors.

```
asian, european, african, latino, latina, hispanic, caucasian,
middle eastern, south asian, east asian
```

## Motion mode

For motion LoRAs (one transformation/motion arc across all training
clips). Captions describe the static first-frame subject; trigger
encodes the motion.

### `motion_verbs`
Every motion-verb family the dataset exhibits, in every tense. The
audit's default list covers melt/slump/drip, spread/merge/fuse,
transform/morph/change families. Extend with `--extra-avoid` for
motions specific to your dataset (e.g., "rotate", "bounce",
"pulsate").

```
melt, melting, melted, slump, slumping, slumped, droop, drooping,
drooped, sag, sagging, sagged, deflate, deflating, deflated,
collapse, collapsing, collapsed, drip, dripping, dripped, flow,
flowing, flowed, ooze, oozing, oozed, pour, pouring, fall, falling,
spread, spreading, fuse, fusing, fused, merge, merging, merged,
blend, blending, blended, combine, combining, combined, join,
joining, joined, unite, uniting, transform, transforming,
transformed, transformation, morph, morphing, morphed,
morphological, change, changing, changed, evolve, evolving, evolved,
shift, shifting, shifted, transition, transitioning, become,
becoming, became, turn into, turning into, develop, developing,
developed, motion, moving, moves, moved, movement, animate,
animated, animation
```

### `time_evolution`
Time-passing words. Even without a motion verb, "gradually" implies
motion.

```
time-lapse, timelapse, time lapse, gradual, gradually, slowly, slow,
eventually, after, before, during, over time, throughout, begin,
beginning, begins, started, starts, end, ending, ends, finished,
progress, progression, progressing, sequence
```

### `anticipatory`
The sneaky one. "About to melt" doesn't have "melting" but encodes
the motion as anticipation. Forbidden.

```
about to, ready to, on the verge of, set to, poised to, preparing to
```

### `video_meta`
Video-as-artifact language. Describe the SUBJECT, not the artifact.

```
video, clip, footage, scene, shot, take, frame, still
```

### `vibe`
Same as style mode — bleeds into motion captions too.

```
atmospheric, ethereal, dreamy, moody, abstract artwork, beautiful,
striking, interesting
```

## Combined mode

The audit's combined mode merges style + character lists. Combined
LoRAs (recurring character archetype + recurring visual signature)
have leakage in both axes; the user interprets the report based on
training intent.

The Wan22 style+motion preset is a different "combined" — that one
encodes motion + visual. For style+motion combined LoRAs, run the
audit in `motion` mode (since the visual is intentionally bound to
the trigger and shouldn't be flagged).

## Extending the avoid-list per dataset

Use `--extra-avoid` to add dataset-specific words from prior runs
that bled. Examples from past projects:

- chemigram: `vein,branching,translucent,leaf-skeleton,vascular`
- VCRIBB: `glitter,rhinestone,beaded,sequined,pearl,dollhouse,kitsch`
- gr4r0cks: `coral,pink-coral,warm-coral,cream-warm`

The script reports these as a `dataset_specific_leaks` category in the
output. If `--extra-avoid` words show up in many captions, your
captioner's avoid-list was incomplete — fix it before recaptioning.

## What this list canNOT detect

- **Semantic leakage** — "photographic-looking" without literal
  "photograph". A human spot-check of 5-10 random captions catches
  these.
- **Style-as-subject overlap** — for chemigram-like LoRAs where leaf
  veins are style-bound but leaves are content-bound, "vein" mentions
  could be either. The audit can't disambiguate; you have to read the
  captions.
- **Composition leakage** — every caption describes "centered on
  white" looks fine per-caption but is leakage in aggregate.

For these, the audit's printed sample captions are the diagnostic —
read them.
