# Captioning Strategy

## Core rule

**The LoRA learns what you don't caption.** Whatever you omit from captions becomes baked into the identity/style bound to the trigger word. Whatever you describe becomes a variable the user can control at inference.

This single principle drives every captioning decision below.

## Character LoRAs

Goal: a LoRA that generates this specific person across many styles/contexts.

**Describe (makes these controllable at inference):**
- Clothing, hats, glasses, jewelry, accessories
- Pose, gesture, expression, action
- Background, setting, location, props
- Lighting direction, quality, color temperature
- Camera framing (close-up, medium shot, full body, low angle)
- Style / medium / format ("photograph", "oil painting", "3d render", etc.)

**Omit (makes these bound to the trigger):**
- Face shape, jawline, cheekbones
- Eye color, eye shape
- Hair color, hair type (mention hairstyle only if it clearly varies)
- Skin tone, freckles, moles, scars
- Body type, height, build
- Age descriptors
- Gender descriptors — use the trigger word in place of "woman"/"man"
- Ethnicity

**Format:** Natural language prose, not comma-separated tags. Transformer models (Flux, Z-Image, Chroma) expect this. Start with the trigger word. One sentence or two, aim for 20-40 words.

Example:
```
p3r5on wearing a cream wool sweater, holding a ceramic mug with both hands, sitting at a round wooden cafe table, soft natural window light from the left, shallow depth of field, photograph
```

## Style LoRAs

Goal: a LoRA that applies an aesthetic to any subject.

**Describe:** Content only. What's in the image.
**Omit:** Anything about the style, medium, or aesthetic.
**End every caption with the exact style descriptor phrase** the user will type at inference to activate the style.

The style descriptor must be identical across all captions in the dataset. Pick one exact phrase before captioning — don't let it drift.

Example (dataset of oil pastel illustrations):
```
shnzng a yellow apple with a blue leaf, on a white background, oil pastel illustration
shnzng a teapot with floral patterns, on a white background, oil pastel illustration
```

Note: both captions describe content and end with the same style phrase. The LoRA will learn that "oil pastel illustration" means "render in this visual style."

## Combined: character + style

This is the trickiest case. User has two datasets — identity references and target-style references (possibly of the same character).

Use **two dataset entries** in the config with different captioning strategies:

**Identity set:** Caption normally (describe variable elements, omit identity). End each caption with the medium descriptor that matches reality (e.g. `"modern digital photograph"`).

**Style set:** Caption variable elements, omit identity. End each caption with the target style descriptor (e.g. `"2010s smartphone photograph"`).

**Same trigger word across both sets.** The trigger binds identity; the style descriptor binds style. At inference:
- `p3r5on in a park, modern digital photograph` → character, modern photo look
- `p3r5on in a park, 2010s smartphone photograph` → character, that specific aesthetic

**Critical: the two style descriptors must be genuinely different phrases that pull on different priors in the base model.** If both sets are photographs, don't caption them both as "photograph" — the LoRA will have no way to separate them.

## Captioning at scale with a VLM

For anything more than ~20 images, use Gemini (or Claude, GPT-4V) with a structured system prompt. The key is hard-requiring the style descriptor and explicitly forbidding identity features.

### System prompt for character LoRA captioning

Replace `{style_descriptor}` with the exact suffix phrase for the run.

```
You are captioning images for a character LoRA training dataset. The character in every image is the same person. Your captions will teach a diffusion model to generate this character across many styles, so what you caption vs. what you omit directly controls what becomes variable vs. baked-in identity.

## Your one core rule

DESCRIBE everything that varies between images. OMIT everything that defines the character's fixed identity.

## ALWAYS describe

- Clothing, hats, glasses, jewelry, accessories
- Pose, gesture, expression, action
- Background, setting, location, props
- Lighting direction, quality, color temperature
- Camera framing (close-up, medium shot, full body, low angle, etc.)

## NEVER describe

- Face shape, jawline, cheekbones
- Eye color, eye shape
- Hair color, hair type (only mention hairstyle if it clearly changes between images)
- Skin tone, complexion, freckles, moles, scars
- Body type, height, build
- Age descriptors ("young", "middle-aged")
- Gender descriptors ("woman", "man", "girl", "boy") — use the trigger word instead
- Ethnicity

## Format rules

- Start every caption with the literal word: TRIGGER
- Write natural-language prose, not comma-separated tags.
- One sentence or two short sentences. Aim for 20–40 words.
- Do not invent details you cannot see.
- Do not use hedging language ("appears to be", "seems like").
- Do not include quality descriptors ("high quality", "detailed", "masterpiece", "4k").

## HARD REQUIREMENT: style descriptor

Every caption in this batch MUST end with exactly this phrase, verbatim and unchanged:

    {style_descriptor}

Do not invent a different style descriptor. End the caption with a comma, then this exact phrase, nothing after.

If the image style does not look like "{style_descriptor}" to you, end with the phrase anyway. The training code relies on this phrase being identical across all images in this batch.

## Examples (for a batch with style descriptor "{style_descriptor}")

GOOD: TRIGGER wearing a cream wool sweater, holding a ceramic mug with both hands, sitting at a round wooden cafe table, soft natural window light from the left, {style_descriptor}
GOOD: TRIGGER standing beside a tall oak tree, wearing a blue coat, rolling green hills behind, overcast diffuse daylight, three quarter view, {style_descriptor}
BAD: TRIGGER, a young woman in a sweater, smiling warmly, {style_descriptor}    (describes age/gender)
BAD: TRIGGER in a sweater, photograph    (wrong style descriptor — must use "{style_descriptor}")

## Your output

Respond with ONLY the caption text. No preamble, no explanation, no quotes, no formatting. One caption, ready to be saved as a .txt file.
```

After writing, swap `TRIGGER` for the actual trigger word (e.g., `p3r5on`) via find/replace.

### Style LoRA variant

Same prompt, but:
- Remove the identity rules (character isn't fixed across images)
- Keep the style descriptor hard requirement — this is the whole point of style LoRAs
- Strip any mention of the character

### Colab script

The repo has a ready-to-run captioning script at `scripts/caption_character_dataset_gemini.py`. The user can run it on Colab. See that script for the Colab cells.

Use `gemini-3.1-pro-preview` — Pro models follow the nuanced negative instructions ("don't describe eye color") much better than Flash. Cost is negligible for small datasets.

## Quality checks

After captioning, spot-check 5-10 captions:

- Do they all start with the trigger word?
- Do they all end with the expected style descriptor?
- Did Gemini sneak in face/hair descriptions? (Edit those out.)
- Are captions roughly the same length?
- Is there variety in the non-identity details described?

A bad captioning pass wastes the entire training run. Spend the 5 minutes to check.
