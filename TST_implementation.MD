# TRIGGER-SELECTIVE STYLE LORA TRAINING

## Ideogram 4 + AI Toolkit

### Implementation Specification - TST v1

Project trigger: <r1X1dOn9mA2>

*Dataset captions use AI Toolkit's native [trigger] placeholder*

Version 1.0 | 13 August 2026

Purpose: controlled experiment for arbitrary-trigger binding without changing inference architecture

# 1. Plain-language summary

The existing V3 experiment already showed that ordinary transformer LoRA training can learn the desired visual style, but the arbitrary literal trigger does not reliably expose that learned effect. The objective of this method is therefore not to relearn the style with a new architecture. It is to train the same kind of LoRA while giving the optimizer three separate responsibilities.

| Path | Simple job | Training behavior |
| --- | --- | --- |
| Path 1 | Learn the visual style | Normal dataset training with the literal trigger. This remains the dominant objective. |
| Path 2 | Do not leak | When [trigger] is replaced by an ordinary or competing style phrase, keep the LoRA close to the base model for that same condition. |
| Path 3 | Prefer the literal trigger | Measure how much the LoRA improves the dataset target under the literal trigger versus a matched non-trigger condition, and require a scheduled trigger advantage. |

> **Core principle:** Early training mostly learns appearance. Later training gradually increases pressure on trigger selectivity. The final LoRA is still an ordinary Ideogram 4 transformer LoRA: no Qwen LoRA, no token-aware inference, and no custom ComfyUI runtime behavior.

The method is intentionally designed as a falsifiable experiment. If the style learns again but the trigger gain does not separate from decoy gain, that is evidence that a static diffusion LoRA may not have enough conditional selectivity for this task. Only then should more invasive methods such as token-aware masking be justified.

# 2. Evidence and design constraints

- **Known positive result.** V3 used a rank-32 / alpha-16 Ideogram transformer LoRA and successfully carried the target visual style.

- **Known binding failure.** The arbitrary literal trigger was weak, while semantic style wording could expose the V3 effect much more strongly.

- **V3 already used DOP.** The V3 YAML had diff_output_preservation=true, multiplier=1, and class="painting". Therefore simply adding ordinary DOP again is not a new hypothesis.

- **Keep content mode.** The learned visual behavior includes composition/content tendencies, so content_or_style remains "content".

- **Minimize architecture changes.** The first test must stay diffusion-side only and preserve normal LoRA inference compatibility.

- **Use early kill gates.** Do not spend 1000-2000 steps unless trigger-selective metrics start moving.

> **Important correction to earlier plans:** The dataset does not literally contain <r1X1dOn9mA2>. It contains [trigger] at deliberately chosen positions, and AI Toolkit replaces that placeholder with the configured trigger_word. TST must reuse that mechanism for every branch rather than manually searching for the literal trigger after conditioning.

# 3. Final three-path training objective

```
ONE dataset sample / one noisy latent x_t / one timestep t

Raw caption template contains: [trigger]

                AI Toolkit trigger resolver
                       |
          +------------+-------------+
          |                          |
   resolve as literal          resolve as sampled
   <r1X1dOn9mA2>               non-trigger style s
          |                          |
     c_trigger                   c_decoy
          |                          |
   +------+-------+             +----+-----+
   |              |             |          |
LoRA ON        LoRA OFF      LoRA ON   LoRA OFF
   |              |             |          |
Path 1 / 3     base trigger   Path 2/3   base decoy
student        reference      student    reference
```

All four predictions use the same current training image latent, the same sampled noise, the same timestep, and the same flow/diffusion target. Text conditioning is the manipulated variable.

# 4. Trigger placeholder handling - reuse AI Toolkit, do not duplicate it

AI Toolkit officially supports [trigger] in captions and replaces it with trigger_word. TST must operate from the raw caption template that still contains [trigger]. The new code should not perform a second ad-hoc literal-trigger search after the normal prompt has already been conditioned.

## 4.1 Required behavior

1. Load the raw caption template exactly as the existing dataset path does. It contains one or more [trigger] placeholders.

1. Path 1 resolves every [trigger] using the existing configured trigger_word, exactly as ordinary AI Toolkit training already does.

1. For a decoy branch, call the same trigger-placeholder resolver but provide the sampled non-trigger phrase as the effective replacement value for that branch.

1. Path 3 reuses the already-resolved Path-1 trigger prompt and the already-resolved Path-2 decoy prompt. It must not create a third independent prompt transformation.

1. Do not rewrite JSON structure, reorder fields, or reconstruct captions through JSON serialization. Only the [trigger] replacement value changes.

## 4.2 Recommended small refactor

If the current AI Toolkit trigger logic is embedded inside a broader conditioning function, extract the smallest reusable helper without changing standard behavior. Conceptually:

```
resolve_trigger_placeholder(raw_prompt, replacement)

Path 1: replacement = configured trigger_word
Path 2: replacement = sampled decoy phrase
Path 3: reuse Path 1 and Path 2 outputs

Standard AI Toolkit training when TST is disabled: unchanged
```

Because this dataset intentionally positions [trigger], TST should support a strict validation mode and default it on for this project: every training caption must contain at least one [trigger]. This avoids the normal automatic-prepend fallback creating different placement semantics between branches.

```
trigger_selective_training:
  enabled: true
  require_trigger_placeholder: true
```

# 5. Path 1 - ordinary style acquisition

Path 1 is deliberately boring. It is the normal training forward that already made V3 a good style carrier. Do not create a parallel implementation.

```
raw caption + AI Toolkit [trigger] resolution
    -> caption containing <r1X1dOn9mA2>
    -> Ideogram text conditioning
    -> base + trainable transformer LoRA
    -> normal flow/diffusion prediction
    -> normal dataset target loss L1
```

The existing network rank, alpha, optimizer, learning rate, sigmoid timestep distribution, content mode, EMA, Differential Guidance and other V3 diffusion-side controls continue to come from the ordinary YAML configuration. This keeps the style-learning mechanism causal and comparable to V3.

> **Compatibility rule:** When trigger_selective_training.enabled=false, Path 1 must execute byte-for-byte-equivalent training logic to the current working trainer as far as the new patch is concerned.

# 6. Non-trigger style bank and sampling

Paths 2 and 3 use a rotating decoy condition. The user controls the number of categories, each category probability, and the phrases within each category from YAML. The first run should include neutral, hard/near semantic activators, and far competing styles.

| Category | Initial probability | Purpose | Examples |
| --- | --- | --- | --- |
| neutral | 0.30 | Test the real deployment case: no alternate style phrase. | "" (empty replacement) |
| hard | 0.40 | Protect against semantic regions already known or suspected to expose the LoRA. | painting; illustration; anime; Ghibli anime; storybook illustration |
| far | 0.30 | Prevent broad leakage into strongly different style semantics. | line art; photorealistic photograph; 3D render; technical drawing; ink drawing |

Sampling should be per training item by default. With batch_size=2, item A and item B may receive different decoys. Within each item, however, Path 2 and Path 3 must use the exact same sampled phrase. This improves diversity without contaminating the paired comparison.

## 6.1 YAML schema

```
trigger_selective_training:
  negative_styles:
    expected_category_count: 3
    sample_scope: per_item

    categories:
      - name: neutral
        probability: 0.30
        phrases:
          - ""

      - name: hard
        probability: 0.40
        phrases:
          - "painting"
          - "illustration"
          - "anime"
          - "Ghibli anime"
          - "storybook illustration"

      - name: far
        probability: 0.30
        phrases:
          - "line art"
          - "photorealistic photograph"
          - "3D render"
          - "technical drawing"
          - "ink drawing"
```

- expected_category_count is a validation value, not the source of truth. Actual count is len(categories). Mismatch is a startup error.

- Category probabilities must sum to 1.0 within tolerance.

- Phrases are uniformly sampled inside a category for v1. Phrase-level weighting can be added later without touching the loss design.

- Reject any decoy phrase containing [trigger] or the configured literal trigger.

# 7. Path 2 - conservative non-trigger preservation

Path 2 is a generalized rotating DOP branch. It does not try to make a decoy prompt bad at the dataset style. It only asks the new LoRA not to materially change what base Ideogram would have done for that same decoy-conditioned caption.

```
c_decoy = resolve_trigger_placeholder(raw_caption, sampled_style_s)

Teacher: base model, LoRA OFF, c_decoy, same x_t and t -> P_base_decoy
Student: base + LoRA, LoRA ON, c_decoy, same x_t and t -> P_lora_decoy

L2 = MSE(P_lora_decoy, stopgrad(P_base_decoy))
```

This starts close to zero when the effective LoRA is near zero and becomes a guardrail only when training causes the LoRA to leak into non-trigger conditions.

## 7.1 Interaction with legacy AI Toolkit DOP

Do not run legacy diff_output_preservation simultaneously with TST Path 2 in the first controlled experiment. V3 already used DOP with the class "painting". TST Path 2 replaces that single fixed preservation class with a sampled bank of conditions.

```
# Recommended validation when TST is enabled
train:
  diff_output_preservation: false

# TST Path 2 may reuse DOP's existing prior-prediction / LoRA-bypass plumbing internally.
```

If both legacy DOP and TST are enabled accidentally, the trainer should raise a clear configuration error unless a future explicit compatibility mode is added. Silent double-preservation would make the experiment uninterpretable.

# 8. Path 3 - preferential trigger gain

Path 2 alone can suppress leakage but does not guarantee that the literal trigger becomes a strong activator. Path 3 adds a relative objective: the LoRA should improve prediction of the dataset target more under the literal trigger than under the matched non-trigger decoy.

The comparison must be baseline-normalized. Raw trigger loss and raw decoy loss cannot be directly ranked because base Ideogram may naturally be closer to the dataset under one phrase than another even before the LoRA learns anything.

```
For the same x_t, timestep t and target y:

D_trigger = dataset_loss(LoRA(c_trigger), y)
B_trigger = dataset_loss(Base(c_trigger), y)

D_decoy   = dataset_loss(LoRA(c_decoy), y)
B_decoy   = dataset_loss(Base(c_decoy), y)

G_trigger = 1 - D_trigger / (B_trigger + eps)
G_decoy   = 1 - D_decoy   / (B_decoy   + eps)
```

G is the normalized improvement attributable to the LoRA under that condition. Near initialization, both gains should be near zero even if the base model has very different absolute losses for "line art" and "Ghibli anime".

## 8.1 Critical gradient rule

Path 3 must not be allowed to satisfy the ranking by deliberately making the decoy branch worse. Therefore G_decoy is detached inside Path 3. Path 2 alone controls the decoy branch. Path 3 can only increase the trigger-conditioned advantage.

```
L3 = max(0, margin(step) - G_trigger + stopgrad(G_decoy))
```

Hinge ranking is recommended for v1. Once the required advantage is satisfied, Path 3 stops pushing that item. This reduces the risk of an endlessly amplifying trigger objective.

# 9. Scheduled Path-3 margin

The margin must itself be scheduled. Early training should require only a small trigger advantage because the LoRA is still learning the visual effect. Later training should require a larger separation, matching the curriculum in which binding becomes more important after the style carrier is established.

| Step | Recommended margin | Interpretation |
| --- | --- | --- |
| 0 | 0.02 | Very mild selectivity pressure while appearance learning dominates. |
| 1000 | 0.08 | Require a clearer normalized trigger advantage at mid-training. |
| 1500+ | 0.12 | Hold a stronger selectivity requirement through the final phase. |

These values are starting defaults, not sacred constants. Margin is a normalized gain gap, not a direct percentage of visible style strength.

## 9.1 YAML schema

```
trigger_selective_training:
  path3:
    loss_type: hinge
    gain_epsilon: 1.0e-6

    margin_schedule:
      interpolation: smoothstep
      keyframes:
        - step: 0
          value: 0.02
        - step: 1000
          value: 0.08
        - step: 1500
          value: 0.12
```

After the last margin keyframe, clamp to the last value. Before the first keyframe, clamp to the first. The same generic scheduler utility should support both linear and smoothstep interpolation.

# 10. Scheduled three-path loss weights

The user controls the nominal percentage assigned to each objective. The initial schedule intentionally keeps style acquisition dominant, then gradually shifts more coefficient weight toward trigger selectivity.

| Step | Path 1 | Path 2 | Path 3 | Training emphasis |
| --- | --- | --- | --- | --- |
| 0 | 80% | 10% | 10% | Mostly learn the style; light leakage/selectivity constraints. |
| 1000 | 60% | 15% | 25% | Style should be established; binding pressure grows. |
| 1500+ | 50% | 15% | 35% | Final phase: preserve style while strongly optimizing trigger selectivity. |

## 10.1 YAML schema

```
trigger_selective_training:
  loss_schedule:
    interpolation: smoothstep
    normalize_weights: true

    keyframes:
      - step: 0
        path1: 0.80
        path2: 0.10
        path3: 0.10

      - step: 1000
        path1: 0.60
        path2: 0.15
        path3: 0.25

      - step: 1500
        path1: 0.50
        path2: 0.15
        path3: 0.35
```

> **Interpret percentages correctly:** These are normalized loss coefficients, not guaranteed percentages of the optimizer update. Different losses can have different scales and gradient norms. Log branch gradient norms during short diagnostics before treating 80/10/10 as literal update shares.

# 11. Interpolation behavior

Both the loss-weight schedule and the margin schedule use the same generic keyframe scheduler. Support two modes in v1:

| Mode | Behavior | Recommended use |
| --- | --- | --- |
| linear | Straight interpolation between adjacent keyframes. | Useful for debugging because values are easy to predict. |
| smoothstep | Uses u^2(3-2u) inside each interval, giving smooth acceleration/deceleration. | Recommended first real run to avoid abrupt slope changes at keyframes. |

Weights are normalized after interpolation when normalize_weights=true. Therefore 80/10/10 and 0.8/0.1/0.1 are equivalent inputs. Keyframe steps must be strictly increasing.

# 12. Recommended first-run YAML block

The following block is the proposed v1 interface. Existing V3 network/train/model settings remain outside it.

```
trigger_word: "<r1X1dOn9mA2>"

trigger_selective_training:
  enabled: true
  require_trigger_placeholder: true

  negative_styles:
    expected_category_count: 3
    sample_scope: per_item
    categories:
      - name: neutral
        probability: 0.30
        phrases: [""]

      - name: hard
        probability: 0.40
        phrases:
          - "painting"
          - "illustration"
          - "anime"
          - "Ghibli anime"
          - "storybook illustration"

      - name: far
        probability: 0.30
        phrases:
          - "line art"
          - "photorealistic photograph"
          - "3D render"
          - "technical drawing"
          - "ink drawing"

  path3:
    loss_type: hinge
    gain_epsilon: 1.0e-6
    margin_schedule:
      interpolation: smoothstep
      keyframes:
        - {step: 0,    value: 0.02}
        - {step: 1000, value: 0.08}
        - {step: 1500, value: 0.12}

  loss_schedule:
    interpolation: smoothstep
    normalize_weights: true
    keyframes:
      - {step: 0,    path1: 0.80, path2: 0.10, path3: 0.10}
      - {step: 1000, path1: 0.60, path2: 0.15, path3: 0.25}
      - {step: 1500, path1: 0.50, path2: 0.15, path3: 0.35}

  logging:
    log_every: 25
    log_category_stats: true
    log_gain_stats: true
    debug_gradient_contributions: false

train:
  # Path 2 supersedes legacy fixed-class DOP in this experiment.
  diff_output_preservation: false
```

# 13. AI Toolkit integration plan

## 13.1 Config layer

- Add TriggerSelectiveTrainingConfig and nested config objects in toolkit/config_modules.py or the project-equivalent config module.

- Validate category count, probability sum, non-empty categories, schedule ordering, non-negative loss weights and legal interpolation modes at startup.

- When TST is enabled, reject legacy diff_output_preservation=true for the first controlled implementation.

## 13.2 New isolated helper module

Prefer a small new module such as toolkit/trigger_selective_training.py. It should contain logic that is independent of Ideogram model internals:

- negative category/phrase sampling;

- shared keyframe scheduler for scalar margin and vector loss weights;

- [trigger] variant resolution wrapper that reuses the existing AI Toolkit trigger mechanism;

- Path-3 gain and hinge-loss computation;

- validation and lightweight metrics aggregation.

## 13.3 SDTrainer integration

Keep the main Path-1 training route intact. Around it, add the decoy and base-reference predictions. Reuse existing DOP prior-prediction / network-bypass mechanisms where they already provide the correct model state.

```
per training item/batch:

1. sample x_t, t, target exactly once
2. sample decoy category + phrase s
3. build c_trigger and c_decoy from the same raw [trigger] caption template
4. compute base-decoy prediction (no grad, LoRA off)
5. compute student-decoy prediction (grad, LoRA on)
6. compute L2 and detached G_decoy
7. compute base-trigger prediction (no grad, LoRA off)
8. run the normal Path-1 student-trigger forward (grad, LoRA on)
9. compute L1, G_trigger and L3 using scheduled margin
10. get scheduled w1/w2/w3
11. accumulate weighted gradients
12. optimizer step follows existing AI Toolkit cadence
```

# 14. Memory-conscious backward strategy

A naive implementation holds both gradient-bearing student graphs at once. That may be unnecessarily expensive on a 24 GB card. Because Path 3 deliberately stop-grads G_decoy, the total gradient can be accumulated in two stages without changing the intended mathematics.

```
# Stage A: decoy branch
base_decoy = no_grad_base_forward(c_decoy)
student_decoy = grad_forward(c_decoy)
L2 = preservation_loss(student_decoy, base_decoy)
G_decoy = gain(student_decoy, base_decoy, target).detach()
backward(w2 * L2)
# release decoy student graph

# Stage B: trigger branch
base_trigger = no_grad_base_forward(c_trigger)
student_trigger = normal_path1_grad_forward(c_trigger)
L1 = dataset_loss(student_trigger, target)
G_trigger = gain(student_trigger, base_trigger, target)
L3 = hinge(margin - G_trigger + G_decoy)
backward(w1 * L1 + w3 * L3)

optimizer.step()  # once, using accumulated gradients
```

This staged backward is mathematically equivalent to a single combined backward under the v1 detach rule, while allowing the decoy graph to be freed before the trigger graph is created. It must be tested carefully with AI Toolkit gradient accumulation and Accelerator semantics. If uncertain, implement the single-combined-loss version first for correctness, then switch to staged backward only after an equivalence test.

> **Reuse rule:** If Differential Guidance or another existing AI Toolkit path already computes an exactly matching base-trigger prediction under the same x_t, timestep and conditioning, it may be reused. Do not reuse a superficially similar prior tensor unless the conditions are proven identical.

# 15. Model-state and autograd safety

- **Teacher/base forwards.** LoRA disabled, torch.no_grad(), detached outputs, base model frozen.

- **Student forwards.** LoRA enabled and trainable; base Ideogram weights remain frozen.

- **State restoration.** Use a scoped context manager or try/finally for LoRA enable/disable. Never scatter manual state toggles through the trainer.

- **Same stochastic state.** All branches share the same sampled x_t, timestep and target. Do not resample noise per branch.

- **Differential Guidance.** Keep the V3 setting unchanged for the first TST experiment. Do not simultaneously change the known-good diffusion recipe.

# 16. Logging and diagnostics

The experiment is only useful if it tells us why it succeeds or fails. Log raw losses, weighted losses, gains, schedule values and negative-category statistics separately.

```
loss/path1_raw
loss/path2_raw
loss/path3_raw
loss/path1_weighted
loss/path2_weighted
loss/path3_weighted
loss/total

weight/path1
weight/path2
weight/path3
path3/margin
path3/margin_satisfied

gain/trigger
gain/decoy
gain/gap

negative/category
negative/phrase
negative/neutral_count
negative/hard_count
negative/far_count
```

For running aggregate analysis, maintain separate EMAs or averages of gain/gap for neutral, hard and far categories. This can reveal, for example, that the trigger separates cleanly from neutral prompts while still failing against anime-like semantics.

## 16.1 Sparse gradient diagnostics

The 80/10/10 schedule is a coefficient schedule, not a guarantee of 80/10/10 gradient influence. During short diagnostic runs, optionally measure branch gradient norms at sparse steps such as 1, 10, 25, 50, 100, 250, 500, 1000 and 1500. Keep this behind a debug flag because it is expensive.

# 17. Required tests before real training

| Area | Required assertion |
| --- | --- |
| Disabled-mode regression | TST disabled -> existing trainer behavior unchanged. |
| Placeholder | Every raw caption contains [trigger] when strict mode is enabled. |
| Placeholder | Path 1 and Path 2 use the same native resolver with different replacement values. |
| Placeholder | All [trigger] occurrences in a caption are resolved; no manual literal-trigger search is required. |
| Sampling | Category probabilities validate and sampled phrase is shared by Path 2 and Path 3 for each item. |
| Scheduling | Exact keyframe values, linear interpolation and smoothstep interpolation all unit-test correctly. |
| Path 2 | Teacher is no-grad/LoRA-off; student is grad/LoRA-on. |
| Path 3 | G_decoy is detached; Path-3-only backward cannot change decoy-branch parameters through its graph. |
| Branch pairing | All predictions share the same x_t, timestep and target. |
| State | LoRA enabled/disabled state is restored after every teacher context. |
| Legacy DOP | TST + legacy diff_output_preservation raises a clear error in v1. |
| Backward | Staged and combined backward produce matching gradients within tolerance on a small deterministic test. |

# 18. Step-zero sanity expectations

| Metric | Expected at effective LoRA initialization |
| --- | --- |
| L1 | High relative to a trained checkpoint; normal dataset acquisition has not happened yet. |
| L2 | Near zero because LoRA student should initially approximate the base teacher on the decoy. |
| G_trigger | Near zero. |
| G_decoy | Near zero. |
| L3 | Usually active because the scheduled margin is positive and the gain gap is near zero. |

If L2 is large at step zero, the teacher/student model state is wrong. If G_trigger and G_decoy are already very different, inspect base-loss normalization before training.

# 19. First controlled experiment and kill gates

Use a fresh LoRA initialization with the same known-good V3 diffusion-side configuration, except that legacy fixed-class DOP is disabled and TST supplies Path 2. Do not simultaneously change rank, timestep sampling, content mode, Differential Guidance, dataset or caption structure.

| Checkpoint | What to inspect |
| --- | --- |
| 25 / 50 | Do L1 and gains move at all? Is L2 still small? Are schedules returning expected values? |
| 100 / 200 | Does G_trigger begin separating from G_decoy? Does visual trigger ON/OFF show any difference? |
| 500 | If no selectivity movement exists, stop rather than assuming 2000 steps will fix it. |
| 1000 | Confirm mid-training schedule transition: 60/15/25 and margin ~0.08. |
| 1500+ | Confirm final schedule: 50/15/35 and margin ~0.12; inspect whether style fidelity remains intact. |

## 19.1 Primary evaluation matrix

```
Fixed seed / sampler / steps / LoRA strength:

1. novel neutral prompt
2. same novel prompt + <r1X1dOn9mA2>

3. Ghibli anime prompt
4. same Ghibli anime prompt + <r1X1dOn9mA2>

5. line art prompt
6. same line art prompt + <r1X1dOn9mA2>

7. unseen style phrase
8. same unseen phrase + <r1X1dOn9mA2>
```

The first success criterion is not perfect suppression of all no-trigger LoRA effects. It is a material, repeatable increase in target-style activation when the literal trigger is present, while Path 2 keeps non-trigger leakage conservative enough that base prompt behavior remains usable.

# 20. Failure interpretation

| Observed behavior | Likely interpretation / next action |
| --- | --- |
| L1 decreases, G_trigger and G_decoy stay similar | Style is relearning without conditional binding. Static LoRA selectivity may be insufficient or Path 3 pressure may be too weak. |
| L2 rises strongly while L1 improves | Leakage outpaces preservation. Inspect actual gradient norms before simply increasing Path-2 coefficient. |
| G_trigger rises, L2 stays low, images still do not change | Diffusion-prediction gain may not map to visible style activation strongly enough; inspect output residual direction and timestep dependence. |
| Path 3 remains unsatisfied late and L1 degrades | Margin and/or Path-3 weight may be too aggressive; reduce scheduled endpoint rather than training longer. |
| Neutral separation works, hard negatives fail | Binding exists but is not robust against pretrained style semantics. Increase hard-negative sampling or add a later simultaneous competitor+trigger extension. |

# 21. Recommended code footprint

| File / area | Change |
| --- | --- |
| toolkit/config_modules.py | Add TST config objects, validation and schedule schema. |
| toolkit/trigger_selective_training.py (new) | Sampling, schedule interpolation, [trigger] variant helper/wrapper, gain calculation, Path-3 hinge, metrics helpers. |
| extensions_built_in/sd_trainer/SDTrainer.py | Integrate decoy/base forwards and weighted losses around the existing Path-1 training route. |
| Existing trigger resolver location | Only if necessary, extract a reusable replacement helper. Preserve existing [trigger] behavior exactly for ordinary training. |
| Generic LoRA/network code | No change expected for v1. |
| Ideogram/Qwen model code | No change expected for v1. |
| ComfyUI loader | No change. Result remains a standard transformer LoRA. |

# 22. Compact mathematical definition

This section is intentionally small. It defines only the quantities the implementation needs.

```
Definitions
-----------
c_raw  = raw dataset caption containing [trigger]
tau    = configured literal trigger <r1X1dOn9mA2>
s      = sampled non-trigger style phrase
R(c,x) = AI Toolkit's native [trigger] resolution using replacement x

c_tau  = R(c_raw, tau)
c_s    = R(c_raw, s)

f_theta = base Ideogram + trainable LoRA
f_0     = frozen base Ideogram with trainable LoRA disabled
y       = current flow/diffusion target

Dataset losses
--------------
D_tau = L(f_theta(x_t, c_tau, t), y)
B_tau = L(f_0    (x_t, c_tau, t), y)
D_s   = L(f_theta(x_t, c_s,   t), y)
B_s   = L(f_0    (x_t, c_s,   t), y)

Normalized LoRA gains
---------------------
G_tau = 1 - D_tau / (B_tau + eps)
G_s   = 1 - D_s   / (B_s   + eps)

Three losses
------------
L1 = D_tau
L2 = MSE(f_theta(x_t,c_s,t), stopgrad(f_0(x_t,c_s,t)))
L3 = max(0, m(step) - G_tau + stopgrad(G_s))

Total objective
---------------
L_total(step) = w1(step)*L1 + w2(step)*L2 + w3(step)*L3

Smoothstep interpolation
------------------------
u in [0,1] between adjacent keyframes
s(u) = u^2 * (3 - 2u)
value = value_A + s(u) * (value_B - value_A)
```

# 23. Decisive implementation summary

- **Do preserve:** the exact V3-style ordinary diffusion training path as Path 1.

- **Do reuse:** AI Toolkit native [trigger] placeholder resolution for both literal-trigger and decoy prompt variants.

- **Do replace:** legacy single-class DOP with Path 2 rotating preservation when TST is enabled.

- **Do add:** Path 3 baseline-normalized trigger-vs-decoy gain ranking, with the decoy gain detached.

- **Do schedule:** both three-path loss coefficients and the Path-3 margin through YAML keyframes using linear or smoothstep interpolation.

- **Do keep:** same x_t, noise, timestep and target across all paired branches.

- **Do expose:** negative categories, category probabilities, phrases, loss schedules, interpolation, margin schedule and diagnostics.

- **Do not add yet:** Qwen LoRA, TARA token masking, text embeddings, custom inference or hard runtime trigger gating.

> **Hypothesis under test:** A standard Ideogram transformer LoRA has enough conditional capacity to learn a strong style residual while making the arbitrary literal trigger preferentially expose that residual, provided training separately optimizes style acquisition, non-trigger preservation, and trigger-relative gain.

# 24. Source basis

Project evidence and current implementation details used to form this specification:

- **Project V3 YAML.** 2026_08_11_ig4_r1X1dOn9mA2_v3.yaml. Relevant facts: trigger_word=<r1X1dOn9mA2>, LoRA rank 32 / alpha 16, content mode, sigmoid timesteps, DOP enabled with class "painting", Differential Guidance scale 3.

- **Project continuation handoff.** Dated 2026-08-12: IDEOGRAM 4 ARBITRARY-TRIGGER STYLE LORA PROJECT. Relevant facts: target objective, dataset structure, previous TI/Qwen-LoRA results, early kill-gate preference, and Ideogram/Qwen architecture notes.

- **AI Toolkit official documentation.** ui/src/docs.tsx (checked 2026-08-13): [trigger] in captions is automatically replaced with the configured trigger_word.

- **AI Toolkit official configuration code.** toolkit/config_modules.py (checked 2026-08-13): current DOP fields and semantics, including diff_output_preservation_class.

- **AI Toolkit trainer source / project trainer snapshots.** The existing prior-prediction path already supports LoRA-bypassed teacher predictions and can be reused as plumbing for Path 2 where conditions match.

This specification intentionally distinguishes established AI Toolkit behavior from the new TST hypothesis. TST itself is an experimental project design, not a claim of an existing published algorithm.
