# Ideogram 4 Three-Phase Trigger Binding
# Current Mathematical Objective and Diagnostics Report

## Purpose

This report is written for an independent mathematical reviewer.

It describes the **actual current implementation**, with the v7 configuration as the concrete experiment, rather than only describing the intended design.

The reviewer should use this report to evaluate:

- whether the objectives are mathematically coherent;
- whether their gradients encourage the intended factorization;
- whether scales and schedules are calibrated;
- whether source aggregation is appropriate;
- whether the logged diagnostics are sufficient to distinguish success, shortcut learning and objective conflict;
- whether the three phases really separate text-side trigger representation learning from diffusion-side style rendering.

The system consists of:

```text
A1: train text activator against frozen base Ideogram diffusion
B:  train diffusion LoRA against frozen A1 text activator
A2: train text activator against frozen B diffusion LoRA
```

The current v7 run uses:

```text
Trigger placeholder: [trigger]
Literal trigger: <r1X1dOn9mA2>
Model: Ideogram 4
Text encoder: frozen Qwen3-VL-8B-Instruct
Diffusion model: Ideogram 4 flow-matching transformer
Batch size: 2
Text activator: one learned embedding + one shared internal adapter + 13 tap adapters
```

---

# 1. Source-of-truth implementation files

The actual objective is primarily implemented in:

```text
extensions_built_in/sd_trainer/SDTrainer.py
  _calculate_trigger_binding_loss()   # actual A1/A2 objective
  _calculate_tst_loss()               # actual B objective
  calculate_loss()                    # B Path-1 reduction/scaling
  _write_trigger_binding_metrics()    # A1/A2 JSONL
  _check_first_trigger_gradient()     # first-real-loss reachability gate

toolkit/trigger_binding_losses.py
  per_item_diffusion_mse()
  normalized_activator_gain()
  activator_gain_floor_hinge()
  pooled_trigger_residual_consistency()
  aggregate_paired_source_losses()

toolkit/trigger_selective_training.py
  normalized_gain()
  trigger_advantage_hinge()
  trigger_gain_floor_hinge()
  schedule interpolation and negative-style sampling

toolkit/models/ideogram4_trigger_activator.py
  mathematical form of embedding/internal/tap components

toolkit/trigger_reachability.py
  optimizer isolation and first-loss gradient checks

extensions_built_in/sd_trainer/ThreePhaseTriggerTrainer.py
  phase-local child configuration and artifact handoff

config/2026_08_15_ig4_r1X1dOn9mA2_v7.yaml
  current schedules, weights and learning rates
```

Important implementation fact:

```text
compute_a1_loss() and compute_a2_loss() exist in trigger_binding_losses.py,
but the current runtime does not call them.
```

The actual A1/A2 objective is assembled manually inside `_calculate_trigger_binding_loss()`.

Any mathematical audit must use the formulas in this report, which reflect that runtime path.

---

# 2. Notation

Let:

```text
i                 batch item index
s                 caption source, structured or natural
x_i               target image latent
ε_i               sampled noise
τ_i               sampled diffusion timestep
z_i(τ_i)          noisy latent produced from x_i and ε_i
Y_i                shared flow-matching target
C_s,i             source caption containing [trigger]
A_θ                trainable text activator, parameters θ
A_0                activator bypass
F_0                frozen base Ideogram diffusion transformer
F_φ                Ideogram transformer with trainable diffusion LoRA φ
F_B                frozen Phase-B diffusion LoRA renderer during A2
P(M,C,z,τ)          diffusion prediction from model M and conditioning C
```

The current shared target is produced by:

```python
Y = sd.get_loss_target(noise=noise, batch=batch, timesteps=timesteps)
```

For flow matching, the fallback is:

```text
Y = ε - x
```

and is detached.

In v7:

```text
do_differential_guidance = false
loss_type = mse
timestep_type = sigmoid
diff_output_preservation = false
```

Therefore the special differential-guidance target transformation is inactive.

All compared branches within one optimizer step share:

- the same image latents;
- the same noise;
- the same timesteps;
- the same noisy latents;
- the same target tensor.

---

# 3. Text activator function

The text activator contains three trainable component families.

## 3.1 Atomic learned embedding

A new atomic tokenizer token is registered for:

```text
<r1X1dOn9mA2>
```

The embedding parameter is initialized from the mean Qwen embedding of:

```text
illustration
```

For trigger-token mask `m`:

```text
H_embed' = lerp(H_embed, E_θ, m)
```

For a binary mask, the trigger positions are replaced by `E_θ` and non-trigger positions remain unchanged.

## 3.2 Shared internal adapter

The same rank-1 adapter is applied after every Qwen decoder layer:

```text
U_int(H) = W_up,int W_down,int H
H' = H + m ⊙ U_int(H) · scale · α/r
```

In v7:

```text
r = 1
α = 1
scale = 1, non-learnable
dropout = 0
```

Only trigger-token positions receive the residual.

## 3.3 Per-tap adapters

There is one separate rank-1 adapter for each Qwen tap layer:

```text
T = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35}
```

At tap layer `ℓ`:

```text
U_ℓ(H_ℓ) = W_up,ℓ W_down,ℓ H_ℓ
Tap_ℓ' = H_ℓ + m ⊙ U_ℓ(H_ℓ) · scale_ℓ · α_ℓ/r_ℓ
```

The tap-adapted tensor is captured for final Ideogram conditioning.

The tap adapter output does not replace the recurrent hidden state passed to the next Qwen layer. The shared internal adapter does alter the recurrent hidden state.

Important actual-versus-declared detail:

```text
The current runtime applies one shared MaskedLowRankAdapter to the full hidden
state after every Qwen decoder layer. The configured parent_modules,
child_modules and layers fields do not currently construct conventional LoRA
wrappers specifically on each Qwen down_proj module.
```

Thus the mathematical object being trained is a shared post-layer rank-1 residual adapter, not a collection of rank-1 `down_proj` LoRAs.

## 3.4 Final text conditioning

The 13 captured tensors are concatenated:

```text
Tap_ℓ' ∈ R^(B × L × 4096)
C ∈ R^(B × L × 53248)
```

Only the text activator parameters are trainable in A1/A2. Base Qwen remains frozen, but gradients propagate through its operations to the small activator parameters.

---

# 4. Common per-item MSE and normalized gain

For a prediction tensor `P_i` and target `Y_i`, the A1/A2 and TST gain helper computes:

```text
D(P_i,Y_i) = mean over all non-batch tensor elements of (float(P_i)-float(Y_i))²
```

The implementation converts both tensors to float32 before MSE.

Given an active/student loss `D_a,i` and bypass/base loss `D_b,i`, normalized gain is:

```text
G_i = 1 - D_a,i / (stopgrad(D_b,i) + ε_g)
```

with:

```text
ε_g = 1e-6
```

Interpretation:

```text
G > 0: active/student branch improves target MSE relative to its baseline
G = 0: equal to baseline
G < 0: active/student branch is worse than baseline
```

Gradient behavior:

```text
∂G/∂D_a = -1 / (D_b + ε_g)
```

The denominator is detached. The gain objective therefore scales gradients inversely with the baseline loss.

This denominator scaling is one of the main mathematical points that should be reviewed.

---

# 5. Phase A1 actual objective

## 5.1 Trainable and frozen parameters

Trainable:

```text
learned trigger embedding θ_E
shared internal adapter θ_I
13 tap adapters θ_T
```

Frozen:

```text
base Qwen parameters
base Ideogram diffusion transformer
all diffusion LoRA parameters, because no diffusion LoRA is trained in A1
VAE
```

The optimizer is statically filtered so only text-activator parameters are present.

## 5.2 Caption sources

For each image item, A1 simultaneously uses both captions:

```text
structured JSON caption: weight 0.75
natural-language caption: weight 0.25
```

These are paired descriptions of the same image.

Both source forwards share the same `z_i(τ_i)`, `ε_i`, `τ_i` and `Y_i`.

## 5.3 Active and bypass branches

For each source `s`:

```text
P_A1,s,i = P(F_0, A_θ(C_s,i), z_i, τ_i)
P_0,s,i  = P(F_0, A_0(C_s,i), z_i, τ_i)
```

The bypass text encoding and bypass diffusion prediction run under `torch.no_grad()` and are detached.

The active branch retains gradients through:

```text
active conditioning
frozen Qwen operations
frozen diffusion operations
text activator parameters
```

This is intentional: frozen weights are differentiable operations even though they are not optimizer parameters.

## 5.4 Per-source diffusion loss

```text
D_s,i = D(P_A1,s,i, Y_i)
B_s,i = D(P_0,s,i, Y_i)
```

Activator gain:

```text
G_s,i = 1 - D_s,i / (stopgrad(B_s,i) + ε_g)
```

## 5.5 A1 gain-floor hinge

Let scheduled floor be `f_A1(k)` at local A1 step `k`.

```text
H_floor,s,i = ReLU(f_A1(k) - G_s,i)
```

v7 uses weight:

```text
λ_floor = 1.0
```

## 5.6 Per-source A1 objective

With diffusion weight `λ_D = 1`:

```text
L_A1,s,i = λ_D D_s,i + λ_floor H_floor,s,i
```

The source-weighted image/gain objective is:

```text
L_A1,source,i = 0.75 L_A1,structured,i + 0.25 L_A1,natural,i
```

Weights are normalized by the runtime aggregator. In v7 they already sum to one.

## 5.7 A1 context-consistency residual

For each caption source and Qwen tap `ℓ`, define active and bypass tap tensors:

```text
T_active,s,i,ℓ
T_bypass,s,i,ℓ
```

The trigger-specific residual is:

```text
Δ_s,i,ℓ,t = T_active,s,i,ℓ,t - stopgrad(T_bypass,s,i,ℓ,t)
```

Only tokens selected by the trigger mask and valid-token mask are pooled.

With mean pooling:

```text
δ_s,i,ℓ = mean over trigger tokens t of Δ_s,i,ℓ,t
```

The structured source is the `source`; natural is the `reference`.

Because v7 sets:

```text
detach_reference = false
```

both `δ_structured` and `δ_natural` retain gradients through their active branches.

For each valid tap:

```text
C_cos,i,ℓ = 1 - cosine(δ_structured,i,ℓ, δ_natural,i,ℓ)
```

Tap validity gate:

```text
||δ_structured,i,ℓ|| >= 1e-6
and
||δ_natural,i,ℓ|| >= 1e-6
```

The validity decision uses detached norms.

The per-item cosine consistency is the mean over valid taps:

```text
C_cos,i = sum_ℓ valid_i,ℓ C_cos,i,ℓ / max(number_valid_taps,1)
```

Magnitude consistency is implemented but disabled in v7:

```text
magnitude_weight = 0
```

Therefore residual magnitudes are not explicitly aligned, only directions.

Important actual-versus-declared detail:

```text
The runtime hard-codes cosine_weight=1 and uses magnitude_weight from config.
The configured context loss_type, alignment and mask strings are validated,
but the active runtime does not branch on them. It always uses pooled trigger
residual consistency on trigger-mask tokens.
```

For v7 this agrees with the declared `loss_type: cosine`, `alignment: trigger_pooled` and `mask: trigger`; changing those strings alone would not currently change the runtime formula.

Warmup:

```text
w_A1(k) = min(max(k / 40, 0), 1)
```

Raw context loss:

```text
C_A1,i = w_A1(k) C_cos,i
```

Configured context weight:

```text
λ_C = 0.1
```

Weighted context contribution:

```text
L_A1,context,i = 0.1 C_A1,i
```

## 5.8 Final A1 objective

```text
L_A1 = mean_i [
    0.75 (D_structured,i + ReLU(f_A1-G_structured,i))
  + 0.25 (D_natural,i    + ReLU(f_A1-G_natural,i))
  + 0.1 w_A1(k) mean_valid_taps(1-cosine(δ_structured,δ_natural))
]
```

## 5.9 v7 A1 floor schedule

Smoothstep interpolation:

```text
k=0:   0.00
k=80:  0.02
k=160: 0.04
k=240: 0.06
k=320: 0.08
k=400: 0.10
```

Between keyframes:

```text
u = (k-k_left)/(k_right-k_left)
smoothstep(u) = u²(3-2u)
f(k) = f_left + smoothstep(u)(f_right-f_left)
```

## 5.10 Mathematical points for A1 review

The reviewer should examine:

1. Whether direct diffusion MSE plus relative gain-floor MSE is redundant or usefully normalized.
2. Whether the normalized gain denominator overweights low-baseline-loss items/timesteps.
3. Whether floor `0.10` is calibrated to achievable activator gain.
4. Whether cosine-only context consistency permits source residual magnitudes to diverge.
5. Whether `detach_reference=false` admits a collapse/co-adaptation solution where both source residuals rotate together without preserving a privileged structured anchor.
6. Whether fixed 0.75/0.25 source weighting is appropriate when the context term already symmetrically couples the two sources.
7. Whether the context term should operate on all 13 taps equally or use layer-dependent weights.
8. Whether mean pooling across all trigger occurrences loses useful occurrence-specific information.

---

# 6. Phase B actual objective

## 6.1 Trainable and frozen parameters

Trainable:

```text
diffusion LoRA φ only
rank = 32
alpha = 16
learning rate = 8e-5
```

Frozen:

```text
A1 learned embedding
A1 shared internal adapter
A1 tap adapters
base Qwen
base Ideogram transformer weights
VAE
```

The frozen A1 activator is loaded from the A1 final artifacts and remains active for trigger-bearing prompts.

## 6.2 Caption source

v7 Phase B is JSON-only:

```text
caption source: main structured .json captions read as full text
source probability: 1.0
```

Natural captions do not supervise the diffusion LoRA in Phase B.

## 6.3 Per-item decoy sampling

For each item, one negative phrase is sampled independently.

Category probabilities:

```text
neutral: 0.30
hard:    0.40
far:     0.30
```

Neutral phrases:

```text
""
```

Hard phrases:

```text
painting
illustration
anime
Ghibli anime
storybook illustration
```

Far phrases:

```text
line art
photorealistic photograph
3D render
technical drawing
ink drawing
```

The raw caption's `[trigger]` is replaced by:

```text
trigger branch: <r1X1dOn9mA2>
decoy branch: sampled decoy phrase
```

The same decoy is used by B Path 2 and Path 3 for that item.

## 6.4 Four diffusion predictions

For each item:

```text
B_trigger,i = P(F_0, A_A1(C_trigger,i), z_i, τ_i)
D_trigger,i = P(F_φ, A_A1(C_trigger,i), z_i, τ_i)

B_decoy,i   = P(F_0, C_decoy,i, z_i, τ_i)
D_decoy,i   = P(F_φ, C_decoy,i, z_i, τ_i)
```

Here:

```text
B = base diffusion LoRA disabled
D = student diffusion LoRA enabled
```

The trigger branch uses the frozen A1 activator in both base and student predictions.

The decoy prompt does not contain the atomic trigger, so its trigger mask is absent/zero and the A1 activator contributes no trigger residual.

Base predictions run under `torch.no_grad()` and are detached.

## 6.5 B Path 1: target-style acquisition

```text
L1 = calculate_loss(D_trigger, Y)
```

For the current v7 settings, this is fundamentally MSE, but it uses the general trainer's `calculate_loss()` path.

That path can include:

- model-specific `scale_loss()`;
- spatial mask multiplier;
- item loss multipliers;
- optional SNR/timestep weighting;
- additional model loss;
- other generic trainer options.

In the current v7 configuration most optional transformations are inactive, masks are normally absent and item multipliers are expected to be one. Nevertheless, mathematically `L1` is not guaranteed to be identical to a plain mean of the helper `per_item_mse()` under all configurations.

This distinction matters because Path 2 and Path 3 use plain float32 per-item MSE without `calculate_loss()`.

## 6.6 B Path 2: non-trigger preservation

Per item:

```text
L2_i = D(D_decoy,i, B_decoy,i)
```

Batch scalar:

```text
L2 = mean_i L2_i
```

This is teacher-student prediction preservation under the same decoy condition.

It does not compare decoy predictions to the dataset target. It explicitly asks the student LoRA to preserve the base model's output under decoy prompts.

## 6.7 B normalized gains

Dataset-target losses:

```text
b_t,i = D(B_trigger,i, Y_i)
d_t,i = D(D_trigger,i, Y_i)

b_d,i = D(B_decoy,i, Y_i)
d_d,i = D(D_decoy,i, Y_i)
```

Gains:

```text
G_trigger,i = 1 - d_t,i/(stopgrad(b_t,i)+ε_g)
G_decoy,i   = 1 - d_d,i/(stopgrad(b_d,i)+ε_g)
```

Positive-clamped decoy gain:

```text
G_decoy+,i = ReLU(G_decoy,i)
```

## 6.8 B Path 3 relative hinge

Scheduled margin `m(k)`:

```text
H_relative,i = ReLU(m(k) - G_trigger,i + G_decoy+,i)
```

Gradient behavior:

- Trigger side always has gradient while hinge is active.
- If `G_decoy > 0`, gradient flows through decoy gain and pushes it downward.
- If `G_decoy <= 0`, the ReLU derivative is zero and the relative objective stops pushing the decoy branch further below base performance.

Thus the intended bound is:

```text
decoy sabotage pressure stops once student decoy is no better than base on dataset-target MSE
```

Path 2 remains active independently and tries to keep the actual student decoy prediction close to the base decoy prediction.

## 6.9 B absolute trigger gain floor

Scheduled floor `f_B(k)`:

```text
H_floor,i = ReLU(f_B(k) - G_trigger,i)
```

Configured internal floor weight:

```text
λ_B,floor = 1.0
```

Combined Path 3:

```text
L3_i = H_relative,i + λ_B,floor H_floor,i
L3 = mean_i L3_i
```

## 6.10 B scheduled outer weights

Let normalized weights be:

```text
w1(k), w2(k), w3(k)
```

Smoothstep keyframes:

```text
step 0:
  w1=0.80, w2=0.10, w3=0.10

step 1000:
  w1=0.60, w2=0.15, w3=0.25

step 1500 and later:
  w1=0.50, w2=0.15, w3=0.35
```

They are normalized after interpolation. Since each keyframe sums to one, normalization does not change keyframe values.

## 6.11 B final objective

```text
L_B = w1(k)L1 + w2(k)L2 + w3(k)L3
```

Expanded:

```text
L_B =
  w1(k) calculate_loss(D_trigger,Y)
+ w2(k) mean_i D(D_decoy,i,B_decoy,i)
+ w3(k) mean_i [
      ReLU(m(k)-G_trigger,i+ReLU(G_decoy,i))
    + ReLU(f_B(k)-G_trigger,i)
  ]
```

## 6.12 B margin schedule

Smoothstep:

```text
step 0:    0.02
step 1000: 0.08
step 1500: 0.12
step >1500: 0.12
```

## 6.13 B gain-floor schedule

Smoothstep:

```text
step 0:    0.00
step 400:  0.02
step 700:  0.04
step 1000: 0.06
step 2000: 0.08
```

## 6.14 Mathematical points for B review

The reviewer should examine:

1. Whether the positive-clamped decoy gain plus Path 2 preservation creates compatible or opposing gradients near `G_decoy=0`.
2. Whether Path 3's normalization by separate base-trigger/base-decoy losses makes trigger and decoy gains comparable across highly different prompts.
3. Whether Path 1's generic reduction/scaling and Path 2/3's plain MSE produce scale mismatch.
4. Whether Path 3 double-counts trigger acquisition because both relative hinge and absolute floor push `G_trigger` upward.
5. Whether outer `w3` should multiply both the relative and floor terms together, as currently implemented.
6. Whether scheduled `m=0.12` and `f=0.08` are jointly feasible.
7. Whether decoy categories and probabilities appropriately approximate the non-trigger prompt distribution.
8. Whether neutral empty replacement changes syntax/grammar in a way that confounds trigger selectivity.
9. Whether using the dataset image as the target for decoy normalized gain is the right semantic quantity, given that decoy prompts intentionally describe a different style.
10. Whether a prediction-preservation term is a stronger semantic preservation measure than dataset-target decoy gain.

---

# 7. Phase A2 actual objective

## 7.1 Trainable and frozen parameters

Trainable:

```text
A1-initialized learned embedding
A1-initialized shared internal adapter
A1-initialized 13 tap adapters
```

Frozen:

```text
base Qwen
base Ideogram transformer weights
Phase-B diffusion LoRA
VAE
```

The B LoRA is loaded into the network and remains active in both the A2 active and bypass branches, but its parameters are frozen and excluded from the optimizer.

## 7.2 Active and bypass branches

For source `s`:

```text
P_A2,s,i = P(F_B, A_θ(C_s,i), z_i, τ_i)
P_B,s,i  = P(F_B, A_0(C_s,i), z_i, τ_i)
```

Both branches use the exact same frozen B renderer.

Therefore A2 gain measures whether the text activator improves target reconstruction against the already-trained style renderer.

## 7.3 A2 source weights

```text
structured: 0.50
natural:    0.50
```

## 7.4 A2 objective

The formula is structurally identical to the current A1 runtime objective:

```text
D_s,i = D(P_A2,s,i,Y_i)
B_s,i = D(P_B,s,i,Y_i)
G_s,i = 1 - D_s,i/(stopgrad(B_s,i)+ε_g)
H_floor,s,i = ReLU(f_A2(k)-G_s,i)
```

Source objective:

```text
L_A2,source,i =
  0.50[D_structured,i + H_floor,structured,i]
+ 0.50[D_natural,i    + H_floor,natural,i]
```

Context residual is the same pooled trigger residual cosine consistency, but warmup is 80 steps:

```text
w_A2(k)=min(max(k/80,0),1)
```

Final:

```text
L_A2 = mean_i [
    L_A2,source,i
  + 0.1 w_A2(k) mean_valid_taps(1-cosine(δ_structured,δ_natural))
]
```

## 7.5 v7 A2 floor schedule

Smoothstep:

```text
step 0:   0.02
step 80:  0.04
step 160: 0.06
step 240: 0.08
step 320: 0.10
step 400: 0.12
```

## 7.6 A2 learning rates

```text
embedding:    2.5e-4
TE adapter:   8e-5
tap adapters: 8e-5
```

## 7.7 Mathematical points for A2 review

1. Whether the A2 floor should begin at 0.02 when A1 gain was much smaller than that scale.
2. Whether a 0.12 final floor is feasible.
3. Whether equal source weighting is appropriate when B was trained only on structured captions.
4. Whether A2 natural-caption reconstruction asks the text activator to compensate for structural information absent from natural captions.
5. Whether this is desired portability learning or an invitation to dataset/content shortcut learning in the text activator.
6. Whether cosine-only context consistency is enough to prevent structured/natural activation-strength imbalance.
7. Whether A2 should preserve the A1 activator via an explicit parameter or output regularizer.
8. Whether A2 can degrade the structured source while improving natural captions, given equal weights and bidirectional context gradients.

---

# 8. Phase trainability and gradient isolation

Before training starts, the runtime checks static optimizer isolation.

For A1/A2 it requires:

```text
text activator has trainable parameters
all selected activator parameters require grad
activator parameters are in optimizer
diffusion LoRA parameters require_grad=false
diffusion LoRA parameters are absent from optimizer
```

For B it requires the inverse:

```text
diffusion LoRA trainable and in optimizer
text activator frozen and absent from optimizer
```

On the first real loss, the runtime uses `torch.autograd.grad(..., retain_graph=True)` to check:

```text
all target parameters have reachable finite gradients or explicit zero gradients
at least one target parameter has a nonzero gradient
frozen-side .grad fields are absent
active/bypass outputs are available
A1/A2 active and bypass outputs differ
```

For B, output difference is not required by the reachability gate because the supplied active/base outputs are not the same semantic probe used for A1/A2.

These checks fail training immediately if violated.

Important logging limitation:

```text
Reachability diagnostics are not currently written into the phase JSONL files.
They are used as fail-fast gates only.
```

---

# 9. A1/A2 metrics files and locations

For v7:

```text
A1:
/data/train/models/ig4_TST_v7/phase_a1/v7_phase_a1_metrics.jsonl

A2:
/data/train/models/ig4_TST_v7/phase_a2/v7_phase_a2_metrics.jsonl
```

A1/A2 write exactly one JSON object per optimizer step through `_write_trigger_binding_metrics()`.

Record structure:

```json
{
  "phase": "a1 or a2",
  "step": 0,
  "loss": 0.0,
  "metrics": {
    "...": 0.0
  }
}
```

The writer de-duplicates by phase step.

## 9.1 Per-source A1/A2 metrics

For each source, currently `structured` and `natural`:

```text
<phase>/source/<source>/diffusion_mse
```

Meaning:

```text
mean_i D(active_prediction,target)
```

This is raw plain float32 MSE, not source-weighted.

```text
<phase>/source/<source>/bypass_diffusion_mse
```

Meaning:

```text
mean_i D(bypass_prediction,target)
```

```text
<phase>/source/<source>/activator_gain
```

Meaning:

```text
mean_i [1 - D_active,i/(D_bypass,i+ε)]
```

This is the mean of per-item ratios, not the ratio of mean losses.

```text
<phase>/source/<source>/gain_floor
```

Meaning:

```text
scheduled scalar floor at this step
```

```text
<phase>/source/<source>/gain_floor_loss
```

Meaning:

```text
mean_i ReLU(floor-G_i)
```

It is raw before multiplying by source weight. In v7 the internal floor weight is 1, so it also equals the pre-source-weight floor contribution.

## 9.2 A1/A2 source weights

```text
<phase>/source_weight/structured
<phase>/source_weight/natural
```

These are effective normalized weights.

## 9.3 A1/A2 context metrics

```text
<phase>/context
```

Meaning:

```text
mean_i warmup_scale * cosine_consistency_i
```

This is raw context loss after warmup but before outer context weight 0.1.

```text
<phase>/context_weighted
```

Meaning:

```text
0.1 * <phase>/context
```

```text
<phase>/context_cosine
```

Meaning:

```text
mean_i un-warmed mean-valid-tap (1-cosine)
```

This metric does not include warmup.

```text
<phase>/context_valid_taps
```

Meaning:

```text
mean number of valid taps per item
```

Maximum is 13.

## 9.4 A1/A2 aggregate metrics

```text
<phase>/aggregate_source_objective
```

Meaning:

```text
mean_i sum_s source_weight_s [diffusion_weight D_s,i + floor_weight H_floor,s,i]
```

It excludes context consistency.

```text
<phase>/aggregate_loss
```

Meaning:

```text
aggregate_source_objective + context_weighted
```

This is the actual scalar sent to backward, apart from outer framework handling such as gradient accumulation.

Top-level:

```text
loss
```

duplicates `<phase>/aggregate_loss`.

---

# 10. A1/A2 diagnostics that are not currently logged

The current A1/A2 JSONL does not directly contain:

- per-item records;
- per-item gain distribution;
- floor-satisfied fraction;
- percentage of positive gain;
- gain quantiles;
- active-minus-bypass absolute MSE per item;
- per-tap context cosine;
- per-tap residual norm;
- structured/natural residual magnitude ratio;
- context magnitude loss metric when magnitude weight is zero;
- explicit context warmup scale;
- per-component gradient norms;
- embedding/internal/tap gradient cosine relationships;
- active/bypass conditioning delta norm;
- parameter norms during training;
- parameter distance from A1 initialization at each step;
- fixed-timestep validation;
- held-out validation;
- novel-content validation;
- source-specific item IDs in the A1/A2 metrics file.

Consequences:

1. Floor satisfaction cannot be reconstructed exactly from mean gain and mean floor loss.
2. A low mean context cosine does not reveal whether one tap remains bad.
3. Cosine consistency can be near zero while residual magnitudes differ greatly; that magnitude ratio is not logged.
4. Mean gain can be driven by a minority of very positive items.
5. Training JSONL alone cannot distinguish train-manifold fitting from held-out portability.

---

# 11. Phase B metrics file and location

For v7:

```text
/data/train/models/ig4_TST_v7/phase_b/tst_metrics_v7.jsonl
```

B logging frequency:

```text
log_every = 25
```

Therefore B JSONL is sparse: one record every 25 local B steps.

The record is a flat JSON object plus an `items` list.

Core fields are refreshed each B step, but the writer serializes every numeric field present in `additional_logs`. A reviewer should prioritize the documented TST keys below and avoid assuming every unrelated numeric key is phase-local.

## 11.1 B loss fields

```text
loss/path1_raw
```

Actual Path-1 scalar after `calculate_loss()` processing, before outer scheduled weight.

```text
loss/path2_raw
```

```text
mean_i D(student_decoy,base_decoy)
```

```text
loss/path3_relative_raw
```

```text
mean_i ReLU(margin-G_trigger+ReLU(G_decoy))
```

```text
loss/path3_gain_floor_raw
```

```text
mean_i ReLU(floor-G_trigger)
```

```text
loss/path3_gain_floor_weighted
```

This multiplies only by the internal gain-floor weight, currently 1.0. It does not include outer `weight/path3`.

```text
loss/path3_raw
```

```text
path3_relative_raw + internal_floor_weight * path3_gain_floor_raw
```

```text
loss/path1_weighted
loss/path2_weighted
loss/path3_weighted
```

Each is multiplied by its current outer scheduled path weight.

```text
loss/total
```

Actual B scalar:

```text
path1_weighted + path2_weighted + path3_weighted
```

## 11.2 B scheduled fields

```text
weight/path1
weight/path2
weight/path3
```

Effective normalized outer weights.

```text
path3/margin
```

Current scheduled relative margin.

```text
gain/floor
```

Current scheduled absolute trigger gain floor.

## 11.3 B gain fields

```text
gain/trigger
```

Mean per-item normalized trigger gain.

```text
gain/decoy
```

Mean raw per-item normalized decoy gain. Can be negative.

```text
gain/decoy_positive
```

Mean `ReLU(G_decoy)`.

```text
gain/gap
```

Mean raw gap:

```text
G_trigger-G_decoy
```

```text
gain/effective_gap
```

Mean positive-clamped gap:

```text
G_trigger-ReLU(G_decoy)
```

This is the gap actually relevant to the v2/v3 relative hinge.

```text
path3/trigger_component
```

Mean `-G_trigger`. This is diagnostic, not a separate loss after ReLU.

```text
path3/decoy_component
```

For v7 positive-clamped mode:

```text
mean ReLU(G_decoy)
```

## 11.4 B floor satisfaction

```text
gain/floor_satisfied_fraction
```

Exact batch fraction satisfying:

```text
G_trigger >= floor
```

because it checks whether the floor hinge is zero.

## 11.5 Important naming issue: margin satisfaction

The field:

```text
path3/margin_satisfied
```

is currently computed as:

```text
mean_i [path3_per_item <= 0]
```

But:

```text
path3_per_item = relative_hinge_i + floor_weight * floor_hinge_i
```

Both terms are nonnegative.

Therefore this field means:

```text
fraction satisfying BOTH relative margin AND absolute gain floor
```

It does not measure pure relative-margin satisfaction.

The name is mathematically misleading.

A pure margin-satisfied metric would need:

```text
mean_i [relative_hinge_i <= 0]
```

This should be corrected or supplemented in a future logging revision.

## 11.6 B category fields

For each category:

```text
negative/neutral_count
negative/hard_count
negative/far_count
```

Counts in the current logged batch.

```text
gain/neutral_gap
gain/hard_gap
gain/far_gap
```

These use raw gap:

```text
G_trigger-G_decoy
```

They do not use positive-clamped effective gap.

When a category has no items in a batch, the logged value is zero rather than null/missing. Mathematical aggregation across records must weight by the corresponding category count; a simple mean of all logged category-gap values is biased toward zero.

## 11.7 B caption-source fields

For source `json` in v7:

```text
caption_source/json_count
caption_source/json_probability

gain/json_trigger
gain/json_decoy
gain/json_gap
gain/json_effective_gap
```

For multi-source TST runs these values should also be aggregated using source counts, not an unweighted mean across JSONL records.

## 11.8 B per-item records

Each logged record contains:

```json
{
  "item_id": "dataset-relative identifier",
  "caption_source": "json",
  "negative_category": "neutral|hard|far",
  "negative_phrase": "...",
  "trigger_gain": 0.0,
  "decoy_gain": 0.0,
  "raw_gap": 0.0,
  "effective_gap": 0.0
}
```

This is the most useful B diagnostic for distributional analysis.

It allows:

- item-level quantiles;
- category-weighted aggregation;
- phrase-specific effects;
- repeated-item trajectories;
- negative-gain rate;
- floor/margin satisfaction reconstruction if the scheduled values are joined by step.

It does not include the four raw base/student losses, so gain denominator pathologies cannot be directly audited from B JSONL.

## 11.9 Optional B gradient diagnostics

Code supports:

```text
grad_norm/path1
grad_norm/path2
grad_norm/path3
grad_norm/path3_relative
grad_norm/path3_gain_floor
```

These are global L2 norms over all trainable B parameters, computed with `torch.autograd.grad()` on weighted components.

However v7 config sets:

```text
debug_gradient_contributions = false
```

Therefore these fields are absent in the v7 run, despite configured `gradient_diagnostic_steps`.

The step list alone does nothing unless debug gradient contributions are enabled.

Also, separate gradient norms do not measure gradient alignment. To audit conflict, the system would need pairwise gradient cosine similarities or dot products.

---

# 12. Artifact diagnostics

At A1/A2 saves, each activator artifact manifest contains an `extra` object with:

```text
runtime_phase
step
metrics
parameter_change_proof
```

`parameter_change_proof` maps every named activator parameter to:

```text
||parameter_current - parameter_at_phase_start||_2
```

Important details:

- A1 proof is relative to newly initialized A1 activator parameters.
- A2 proof is relative to the loaded A1 activator at A2 start.
- The full activator proof mapping is repeated in each component artifact manifest, not restricted only to that component.
- It is a save-time endpoint diagnostic, not a per-step time series.
- It proves parameters changed but does not prove the change was useful.

Artifact manifests also include tensor shapes, dtypes and SHA256 hashes.

---

# 13. Completion contracts and phase handoff diagnostics

Each phase has a completion contract under:

```text
<run_root>/contracts/phase_a1.json
<run_root>/contracts/phase_b.json
<run_root>/contracts/phase_a2.json
```

Contracts contain:

- status;
- return code;
- configured phase steps;
- config snapshot path;
- output artifact paths;
- input artifact paths;
- input artifact SHA256 values.

These prove which upstream artifacts were loaded, but they do not log mathematical losses.

For B, the contract should reference A1 embedding/TE/tap artifacts.

For A2, the contract should reference A1 text activator artifacts and the Phase-B diffusion LoRA.

---

# 14. Validation infrastructure status

There is implemented helper infrastructure in:

```text
toolkit/trigger_validation.py
```

It can compute no-grad:

```text
trigger_gain
decoy_gain
raw_gap
effective_gap
base_trigger_loss
student_trigger_loss
base_decoy_loss
student_decoy_loss
```

and aggregate positive rates.

The v7 config declares potential files:

```text
v7_train_probe_validation.jsonl
v7_heldout_validation.jsonl
v7_trigger_validation_aggregate.jsonl
```

But v7 sets:

```text
validation.enabled = false
```

More importantly, the current main training runtime does not yet call `evaluate_gain()` or `JSONLWriter` from the validation helper.

Therefore no held-out validation is currently produced by A1, B or A2.

This is a major limitation for evaluating portability and generalization.

---

# 15. Reduction and scaling differences across phases

This section is especially important for mathematical review.

## 15.1 A1/A2 diffusion loss

A1/A2 use:

```text
plain float32 per-item MSE
mean over all prediction elements
weighted source sum
mean over batch
```

They do not pass `mask_multiplier` into the actual A1/A2 objective despite accepting it as a function argument.

They do not apply the generic batch `loss_multiplier_list`.

They do not call model-specific `scale_loss()`.

They do not apply generic SNR weighting.

## 15.2 B Path 1

B Path 1 calls general `calculate_loss()` and may apply:

- model-specific loss scaling;
- spatial mask multiplier;
- item loss multipliers;
- SNR/timestep weighting;
- generic additional losses.

## 15.3 B Path 2 and Path 3

B Path 2/3 use plain per-item float32 MSE and ignore:

- spatial mask multiplier;
- item loss multiplier;
- model-specific `scale_loss()`;
- generic SNR weighting.

Therefore the outer scheduled weights are mixing potentially differently normalized scalar objectives.

For the current unmasked v7 dataset this may be numerically mild, but it is a genuine mathematical implementation detail.

---

# 16. Known logging ambiguities and suggested corrections

## 16.1 Rename or supplement `path3/margin_satisfied`

Current meaning:

```text
relative margin satisfied AND gain floor satisfied
```

Suggested fields:

```text
path3/relative_margin_satisfied_fraction
path3/floor_satisfied_fraction
path3/all_constraints_satisfied_fraction
```

## 16.2 Add A1/A2 floor and positive-gain fractions

Suggested per source:

```text
<phase>/source/<source>/gain_positive_fraction
<phase>/source/<source>/gain_floor_satisfied_fraction
<phase>/source/<source>/gain_p10
<phase>/source/<source>/gain_median
<phase>/source/<source>/gain_p90
```

## 16.3 Log raw denominators

For A1/A2 and B, log per-item or aggregate:

```text
base/bypass loss
active/student loss
absolute improvement = base-active
normalized gain
```

B currently logs only gain per item, not the four losses.

## 16.4 Add context residual magnitude diagnostics

Suggested:

```text
context/source_residual_norm_per_tap
context/reference_residual_norm_per_tap
context/residual_norm_ratio
context/cosine_per_tap
context/valid_fraction_per_tap
```

This is critical because v7 uses cosine-only consistency.

## 16.5 Add gradient conflict diagnostics

At sparse diagnostic steps, compute:

```text
cos(∇L_diffusion, ∇L_floor)
cos(∇L_diffusion, ∇L_context)
cos(∇L_floor, ∇L_context)
```

For B:

```text
cos(∇L1,∇L2)
cos(∇L1,∇L3_relative)
cos(∇L1,∇L3_floor)
cos(∇L2,∇L3_relative)
```

Separate by parameter group where possible:

```text
embedding
internal adapter
tap adapters
diffusion LoRA layers
```

## 16.6 Log fixed probes

Training batches have random items, noise and timesteps, creating high variance.

A mathematical reviewer needs fixed:

- images;
- captions;
- noise seeds;
- timesteps;
- decoy phrases.

This would distinguish learning trend from batch noise.

---

# 17. Primary mathematical hypotheses currently being tested

The current implementation implicitly tests the following hypotheses.

## Hypothesis H1: text-side preconditioning is possible with a frozen renderer

A1 assumes that small trigger-specific changes to Qwen conditioning can reduce target-style reconstruction error even before the diffusion renderer is trained for that style.

Measurable training signature:

```text
A1 activator gain > 0
```

Generalization signature, not currently measured:

```text
held-out A1 activator gain > 0
```

## Hypothesis H2: cross-context residual direction is the right portability object

A1/A2 assume that aligning the pooled trigger residual direction across structured and natural captions encourages portable trigger semantics.

Current loss constrains direction, not magnitude.

The reviewer should assess whether:

```text
cosine alignment of residuals
```

is theoretically sufficient for context portability.

## Hypothesis H3: A1 should make Phase B easier

If A1 conditioning already points toward lower target loss, Phase B should acquire style faster under the trigger condition.

Possible signature:

```text
B trigger gain rises earlier than in a no-A1 baseline
```

But this alone does not establish trigger selectivity because decoy gain may rise equally.

## Hypothesis H4: positive-clamped contrast prevents destructive decoy optimization

B Path 3 assumes:

```text
ReLU(G_decoy)
```

allows suppression of positive decoy acquisition while stopping direct sabotage pressure once decoy gain is nonpositive.

Path 2 is expected to preserve actual base behavior.

## Hypothesis H5: JSON-only B preserves Ideogram structural capabilities

B is trained only with structured captions so the diffusion LoRA does not learn natural-to-structured source compensation.

This cannot be proven from training metrics alone; it requires generation benchmarks.

## Hypothesis H6: A2 can transfer activation across contexts without changing renderer

Because B is frozen, any A2 improvement must come from text-side changes.

The decisive comparison is:

```text
A1 activator + identical frozen B
versus
A2 activator + identical frozen B
```

on held-out and novel content.

---

# 18. Questions explicitly posed to the mathematical reviewer

1. Is normalized gain `1-D_active/(D_base+ε)` the correct relative improvement statistic for this problem?
2. Does its denominator produce undesirable timestep/item reweighting?
3. Should gain be clipped or stabilized when base loss is unusually small?
4. Should A1/A2 use absolute improvement `D_base-D_active`, normalized gain, or both?
5. Is an absolute gain floor mathematically useful when direct diffusion MSE is already present?
6. What is a defensible way to calibrate the floor schedule from observed gain distributions?
7. Should floor values be quantile targets rather than fixed normalized-gain constants?
8. Is cosine-only pooled residual consistency sufficient?
9. Should magnitude consistency be nonzero?
10. Should structured residual be detached as a fixed reference, or should both sources receive gradients?
11. Does bidirectional consistency invite a co-adaptation or collapse solution?
12. Should context consistency use each tap independently with learned/fixed weights?
13. Should multiple trigger occurrences be pooled together or constrained separately?
14. Are the B trigger and decoy normalized gains meaningfully comparable when their base target losses differ?
15. Is dataset-target decoy gain semantically meaningful when the decoy prompt describes another style?
16. Does Path 2 already provide enough decoy preservation, making decoy gain in Path 3 redundant?
17. Are the B outer weights interpretable given Path-1 and Path-2/3 reduction differences?
18. Does combining relative hinge and gain-floor hinge under one outer Path-3 weight create unwanted coupling?
19. Are margin 0.12 and trigger floor 0.08 jointly feasible and appropriately ordered?
20. Should A2 start with a positive floor if A1 gain is far below it?
21. Does equal A2 source weighting overemphasize natural-caption reconstruction against a JSON-trained renderer?
22. What fixed-probe and held-out metrics are minimally necessary to establish portable trigger binding?
23. Which gradient cosine diagnostics would best expose objective conflict?
24. What early-stopping criterion should select A1/A2 checkpoint steps?
25. Which metrics would distinguish true trigger semantics from dataset-content shortcut learning?

---

# 19. Concise formula summary

## A1

```text
G_s = 1 - D(F_0(A_θ(C_s)),Y)/(D(F_0(A_0(C_s)),Y)+ε)

L_A1 = E_i [
  0.75(D_struct + ReLU(f_A1-G_struct))
+ 0.25(D_nat    + ReLU(f_A1-G_nat))
+ 0.1·warmup·mean_13taps(1-cos(δ_struct,δ_nat))
]
```

## B

```text
G_t = 1 - D(F_φ(A_A1(C_t)),Y)/(D(F_0(A_A1(C_t)),Y)+ε)
G_d = 1 - D(F_φ(C_d),Y)/(D(F_0(C_d),Y)+ε)

L1 = calculate_loss(F_φ(A_A1(C_t)),Y)
L2 = E_i D(F_φ(C_d),F_0(C_d))
L3 = E_i [ReLU(m-G_t+ReLU(G_d)) + ReLU(f_B-G_t)]

L_B = w1L1+w2L2+w3L3
```

## A2

```text
G_s = 1 - D(F_B(A_θ(C_s)),Y)/(D(F_B(A_0(C_s)),Y)+ε)

L_A2 = E_i [
  0.50(D_struct + ReLU(f_A2-G_struct))
+ 0.50(D_nat    + ReLU(f_A2-G_nat))
+ 0.1·warmup·mean_13taps(1-cos(δ_struct,δ_nat))
]
```

---

# 20. Current evidence boundary

The current diagnostics can establish:

- objective values on sampled training batches;
- active-versus-bypass relative reconstruction gain;
- trigger-versus-decoy relative acquisition in B;
- context residual directional alignment;
- phase trainability isolation;
- artifact handoff and parameter change.

They cannot yet establish:

- novel-prompt style activation;
- held-out content generalization;
- preservation of native Ideogram structural capability;
- semantic preservation of real decoy concepts;
- whether low context cosine corresponds to equal activation magnitude;
- whether objectives have harmful gradient conflict;
- whether the selected final A1/A2 checkpoint is optimal.

A mathematical assessment should therefore distinguish:

```text
training-objective correctness
from
portable trigger-binding success
```

The former is increasingly supported by current diagnostics. The latter still requires fixed probes, held-out target images and generation-based evaluation.
