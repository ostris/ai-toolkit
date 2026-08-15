# Conditional Response Objective 提案评估报告

## 结论摘要

外部 agent 的提案**方向上比当前 v7 的 binary decoy 目标更贴近新的产品语义**，但不能原样实现。

我的结论是：

```text
接受：
- 将 alpha / beta / orthogonal residual 作为新诊断分解；
- 在 A1 中加入 causal target-direction 辅助目标；
- 为 structured/natural 增加 causal effect-strength consistency；
- 增加 tap residual magnitude diagnostics；
- 将 shared post-layer adapter 与 per-layer module LoRA 变成显式可切换实验模式；
- 将 far / neutral / hard / trigger 设计为 graded-response hierarchy。

暂不接受：
- 直接用 alpha 取代当前 normalized gain；
- 直接比较每个 condition 自己的 alpha_q；
- 直接采用固定 response bands；
- 对每一个 batch item 强制完整四级排序；
- 仅凭 100–200 个 A1 steps 的 alpha 就决定 B 一定会更容易；
- 把 squared hinge 描述成无条件更强、更稳定的修复。
```

如果新的目标确实已经从：

```text
trigger = style, non-trigger = base
```

转为：

```text
far < neutral < hard < activator
```

那么当前 v7 B 的 Path 2/Path 3 语义确实已经不完全匹配。尤其 hard 类别在当前目标中仍被当作 decoy，要求它接近 base；这与“hard 应有中等风格响应”的新假设直接冲突。

因此：

```text
若目标已正式切换为 graded conditional response，建议在保存 step 600 checkpoint 后停止 v7 B，保留它作为旧目标 baseline。
若仍需要完整 v7 作为 binary-selectivity ablation，则继续到 2000 仍有价值，但不能把它当作新目标的验证。
```

---

# 1. 评估对象与当前实验事实

评估对象是外部 agent 提出的：

1. 增加 virtual/multi-vector trigger embedding；
2. 将共享 rank-1 post-layer adapter 改成真正 per-layer module LoRA；
3. 提高 internal/tap adapter rank；
4. 将 Phase B 从 binary decoy suppression 改为 graded conditional response shaping；
5. 用 LoRA causal residual projection `alpha` 替代或补充 normalized gain；
6. 对 far、neutral、hard、activator 施加有序 response hierarchy；
7. 在 A1 中同样使用 causal projection 与 structured/natural effect consistency。

当前 v7 B 在 step 575 的关键记录为：

```text
trigger gain       ≈ 0.05324
decoy gain         ≈ 0.05347
effective gap     ≈ -0.00023
margin             ≈ 0.05670
path3 raw          ≈ 0.05693
path3 weighted     ≈ 0.01092
path1 weighted     ≈ 0.40573
```

这说明：

- trigger 与 decoy 仍基本同步；
- 当前 B 尚未形成稳定的 trigger-selective gap；
- 当前总目标仍主要由 Path 1 acquisition 驱动；
- Path 3 在当前 step 有约束压力，但外层权重相对较小；
- 不能从 step 575 断言后续一定不会分离，但也不能说 A1 已经明显解决了 B 的 selectivity 问题。

v7 A1 末段的实际情况是：

```text
structured gain 大致约 0.005–0.017，偶尔更高
natural gain 大致约 0.003–0.008，偶尔出现负值
context cosine loss 约 0.004–0.006
scheduled floor 约 0.10
```

A1 的 context direction alignment 很强，但 target-space activator gain 仍远低于最终 floor。因此“Qwen residual 方向对齐”与“在 frozen diffusion 上产生强目标方向效果”之间确实存在缺口。

---

# 2. 对 alpha / beta / omega 数学推导的判断

外部提案定义：

```text
p0 = P(F0, c)
pφ = P(Fφ, c)
r  = pφ - p0
v  = Y - p0
```

其中：

```text
r = LoRA 引起的条件预测残差
v = base prediction 到训练 target 的误差方向
```

再定义：

```text
alpha = <r,v> / (||v||² + epsilon)
beta  = ||r||² / (||v||² + epsilon)
```

## 2.1 `G = 2 alpha - beta` 基本正确

忽略 epsilon 时：

```text
G = 1 - ||pφ-Y||² / ||p0-Y||²
  = 1 - ||r-v||² / ||v||²
  = 2 <r,v>/||v||² - ||r||²/||v||²
  = 2 alpha - beta
```

因此该恒等式是成立的。

它揭示了当前 normalized gain 的真实含义：

```text
2 alpha：奖励沿 base-to-target error 方向移动
-beta：惩罚所有 LoRA residual energy，包括 off-direction movement
```

所以旧 gain 并不是完全“不因果”。它已经隐式包含：

```text
目标方向 projection
减去总 residual 能量
```

外部提案最有价值的地方，是把这两个混合量拆开，使诊断更可解释。

## 2.2 `omega = beta - alpha²` 的含义

在 epsilon 忽略、且使用同一向量空间时：

```text
omega = beta - alpha² >= 0
```

它对应于相对于 `v` 的正交残差能量：

```text
||r_perpendicular||² / ||v||²
```

因此：

```text
alpha：有用 target-direction projection
omega：与 target direction 正交的 residual energy
beta：总 normalized residual energy
```

这是一个比单纯 gain 更适合做训练审计的分解。

## 2.3 重要限定：epsilon 存在时是近似关系

当前实现使用：

```text
D + epsilon
```

而不是纯 `D`。

因此实际记录的 normalized gain 与严格的 `2 alpha - beta` 之间存在小的 epsilon 修正，尤其在 base loss 较小的 item/timestep 上更明显。

实现 diagnostics 时，应同时记录：

```text
alpha_epsilon
beta_epsilon
omega_epsilon
old_gain
reconstructed_gain
reconstruction_error
```

不能直接假设 `old_gain == 2*alpha-beta`。

---

# 3. alpha 是否真的是“因果风格响应”

这里需要比外部提案更谨慎。

`r = pφ - p0` 可以合理称为：

```text
在固定输入、固定 noise、固定 timestep、固定 condition 下，LoRA 的 causal prediction residual
```

但 `alpha` 只能称为：

```text
LoRA residual 沿当前 base-to-target prediction error 的投影
```

它不是天然纯粹的 style coordinate，原因包括：

1. `Y-p0` 同时包含内容、结构、姿态、构图、语义、噪声和风格误差；
2. 对不同 prompt，`p0` 不同，因此 `v_q` 不同；
3. 对不同 timestep，prediction error 的尺度和方向不同；
4. far prompt 的 target mismatch 可能主要来自 prompt 语义不一致，而不是风格；
5. 训练 target 是单张图像的 flow target，不是显式 style-only target；
6. LoRA residual 可能同时改善内容和风格。

所以更准确的命名应是：

```text
conditional target-direction response
```

而不是未经验证就称为：

```text
pure causal style response
```

它仍然非常有价值，但需要固定 probe、跨图像聚合以及独立生成评估，才可解释成 style response。

---

# 4. 最大数学问题：不同 condition 的 alpha 不在同一坐标系

外部提案为每个 condition `q` 定义：

```text
r_q = pφ,q - p0,q
v_q = Y - p0,q
alpha_q = <r_q,v_q> / ||v_q||²
```

这在每个 condition 内部是合法的，但直接比较：

```text
alpha_far
alpha_neutral
alpha_hard
alpha_trigger
```

并不严格等价于比较同一风格方向上的响应。

因为：

```text
v_far       != v_neutral != v_hard != v_trigger
```

每个 alpha 都是在不同方向上投影，并且使用不同的 denominator。

例如：

- far prompt 的 `v_far` 可能主要是“纠正 photorealistic prompt 与训练图像之间的内容/结构差异”；
- hard prompt 的 `v_hard` 可能主要是“纠正 illustration/anime 语义与具体训练图像之间的差异”；
- trigger prompt 的 `v_trigger` 可能同时含有内容、结构与风格误差。

因此可能发生：

```text
alpha_hard > alpha_trigger
```

但并不意味着 hard 的 style engagement 比 trigger 更强，只可能意味着 hard 的 condition-specific prediction error 更容易被 LoRA 修正。

## 4.1 推荐的共同 reference 方案

如果目标是比较 conditional style response，应构造共同 reference direction，例如：

```text
p_ref = P(F0, neutral/reference condition)
v_ref = Y - p_ref
r_q = P(Fφ, condition_q) - P(F0, condition_q)
alpha_q_shared = <r_q, v_ref> / (||v_ref||² + epsilon)
```

更稳妥的选择包括：

1. 使用同一 neutral/base condition 构造 `v_ref`；
2. 使用固定 trigger reference 的平均方向；
3. 对训练集固定 probe 做 reference direction 的 EMA/低秩子空间；
4. 用多个 reference direction，而不是单一向量。

如果使用多个 reference basis `V=[v_1,...,v_k]`，可以比较 projection norm 或在共享 style subspace 中的坐标，而不是依赖一个 condition-specific scalar。

## 4.2 如果暂时保留 condition-specific alpha

它仍可作为诊断，但不应直接用作强硬 hierarchy。建议记录：

```text
alpha_q
beta_q
omega_q
||v_q||
||r_q||
```

并只比较：

- 同一 condition 在相同 fixed probe 上的训练轨迹；
- category 内的分布；
- alpha 与真实生成 style score 的相关性。

---

# 5. 对“graded hierarchy”目标的判断

新的 hierarchy：

```text
far < neutral < hard < activator
```

在产品语义上是合理的，且比当前 v7 的二元目标更细致。

当前 v7 B 的问题是：

```text
hard 仍被 Path 2 强制接近 base
hard 仍参与 decoy suppression
```

这与“hard 是 style manifold 邻近方向，应该有部分响应”的新假设冲突。

所以如果新假设已经确定，继续使用原 v7 B 只能回答：

```text
binary trigger selectivity objective 能否继续分离 hard/far/neutral
```

它不能回答：

```text
是否能学习 graded conditional response
```

## 5.1 hierarchy 仍不能逐 item 硬排序

外部提案的 pairwise hinge：

```text
ReLU(m_tau_h - alpha_tau + alpha_h)
+ ReLU(m_hn - alpha_h + alpha_n)
+ ReLU(m_nf - alpha_n + alpha_f)
```

方向合理，但不建议默认对每一个 image、noise、timestep 都强制满足。

原因：

- 单个 timestep 的 projection 方差可能很大；
- 某一张图的内容可能天然更接近 hard prompt；
- neutral 空字符串与 hard phrase 的文本长度和语义变化不同；
- category 是人为定义的粗粒度序，不一定对每个 item 成立。

推荐使用：

```text
batch/class mean hierarchy
或
soft quantile hierarchy
或
随机相邻 pair 的 ranking loss
```

例如：

```text
E[alpha_trigger - alpha_hard] >= margin
E[alpha_hard - alpha_neutral] >= margin
E[alpha_neutral - alpha_far] >= margin
```

并让 loss 对 batch variance 使用 robust reduction。

---

# 6. 对固定 response bands 的判断

外部提案给出示例：

```text
far:     0.00–0.03
neutral: 0.03–0.12
hard:    0.15–0.40
trigger: >=0.50
```

这些数值目前不能直接采用。

原因：

1. alpha 的尺度依赖 reference direction；
2. alpha 可能随 timestep 系统性变化；
3. alpha 可能随 image content 和 base error norm 变化；
4. trigger 的 alpha 与 hard 的 alpha 是否在同一 coordinate system 尚未确定；
5. 目前没有 alpha 历史分布；
6. 也没有 alpha 与实际 style strength 的标定曲线。

推荐流程：

```text
第一步：在 v3/v7 checkpoint 上离线计算 alpha/beta/omega 分布
第二步：按 condition、timestep、image bucket 统计 quantiles
第三步：用 quantile/EMA target，而非手写绝对常数
第四步：用软 band loss，且只对明显越界部分施压
```

还需要避免一个常见误解：

```text
alpha=0.5 并不普遍意味着“风格强度 50%”。
```

它只意味着在当前定义的 reference direction 上，LoRA residual 的投影达到约半个 base-to-target displacement；这不是直接可视化的 style percentage。

---

# 7. 对 alpha floor 与 squared hinge 的判断

用：

```text
ReLU(a - alpha)^2
```

代替：

```text
ReLU(a - alpha)
```

有合理性，但不能说无条件更好。

## 7.1 squared hinge 的优点

当 deficit 很大时：

```text
deficit = a - alpha
```

平方 hinge 的梯度大小为：

```text
2 * deficit
```

因此会对大 deficit 施加更大惩罚，不像线性 hinge 那样始终是常数斜率。

## 7.2 squared hinge 的限制

在当前 alpha 的实际单位下：

```text
如果 deficit < 0.5，2*deficit 反而小于线性 hinge 的 slope 1
```

所以它不是简单地“比线性更强”；它是：

```text
对大 deficit 更强
对小 deficit 更弱
```

此外，alpha 可以为负，或者受到极端 denominator 影响，平方项可能放大 outlier。

推荐：

```text
Huberized squared hinge
或
clamped deficit squared hinge
或
linear + small quadratic mixture
```

并记录：

```text
alpha floor violation mean
alpha floor violation p90
alpha floor satisfied fraction
```

---

# 8. 对 A1 改造建议的判断

这是外部提案中最可信、也最适合优先验证的部分。

当前 A1 的实际问题是：

```text
text-space residual cosine alignment 很强
但 diffusion target-space gain 很弱
```

因此只优化：

```text
Qwen tap residual direction consistency
```

并不能保证 activator effect 真正指向数据 target。

## 8.1 推荐的 A1 causal auxiliary term

对 structured/natural 各自计算：

```text
p0_s = P(F0, A0(C_s))
pA_s = P(F0, Aθ(C_s))
rA_s = pA_s - p0_s
v_s  = Y - p0_s
alphaA_s = <rA_s,v_s>/(||v_s||²+epsilon)
```

推荐加入：

```text
L_A1_alpha_floor = soft_hinge(a_A1 - alphaA_structured)
                  + soft_hinge(a_A1 - alphaA_natural)
```

以及：

```text
L_A1_effect_consistency = (alphaA_structured-alphaA_natural)^2
```

但不应删除原来的：

```text
L_diffusion_MSE
L_context_cosine
```

## 8.2 必须补充 residual energy control

因为 alpha 单独只要求 projection 变大，可能通过大规模 off-direction residual 实现。

应加入以下之一：

```text
L_off = omega_A
```

或：

```text
L_residual = beta_A
```

或保留旧 gain 作为 regularizer：

```text
L_gain = -G_A
```

推荐初始版本：

```text
L_A1 = L_diffusion
     + λ_gain * soft_hinge(f_gain - G)
     + λ_alpha * soft_hinge(a_alpha - alpha)
     + λ_off * omega
     + λ_context * L_context
     + λ_effect * (alpha_struct-alpha_nat)^2
```

并从小的 `λ_alpha`、`λ_off` 开始，而不是让新目标一开始压过 diffusion MSE。

---

# 9. 对 B 改造建议的判断

B 可以使用 conditional response hierarchy，但不建议一次性完全重写成只有 alpha 的目标。

推荐分层：

```text
L_style:
  trigger condition 对 target 的 MSE

L_preserve:
  far 高权重 residual preservation
  neutral 中等权重 preservation
  hard 低权重 preservation

L_rank:
  shared-reference alpha 的 soft hierarchy

L_off:
  orthogonal residual penalty

L_trigger_floor:
  shared-reference trigger alpha floor
```

推荐抽象形式：

```text
L_B = λ_style L_style
    + λ_pres(q) L_preserve(q)
    + λ_rank L_rank
    + λ_off L_off
    + λ_floor L_trigger_floor
```

## 9.1 far preservation

对 far：

```text
L_far_preserve = ||Pφ(c_far)-P0(c_far)||²
```

高权重是合理的，因为 far 条件应尽量不触发 learned style LoRA。

## 9.2 neutral preservation

neutral 不应被要求完全不动，否则最终行为会退化为严格 trigger gate。

可以使用中等 preservation 权重，并让 neutral alpha 只保持在小正区间。

## 9.3 hard preservation

如果 hard 的语义确实是 target style manifold 的邻近方向，那么 preservation 应低，不能继续像 v7 那样强迫它回到 base。

不过“低/零 preservation”不代表应该强迫 hard 的 target-MSE gain 变大到 trigger 级别。它只意味着允许部分响应。

---

# 10. off-direction penalty 是否必须

必须。

如果只优化：

```text
alpha_trigger 上升
```

LoRA 可能通过增大 `||r||` 获得更大 projection，同时造成结构破坏。

alpha/beta/omega 分解提供了自然的控制量：

```text
alpha：希望提高
omega：希望降低
beta：希望受控
```

推荐至少记录并监控：

```text
mean alpha
mean beta
mean omega
p90 omega
omega/alpha ratio
```

训练目标中可使用：

```text
L_off = mean clamp(omega, 0, omega_max)
```

或简单：

```text
L_off = mean omega
```

但需要注意 numeric scale，因为 `omega` 与 normalized prediction error 相关，不能未经 calibration 直接给 1.0 权重。

---

# 11. 容量建议评估

## 11.1 shared rank-1 internal adapter 确实值得怀疑

当前实际实现是：

```text
同一个 rank-1 MaskedLowRankAdapter
在每个 Qwen decoder layer 后使用
```

其单层线性 residual 的即时输出位于一个 rank-1 output subspace 内。

因此它的表达能力确实受到限制，尤其相对于：

```text
Qwen3-VL 8B
13 个最终 tap
53248 维 Ideogram conditioning
```

rank 1 可能过于保守。

## 11.2 “只有一个输出方向”需要加限定

严格说，整个 recurrent system 不一定只有一个最终方向，因为：

- `down` scalar 随 hidden state 变化；
- 后续 Qwen 非线性层会混合和旋转扰动；
- 不同层的输入状态不同；
- 相同 adapter 在不同 depth 的作用位置不同。

所以更准确的说法是：

```text
每个注入点的直接 residual 是 rank-1，且跨层共享参数；
这构成很强的容量和归纳偏置限制，但不等于整个网络最终只能输出一个固定方向。
```

## 11.3 per-layer module LoRA 值得做

将实际实现改成显式 adapter mode：

```yaml
te_adapter:
  mode: shared_post_layer
```

和：

```yaml
te_adapter:
  mode: module_lora
  target_modules: [down_proj]
  rank: 4
  per_layer: true
```

这是一个良好的实验设计，因为它可以把：

```text
实现错误/配置名不一致
```

转化成：

```text
可复现的 architecture ablation
```

还应保留：

```text
[o_proj]
[down_proj, o_proj]
```

作为后续实验，而不是直接假设 `down_proj` 一定正确。

## 11.4 rank 4 / rank 2 是合理候选，不是已证实答案

推荐候选：

```text
internal module LoRA rank 4
tap adapter rank 2
```

这个容量增加是合理的，但应配合：

- parameter count；
- parameter norm；
- alpha/beta/omega；
- held-out gain；
- generation quality；
- memory/step time。

不能只用训练 MSE 判断 rank 4 更好。

## 11.5 virtual tokens 的优先级低于 adapter ablation

4 virtual tokens 可能增加 trigger bandwidth，但它会同时改变：

- sequence length；
- position IDs；
- causal attention pattern；
- trigger mask 长度；
- tap token alignment；
- padding；
- artifact schema；
- ComfyUI prompt runtime；
- atomic token expansion 逻辑。

因此它不是一个“只改 embedding shape”的小改动。

建议顺序：

```text
先做 1-token + shared rank-1 baseline
再做 1-token + per-layer rank-4
再做 4-token + 最优 adapter
```

这样才能判断收益来自：

```text
token bandwidth
还是
adapter capacity
```

---

# 12. “停止当前 B”建议的判断

## 12.1 支持停止的理由

如果新的研究问题已经变成：

```text
graded conditional response
```

那么当前 B objective 确实不再是正确实验：

- hard 被当作 decoy；
- Path 2 试图保持 hard 接近 base；
- Path 3 把 hard 的 positive target gain 当作需要抑制的量；
- neutral/far/hard 的差别目前主要用于 sampling diagnostics，而不是不同数学 role。

在这种情况下，继续 1400 steps 不能干净地验证新假设。

## 12.2 不足以证明必须停止的理由

仅凭 step 575 还不能证明：

- 当前 B 后期绝不会分离；
- A1 对 B 完全没有帮助；
- 完整 v7 没有 baseline 价值。

尤其当前 B 的 schedule 到后期还会继续提高 Path 3 权重和 margin，理论上仍可能在 800–1500 steps 发生变化。

## 12.3 最终建议

我的实际建议是：

```text
在 step 600 保存 checkpoint 后停止，是合理的折中。
```

然后：

1. 保留 step 600 作为 v7 old-objective baseline；
2. 对固定 prompt/noise/timestep 补算 alpha/beta/omega；
3. 对 far/neutral/hard/trigger 做一次同条件 probe；
4. 运行少量 novel-scene generation；
5. 不把 step 600 的 alpha 排序当作最终证明；
6. 改造目标后，用短 A1 + short B probe 做新实验。

这比现在直接让 v7 B 跑满、然后用一个与研究问题不匹配的结果来决策更有效率。

---

# 13. 推荐的实验路线

## Stage 0：离线诊断，不改训练

对现有 v7 A1/B checkpoints 计算：

```text
alpha
beta
omega
old normalized gain
reconstructed gain
base loss
student loss
```

至少按以下维度拆分：

```text
condition class
caption source
fixed timestep bucket
item id
```

目标是确认：

```text
alpha 是否与 old gain、实际生成 style strength 相关
omega 是否在 trigger branch 更大
不同 category 的 alpha 是否真的形成稳定次序
```

## Stage 1：A1 objective augmentation

保留现有：

```text
plain diffusion MSE
context cosine
```

新增：

```text
alpha floor
structured/natural alpha consistency
small off-direction penalty
magnitude consistency
```

但暂时不引入 virtual tokens。

## Stage 2：adapter capacity ablation

至少做：

```text
A: shared post-layer rank 1
B: per-layer down_proj rank 4
C: per-layer down_proj rank 4 + tap rank 2
```

每个只跑短 A1，使用固定 probes。

## Stage 3：short B learnability probe

每个 A1 candidate 用相同：

- initial diffusion LoRA；
- seed；
- batch order；
- fixed probe；
- B steps，例如 100–300。

比较：

```text
trigger target loss
shared-reference alpha_trigger
alpha_hard
alpha_neutral
alpha_far
omega
```

只有出现稳定 improvement 才继续完整 B。

## Stage 4：virtual-token ablation

在 adapter 最优候选上比较：

```text
1 virtual token
2 virtual tokens
4 virtual tokens
```

此阶段必须同时更新：

- tokenizer/runtime expansion；
- trigger mask；
- tap alignment；
- artifact metadata；
- ComfyUI loader/encoder；
- validation probes。

---

# 14. 推荐的修正版 objective

## 14.1 A1

```text
L_A1 = L_diffusion
     + λ_gain L_gain_floor
     + λ_alpha L_alpha_floor
     + λ_off L_offdirection
     + λ_context L_context_cosine
     + λ_mag L_context_magnitude
     + λ_effect (alpha_struct-alpha_nat)^2
```

其中：

```text
L_gain_floor  = soft_hinge(f_gain - G)
L_alpha_floor = soft_hinge(a_alpha - alpha)
L_offdirection = omega 或受控 beta
```

## 14.2 B

使用共同 reference direction `v_ref`：

```text
alpha_q = <r_q,v_ref>/(||v_ref||²+epsilon)
```

然后：

```text
L_B = λ_style L_trigger_target
    + λ_rank [
        soft_hinge(m_tau_h - (E alpha_tau - E alpha_hard))
      + soft_hinge(m_h_n - (E alpha_hard - E alpha_neutral))
      + soft_hinge(m_n_f - (E alpha_neutral - E alpha_far))
      ]
    + λ_pres,far L_preserve_far
    + λ_pres,neutral L_preserve_neutral
    + λ_pres,hard L_preserve_hard
    + λ_off E[omega_q]
    + λ_floor soft_hinge(a_tau - E alpha_tau)
```

不建议第一版使用硬 bands；先使用 soft hierarchy 与 trigger floor。

## 14.3 语义角色

```text
far:
  high preservation, low/zero desired response

neutral:
  medium preservation, small positive response allowed

hard:
  low preservation, intermediate response allowed

trigger:
  target acquisition + largest response floor
```

---

# 15. 最终判断

外部 agent 的提案不是简单的“换一个 loss 就会更好”，而是提出了一个更有解释力的研究框架：

```text
从 binary trigger selectivity
转向 conditional response shaping
```

这个框架在语义上比当前 v7 B 更合理。

但是必须避免三个数学错误：

```text
错误 1：把 alpha 单独当成纯 style causality
错误 2：把不同 condition-specific alpha 直接当作同一尺度比较
错误 3：删除旧 gain 的 beta/off-direction control
```

最稳妥的方案是：

```text
alpha / beta / omega 先作为诊断分解；
A1 先加入小权重 causal auxiliary objective；
B 使用共同 reference direction 的 soft hierarchy；
保留 trigger target reconstruction、far preservation 与 off-direction penalty；
adapter capacity 通过显式 mode 做 ablation；
virtual tokens 放在 adapter ablation 之后。
```

因此我的最终回答是：

```text
是的，新 proposed objective 的研究方向比当前 setup 更有意义；
不是的，原提案的 alpha hierarchy 公式还不能直接作为最终训练目标；
是的，停止 v7 B 并转向一个短 A1 + short B probe 是合理的；
不是的，不能仅凭 A1 的 alpha 或 rank 4 的训练 gain 就宣称新方法成功。
```
