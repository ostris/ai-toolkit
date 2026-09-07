"""Vendored from FastVideo (https://github.com/hao-ai-lab/FastVideo),
``fastvideo-kernel/python/fastvideo_kernel/triton_kernels/block_sparse_attn_triton.py``.
Apache-2.0; Copyright the FastVideo team. Vendored unmodified so the
MiniMax-H3 VSA fine stage can run FastVideo's trained-policy Triton
block-sparse kernels (fwd + bwd) without the fastvideo-kernel package."""

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team
"""

import torch
import triton
import triton.language as tl

# ──────────────────────────── SPARSE ADDITION BEGIN ───────────────────────────
import math  # small utility needed by the sparse wrapper
# ──────────────────────────── SPARSE ADDITION END ─────────────────────────────

# BLOCK_M / BLOCK_N are fixed at 64 because they are structural, not tunable:
# the kernel indexes the top-k list per BLOCK_M q-tile and addresses keys as
# kv_idx * BLOCK_N, so both must match the granularity q2k_index and
# variable_block_sizes were built at.
#
# num_stages / num_warps ARE free, and the previous {3, 4, 7} was inherited from
# the upstream tutorial rather than tuned here. It skips 5 and 6; on Blackwell
# (sm_121) the optimum is num_stages=5, so the search could not reach it. Both
# block paths independently select 5 once it is available. Autotune still picks
# per architecture, so other GPUs re-tune rather than inheriting this choice.
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64]
    for BN in [64]
    for s in [2, 3, 4, 5, 6, 7]
    for w in [4, 8]
]


# ──────────────────────────── SPARSE ADDITION BEGIN ───────────────────────────
@triton.autotune(configs, key=["N_CTX_Q", "HEAD_DIM"])
@triton.jit
def _attn_fwd_sparse(
    Q,
    K,
    V,
    sm_scale,  #
    q2k_index,
    q2k_num,
    max_kv_blks,  #
    variable_block_sizes,
    M,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX_Q,  #
    N_CTX_KV,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    64×64 **block-sparse** forward kernel. Back-prop kernels remain dense
    (32×64 and 64×32) – memory footprint unchanged.
    """

    # ----- program-id mapping -----
    q_blk = tl.program_id(0)  # Q-tile index
    off_hz = tl.program_id(1)  # fused (batch, head)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX_Q // BLOCK_M
    meta_base = (b * H + h) * q_tiles + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)  # int32
    kv_ptr = q2k_index + meta_base * max_kv_blks  # ptr to list

    # ----- base pointers -----
    # Note: when q and kv have different sequence lengths, their per-(batch,head)
    # strides differ, so we must compute separate base offsets.
    q_off = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    k_off = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    v_off = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    o_off = b.to(tl.int64) * stride_oz + h.to(tl.int64) * stride_oh

    Q_ptr = tl.make_block_ptr(
        base=Q + q_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    K_base = tl.make_block_ptr(
        base=K + k_off,
        shape=(HEAD_DIM, N_CTX_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )

    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_base = tl.make_block_ptr(
        base=V + v_off,
        shape=(N_CTX_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )

    O_ptr = tl.make_block_ptr(
        base=Out + o_off,
        shape=(N_CTX_Q, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # ----- accumulators -----
    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln2
    q = tl.load(Q_ptr)

    # ----- sparse loop over valid K/V tiles -----
    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        K_ptr = tl.advance(K_base, (0, kv_idx * BLOCK_N))
        V_ptr = tl.advance(V_base, (kv_idx * BLOCK_N, 0))

        k = tl.load(K_ptr)
        qk = tl.dot(q, k)
        # mask out invalid columns
        mask = tl.arange(0, BLOCK_N) < block_size
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_ptr)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # ----- epilogue -----
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX_Q + offs_m, m_i)
    tl.store(O_ptr, acc.to(Out.type.element_ty))


# ──────────────────────────── SPARSE ADDITION END ─────────────────────────────


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,  #
    Delta,  #
    Z,
    H,
    N_CTX,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    k2q_index,
    k2q_num,
    max_q_blks,
    variable_block_sizes,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX_KV,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    step_m = BLOCK_M1
    kv_blk = tl.program_id(0)  # Q-tile index
    off_hz = tl.program_id(2)  # fused (batch, head)
    b = off_hz // H
    h = off_hz % H
    kv_tiles = N_CTX_KV // BLOCK_N1
    meta_base = (b * H + h) * kv_tiles + kv_blk

    q_blocks = tl.load(k2q_num + meta_base)  # int32
    q_ptr = k2q_index + meta_base * max_q_blks  # ptr to list
    block_size = tl.load(variable_block_sizes + kv_blk)

    for blk_idx in range(q_blocks * 2):
        block_sparse_offset = (
            tl.load(q_ptr + blk_idx // 2).to(tl.int32) * 2 + blk_idx % 2
        ) * step_m
        qT = tl.load(qT_ptrs + block_sparse_offset * stride_tok)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = start_m + block_sparse_offset + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        # Recompute logits exactly as the forward does: raw bf16 operands into
        # the dot, fp32 scale after accumulation. A bf16 pre-scaled K perturbs
        # the recomputed logits relative to the saved M by an error
        # proportional to |logit|, which exp2 amplifies into arbitrarily wrong
        # probabilities at large activations.
        qkT = tl.dot(k, qT) * (sm_scale * 1.4426950408889634)
        pT = tl.math.exp2(qkT - m[None, :])
        mask = tl.arange(0, BLOCK_N1) < block_size
        pT = tl.where(mask[:, None], pT, 0.0)

        do = tl.load(do_ptrs + block_sparse_offset * stride_tok)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.bfloat16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    D,
    sm_scale,
    # shared by Q/K/V/DO.
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    step_n = BLOCK_N2

    q_blk = tl.program_id(0)  # Q-tile index
    off_hz = tl.program_id(2)  # fused (batch, head)
    b = off_hz // H
    h = off_hz % H
    q_tiles = N_CTX // BLOCK_M2
    meta_base = (b * H + h) * q_tiles + q_blk

    kv_blocks = tl.load(q2k_num + meta_base)  # int32
    kv_ptr = q2k_index + meta_base * max_kv_blks  # ptr to list

    for blk_idx in range(kv_blocks * 2):
        kv_idx = tl.load(kv_ptr + blk_idx // 2).to(tl.int32)
        # variable_block_sizes is defined per KV block (tile). Mask must therefore
        # use kv_idx (not q_blk). Also, because we split each 64-token block into
        # two 32-token halves, the mask must account for the half-block offset.
        block_size = tl.load(variable_block_sizes + kv_idx).to(tl.int32)
        half = (blk_idx % 2).to(tl.int32)
        block_sparse_offset = (kv_idx * 2 + half) * step_n * stride_tok
        kT = tl.load(kT_ptrs + block_sparse_offset)
        vT = tl.load(vT_ptrs + block_sparse_offset)
        qk = tl.dot(q, kT) * (sm_scale * 1.4426950408889634)
        p = tl.math.exp2(qk - m)
        offs_in_block = half * step_n + tl.arange(0, BLOCK_N2)
        mask = offs_in_block < block_size
        p = tl.where(mask[None, :], p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        # Compute dQ (kT is raw; the caller applies sm_scale once at the end).
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
    return dq


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    q2k_index,
    q2k_num,
    max_kv_blks,
    k2q_index,
    k2q_num,
    max_q_blks,
    variable_block_sizes,
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
):
    LN2 = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = N_CTX // BLOCK_M1

    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        k2q_index,
        k2q_num,
        max_q_blks,
        variable_block_sizes,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = 0

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    num_steps = N_CTX // BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        sm_scale,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n,
        num_steps,  #
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= sm_scale
    tl.store(dq_ptrs, dq)


@triton.jit
def _attn_bwd_dkdv_kernel(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DK,
    DV,  #
    M,
    D,
    k2q_index,
    k2q_num,
    max_q_blks,
    variable_block_sizes,
    # shared token/dim strides (assumed contiguous along token and dim)
    stride_tok,
    stride_d,  #
    # batch/head strides (may differ between Q and KV)
    stride_qz,
    stride_qh,
    stride_kz,
    stride_kh,
    stride_vz,
    stride_vh,
    stride_doz,
    stride_doh,
    stride_dkz,
    stride_dkh,
    stride_dvz,
    stride_dvh,
    H,
    N_CTX_Q,
    N_CTX_KV,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
):
    """
    Backward kernel that computes dK and dV for each KV block (64 tokens).
    Grid:
      pid0: kv_blk in [0, N_CTX_KV/BLOCK_N1)
      pid2: fused (batch, head) in [0, B*H)
    """
    bhid = tl.program_id(2)
    b = bhid // H
    h = bhid % H
    kv_blk = tl.program_id(0)

    q_adj = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    kv_adj_k = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    kv_adj_v = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    do_adj = b.to(tl.int64) * stride_doz + h.to(tl.int64) * stride_doh
    dk_adj = b.to(tl.int64) * stride_dkz + h.to(tl.int64) * stride_dkh
    dv_adj = b.to(tl.int64) * stride_dvz + h.to(tl.int64) * stride_dvh

    Q = Q + q_adj
    K = K + kv_adj_k
    V = V + kv_adj_v
    DO = DO + do_adj
    DK = DK + dk_adj
    DV = DV + dv_adj

    # M and D (delta) are always sized by Q length.
    M = M + (bhid * N_CTX_Q).to(tl.int64)
    D = D + (bhid * N_CTX_Q).to(tl.int64)

    offs_k = tl.arange(0, HEAD_DIM)
    start_n = kv_blk * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    dv_acc = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk_acc = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    num_steps = N_CTX_Q // BLOCK_M1
    dk_acc, dv_acc = _attn_bwd_dkdv(
        dk_acc,
        dv_acc,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        k2q_index,
        k2q_num,
        max_q_blks,
        variable_block_sizes,
        stride_tok,
        stride_d,
        H,
        N_CTX_KV,
        BLOCK_M1=BLOCK_M1,
        BLOCK_N1=BLOCK_N1,
        HEAD_DIM=HEAD_DIM,
        start_n=start_n,
        start_m=0,
        num_steps=num_steps,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv_acc)

    dk_acc *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk_acc)


@triton.jit
def _attn_bwd_dq_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,  #
    DQ,
    M,
    D,
    q2k_index,
    q2k_num,
    max_kv_blks,
    variable_block_sizes,
    # shared token/dim strides (assumed contiguous along token and dim)
    stride_tok,
    stride_d,  #
    # batch/head strides (may differ between Q and KV)
    stride_qz,
    stride_qh,
    stride_kz,
    stride_kh,
    stride_vz,
    stride_vh,
    stride_doz,
    stride_doh,
    stride_dqz,
    stride_dqh,
    H,
    N_CTX_Q,
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
):
    """
    Backward kernel that computes dQ for each Q block (64 tokens).
    Grid:
      pid0: q_blk in [0, N_CTX_Q/BLOCK_M2)
      pid2: fused (batch, head) in [0, B*H)
    """
    LN2 = 0.6931471824645996  # = ln(2)
    bhid = tl.program_id(2)
    b = bhid // H
    h = bhid % H
    q_blk = tl.program_id(0)

    q_adj = b.to(tl.int64) * stride_qz + h.to(tl.int64) * stride_qh
    kv_adj_k = b.to(tl.int64) * stride_kz + h.to(tl.int64) * stride_kh
    kv_adj_v = b.to(tl.int64) * stride_vz + h.to(tl.int64) * stride_vh
    do_adj = b.to(tl.int64) * stride_doz + h.to(tl.int64) * stride_doh
    dq_adj = b.to(tl.int64) * stride_dqz + h.to(tl.int64) * stride_dqh

    Q = Q + q_adj
    K = K + kv_adj_k
    V = V + kv_adj_v
    DO = DO + do_adj
    DQ = DQ + dq_adj

    M = M + (bhid * N_CTX_Q).to(tl.int64)
    D = D + (bhid * N_CTX_Q).to(tl.int64)

    offs_k = tl.arange(0, HEAD_DIM)
    start_m = q_blk * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)[:, None]

    dq_acc = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    num_steps = 0  # unused in _attn_bwd_dq
    dq_acc = _attn_bwd_dq(
        dq_acc,
        q,
        K,
        V,
        do,
        m,
        D,
        sm_scale,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        stride_tok,
        stride_d,
        H,
        N_CTX_Q,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        HEAD_DIM=HEAD_DIM,
        start_m=start_m,
        start_n=0,
        num_steps=num_steps,
    )

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq_acc *= sm_scale
    tl.store(dq_ptrs, dq_acc)


# ──────────────────────────── SPARSE ADDITION BEGIN ───────────────────────────
def triton_block_sparse_attn_forward(q, k, v, q2k_index, q2k_num, variable_block_sizes):
    B, H, Tq, D = q.shape
    Tkv = k.shape[2]
    sm_scale = 1.0 / math.sqrt(D)
    max_kv_blks = q2k_index.shape[-1]
    assert Tq % 64 == 0, f"q length must be a multiple of 64, but got {Tq}"
    assert Tkv % 64 == 0, f"kv length must be a multiple of 64, but got {Tkv}"
    assert q2k_num.shape[-1] == Tq // 64, (
        f"shape mismatch, Tq // 64 = {Tq // 64}, q2k_num.shape[-2] = {q2k_num.shape[-2]}"
    )
    assert variable_block_sizes.numel() == Tkv // 64, (
        f"shape mismatch, variable_block_sizes must have length {Tkv // 64}, "
        f"got {variable_block_sizes.numel()}"
    )
    o = torch.empty_like(q)
    M = torch.empty((B, H, Tq), dtype=torch.float32, device=q.device)

    grid = lambda _: (triton.cdiv(Tq, 64), B * H, 1)
    _attn_fwd_sparse[grid](
        q,
        k,
        v,
        sm_scale,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        M,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        B,
        H,
        Tq,
        Tkv,
        HEAD_DIM=D,
        STAGE=3,
    )

    return o, M


def triton_block_sparse_attn_backward(
    do, q, k, v, o, M, q2k_index, q2k_num, k2q_index, k2q_num, variable_block_sizes
):
    assert do.is_contiguous()

    B, H, Tq, D = q.shape
    Tkv = k.shape[2]
    sm_scale = 1.0 / math.sqrt(D)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_HEAD = q.shape[:2]
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
    # K stays raw: the backward kernels apply sm_scale in fp32 after the dot,
    # matching the forward's rounding exactly. (A bf16 pre-scaled K perturbs
    # the recomputed logits vs the saved M; exp2 turns that into unboundedly
    # wrong probabilities at large activations.)
    arg_k = k
    PRE_BLOCK = 64
    assert Tq % PRE_BLOCK == 0
    pre_grid = (Tq // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(M)
    _attn_bwd_preprocess[pre_grid](
        o,
        do,  #
        delta,  #
        BATCH,
        N_HEAD,
        Tq,  #
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=D,  #
    )

    max_q_blks = k2q_index.shape[-1]
    max_kv_blks = q2k_index.shape[-1]

    # dK/dV kernel: grid over KV blocks
    grid_kv = (Tkv // BLOCK_N1, 1, BATCH * N_HEAD)
    _attn_bwd_dkdv_kernel[grid_kv](
        q,
        arg_k,
        v,
        sm_scale,
        do,
        dk,
        dv,
        M,
        delta,
        k2q_index,
        k2q_num,
        max_q_blks,
        variable_block_sizes,
        q.stride(2),
        q.stride(3),
        q.stride(0),
        q.stride(1),
        arg_k.stride(0),
        arg_k.stride(1),
        v.stride(0),
        v.stride(1),
        do.stride(0),
        do.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        N_HEAD,
        Tq,
        Tkv,
        BLOCK_M1=BLOCK_M1,
        BLOCK_N1=BLOCK_N1,
        HEAD_DIM=D,
    )

    # dQ kernel: grid over Q blocks
    grid_q = (Tq // BLOCK_M2, 1, BATCH * N_HEAD)
    _attn_bwd_dq_kernel[grid_q](
        q,
        arg_k,
        v,
        sm_scale,
        do,
        dq,
        M,
        delta,
        q2k_index,
        q2k_num,
        max_kv_blks,
        variable_block_sizes,
        q.stride(2),
        q.stride(3),
        q.stride(0),
        q.stride(1),
        arg_k.stride(0),
        arg_k.stride(1),
        v.stride(0),
        v.stride(1),
        do.stride(0),
        do.stride(1),
        dq.stride(0),
        dq.stride(1),
        N_HEAD,
        Tq,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        HEAD_DIM=D,
    )

    return dq, dk, dv
