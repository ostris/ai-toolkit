"""Small fused Triton kernels for the Qwen3.8 decode path. Inference only: every op
falls back to the eager formulation under autograd or off the GPU."""



import torch

_kernels = None


def _get():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def rmsnorm_kernel(x_ptr, w_ptr, o_ptr, D, eps, ADD_ONE: tl.constexpr, BLOCK: tl.constexpr):
        # one row per program: fp32 math, out = x * rsqrt(mean(x^2) + eps) * ((1 +) w)
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if ADD_ONE:
            w = w + 1.0
        y = x * (1.0 / tl.sqrt(var + eps)) * w
        tl.store(o_ptr + row * D + offs, y.to(o_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def decode_attn_kernel(
        q_ptr, k_ptr, v_ptr, qpos_ptr, o_ptr,
        L, scale, min_key,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kl,
        stride_ob, stride_oh, stride_om,
        G: tl.constexpr, M: tl.constexpr, MPAD: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr, BLOCK_L: tl.constexpr,
    ):
        # one program per (batch, kv head): the G query heads of the group x M query
        # tokens (rows padded to ROWS >= 16 for the MMA) attend the cache up to the
        # largest query position, read from a device tensor so a CUDA graph never bakes a
        # length in. Online softmax in fp32; scores in fp32 straight from the bf16 dot.
        b = tl.program_id(0)
        hk = tl.program_id(1)
        rows = tl.arange(0, ROWS)
        rmask = rows < G * M
        g_idx = tl.where(rmask, rows // M, 0)
        m_idx = tl.where(rmask, rows % M, 0)
        offs_d = tl.arange(0, D)
        qpos = tl.load(qpos_ptr + m_idx)
        offs_mp = tl.arange(0, MPAD)
        max_pos = tl.max(tl.load(qpos_ptr + offs_mp, mask=offs_mp < M, other=0), axis=0)
        q = tl.load(
            q_ptr + b * stride_qb + (hk * G + g_idx)[:, None] * stride_qh + m_idx[:, None] * stride_qm + offs_d[None, :],
            mask=rmask[:, None], other=0.0,
        )  # [ROWS, D] bf16
        m_i = tl.full((ROWS,), float("-inf"), tl.float32)
        l_i = tl.zeros((ROWS,), tl.float32)
        acc = tl.zeros((ROWS, D), tl.float32)
        n_blocks = (max_pos + BLOCK_L) // BLOCK_L
        for blk in range(0, n_blocks):
            offs_l = blk * BLOCK_L + tl.arange(0, BLOCK_L)
            kmask = offs_l < L
            k = tl.load(k_ptr + b * stride_kb + hk * stride_kh + offs_l[:, None] * stride_kl + offs_d[None, :],
                        mask=kmask[:, None], other=0.0)  # [BLOCK_L, D]
            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale  # [ROWS, BLOCK_L]
            allowed = (offs_l[None, :] <= qpos[:, None]) & (offs_l[None, :] >= min_key) & kmask[None, :]
            s = tl.where(allowed, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + b * stride_kb + hk * stride_kh + offs_l[:, None] * stride_kl + offs_d[None, :],
                        mask=kmask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            m_i = m_new
        out = acc / l_i[:, None]
        tl.store(o_ptr + b * stride_ob + (hk * G + g_idx)[:, None] * stride_oh + m_idx[:, None] * stride_om + offs_d[None, :],
                 out.to(o_ptr.dtype.element_ty), mask=rmask[:, None])

    @triton.jit
    def add_rmsnorm_kernel(h_ptr, x_ptr, w_ptr, hout_ptr, y_ptr, D, eps, ADD_ONE: tl.constexpr, BLOCK: tl.constexpr):
        # h_new = h + x (bf16, written back) and y = rmsnorm(h_new) in one pass
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        h = tl.load(h_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
        hn = (h + x).to(hout_ptr.dtype.element_ty)
        tl.store(hout_ptr + row * D + offs, hn, mask=mask)
        hf = hn.to(tl.float32)
        var = tl.sum(hf * hf, axis=0) / D
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if ADD_ONE:
            w = w + 1.0
        tl.store(y_ptr + row * D + offs, (hf * (1.0 / tl.sqrt(var + eps)) * w).to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def swiglu_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + offs, (g * tl.sigmoid(g) * u).to(o_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _conv_in(state_ptr, x_ptr, b, offs_c, mask, C, i: tl.constexpr, M: tl.constexpr, KS: tl.constexpr):
        # input i of the window [state (KS-1 past inputs) | M new inputs] for channels offs_c
        if i < KS - 1:
            return tl.load(state_ptr + b * C * (KS - 1) + offs_c * (KS - 1) + i, mask=mask, other=0.0).to(tl.float32)
        else:
            return tl.load(x_ptr + (b * M + (i - (KS - 1))) * C + offs_c, mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def conv_step_kernel(state_ptr, x_ptr, w_ptr, o_ptr, C, M: tl.constexpr, KS: tl.constexpr, WRITE_STATE: tl.constexpr, BLOCK_C: tl.constexpr):
        # depthwise causal conv (KS taps) over [state | M new inputs] + silu, x and out token
        # major [B, M, C] (as the projection produced them), state [B, C, KS-1].
        # Optionally slides the state window forward by M.
        b = tl.program_id(0)
        offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offs_c < C
        for m in tl.static_range(M):
            acc = tl.zeros((BLOCK_C,), tl.float32)
            for j in tl.static_range(KS):
                w = tl.load(w_ptr + offs_c * KS + j, mask=mask, other=0.0).to(tl.float32)
                acc += w * _conv_in(state_ptr, x_ptr, b, offs_c, mask, C, m + j, M, KS)
            tl.store(o_ptr + (b * M + m) * C + offs_c, (acc * tl.sigmoid(acc)).to(o_ptr.dtype.element_ty), mask=mask)
        if WRITE_STATE:
            for j in tl.static_range(KS - 1):
                v = _conv_in(state_ptr, x_ptr, b, offs_c, mask, C, M + j, M, KS)
                tl.store(state_ptr + b * C * (KS - 1) + offs_c * (KS - 1) + j, v.to(state_ptr.dtype.element_ty), mask=mask)

    _kernels = (rmsnorm_kernel, decode_attn_kernel, add_rmsnorm_kernel, swiglu_kernel, conv_step_kernel)
    return _kernels


@torch.library.custom_op("ostris::qwen38_rmsnorm", mutates_args=())
def _rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float, add_one: bool) -> torch.Tensor:
    shape = x.shape
    d = shape[-1]
    x2 = x.reshape(-1, d)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = torch.empty_like(x2)
    kernel = _get()[0]
    block = 1 << (d - 1).bit_length()
    kernel[(x2.shape[0],)](x2, weight, out, d, eps, ADD_ONE=add_one, BLOCK=block, num_warps=8 if block >= 2048 else 2)
    return out.view(shape)


@_rmsnorm_op.register_fake
def _rmsnorm_fake(x, weight, eps, add_one):
    return torch.empty_like(x)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, add_one: bool = True) -> torch.Tensor:
    """Fused RMSNorm when no gradient is needed; eager fp32 math otherwise."""
    if x.is_cuda and not torch.is_grad_enabled() and x.shape[-1] <= 16384:
        return _rmsnorm_op(x, weight, eps, add_one)
    xf = x.float()
    w = weight.float()
    if add_one:
        w = 1.0 + w
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).type_as(x)


@torch.library.custom_op("ostris::qwen38_decode_attn", mutates_args=())
def _decode_attn_op(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, qpos: torch.Tensor, scale: float, min_key: int) -> torch.Tensor:
    """q [B, Hq, M, D] (M power of two <= 16), caches [B, Hk, L, D], qpos [M] int64.
    Reads the cache only up to the largest query position."""
    b, hq, m, d = q.shape
    hk, L = k_cache.shape[1], k_cache.shape[2]
    g = hq // hk
    q = q.contiguous()
    out = torch.empty_like(q)
    kernel = _get()[1]
    rows = max(16, 1 << (g * m - 1).bit_length())
    kernel[(b, hk)](
        q, k_cache, v_cache, qpos, out,
        L, scale, min_key,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        G=g, M=m, MPAD=1 << max(0, (m - 1).bit_length()), ROWS=rows, D=d, BLOCK_L=32, num_warps=4, num_stages=2,
    )
    return out


@_decode_attn_op.register_fake
def _decode_attn_fake(q, k_cache, v_cache, qpos, scale, min_key):
    return torch.empty_like(q)


# off by default: one program per kv head leaves most SMs idle, so past ~500 keys the
# cuBLAS path (full-cache masked bmm) is faster; a split-key version would fix that.
DECODE_ATTN_FUSED = False


def decode_attention_fused_ok(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    b, hq, m, d = q.shape
    hk = k_cache.shape[1]
    return DECODE_ATTN_FUSED and q.is_cuda and m <= 16 and (hq // hk) * m <= 128 and d in (64, 128, 256) and not torch.is_grad_enabled()


@torch.library.custom_op("ostris::qwen38_add_rmsnorm", mutates_args=())
def _add_rmsnorm_op(h: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, add_one: bool) -> list[torch.Tensor]:
    shape = h.shape
    d = shape[-1]
    h2 = h.reshape(-1, d).contiguous()
    x2 = x.reshape(-1, d).contiguous()
    hout = torch.empty_like(h2)
    y = torch.empty_like(h2)
    kernel = _get()[2]
    block = 1 << (d - 1).bit_length()
    kernel[(h2.shape[0],)](h2, x2, weight, hout, y, d, eps, ADD_ONE=add_one, BLOCK=block, num_warps=8 if block >= 2048 else 2)
    return [hout.view(shape), y.view(shape)]


@_add_rmsnorm_op.register_fake
def _add_rmsnorm_fake(h, x, weight, eps, add_one):
    return [torch.empty_like(h), torch.empty_like(h)]


def add_rmsnorm(h: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, add_one: bool = True):
    """(h + x, rmsnorm(h + x)): fused on the no-grad GPU path."""
    if h.is_cuda and not torch.is_grad_enabled() and h.shape[-1] <= 16384 and h.dtype == x.dtype:
        hn, y = _add_rmsnorm_op(h, x, weight, eps, add_one)
        return hn, y
    hn = h + x
    return hn, rmsnorm(hn, weight, eps, add_one)


@torch.library.custom_op("ostris::qwen38_swiglu", mutates_args=())
def _swiglu_op(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    g2 = g.contiguous()
    u2 = u.contiguous()
    out = torch.empty_like(g2)
    n = g2.numel()
    kernel = _get()[3]
    kernel[(-(-n // 4096),)](g2, u2, out, n, BLOCK=4096, num_warps=4)
    return out


@_swiglu_op.register_fake
def _swiglu_fake(g, u):
    return torch.empty_like(g)


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    if g.is_cuda and not torch.is_grad_enabled() and g.dtype == u.dtype:
        return _swiglu_op(g, u)
    return torch.nn.functional.silu(g) * u


@torch.library.custom_op("ostris::qwen38_conv_step", mutates_args={"state"})
def _conv_step_op(state: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, write_state: bool) -> torch.Tensor:
    """Depthwise causal conv + silu of M new tokens x [B, M, C] against state [B, C, KS-1]
    (conv weight [C, 1, KS]); returns [B, M, C]. write_state slides the window forward."""
    b, m, c = x.shape
    ks = weight.shape[-1]
    x2 = x.contiguous()
    out = torch.empty_like(x2)
    kernel = _get()[4]
    block_c = 1024
    kernel[(b, -(-c // block_c))](state, x2, weight.reshape(c, ks).contiguous(), out, c, M=m, KS=ks, WRITE_STATE=write_state, BLOCK_C=block_c, num_warps=4)
    return out


@_conv_step_op.register_fake
def _conv_step_fake(state, x, weight, write_state):
    return torch.empty_like(x)
