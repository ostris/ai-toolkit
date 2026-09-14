"""Sampling for YuE2: AR codec-token generation, then NAR flow matching.

Follows the released protocol: semantic sampling with temperature 1.0,
top-p 0.95, top-k 100, repetition penalty 1.2 over the last 50 tokens, at
least 200 tokens before MUSIC_END; acoustic synthesis with a midpoint ODE
solver from t=1 (noise) to t=0, one AR prefill per context chunk.
"""

from typing import Callable, List, Optional, Tuple

import torch

from .model import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    EOD,
    KVCache,
    MUSIC_END,
    YuE2Model,
    rope_cos_sin,
)
from .vae import HOP, YuE2VAE


def chunk_ranges(frames: int, prefix_tokens: int, context: int = CONTEXT) -> List[Tuple[int, int]]:
    size = (context - prefix_tokens - 3) // 2
    if frames < 1 or size < 1:
        raise ValueError("YuE2 needs codec tokens and enough context for at least one acoustic frame.")
    return [(start, min(start + size, frames)) for start in range(0, frames, size)]


def sample_logits(
    logits: torch.Tensor,
    history: List[int],
    step: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    penalty_window: int,
    min_tokens: int,
    generator: torch.Generator,
    phase: str = "semantic",
    legacy_off: bool = False,
) -> int:
    """ComfyUI's ``distribution`` for one row: ``abc`` allows the text vocabulary + ABC_END,
    ``semantic`` the codec ids + MUSIC_END. ``legacy_off`` = cot "off" (bf16 scores, keep top 3)."""
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, -torch.inf)
    if phase == "abc":
        allowed[:EOD] = 0
    else:
        allowed[CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[end] = 0
    scores += allowed
    if step < min_tokens:
        scores[end] = -torch.inf
    if repetition_penalty != 1.0 and history:
        recent = torch.tensor(history[-penalty_window:], dtype=torch.long, device=scores.device)
        counts = torch.zeros_like(scores).scatter_add_(0, recent, torch.ones_like(recent, dtype=scores.dtype))
        penalty = repetition_penalty ** counts
        scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    if temperature == 0:
        return int(scores.argmax().item())
    scores /= temperature
    threshold = scores.topk(min(top_k, scores.shape[-1])).values[-1]
    scores.masked_fill_(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        probs = values.softmax(-1)
        removed = probs.cumsum(-1) - probs > top_p
        removed[: 3 if legacy_off else 1] = False
        values.masked_fill_(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    probs = scores.softmax(-1)
    return int(torch.multinomial(probs.to(generator.device), 1, generator=generator).item())


class ARDecodeGraph:
    """One-token AR decode step captured as a CUDA graph over static buffers:
    token id in, logits out, preallocated [1, kv_heads, capacity, head_dim]
    cache per layer. Attention is hand-grouped (GQA) with a length mask so the
    whole step is capturable; the eager KVCache path stays as the fallback."""

    def __init__(self, ar, capacity: int, device, dtype):
        cfg = ar.cfg
        self.ar = ar
        self.capacity = capacity
        self.kv_heads = cfg.num_key_value_heads
        self.groups = cfg.num_attention_heads // cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        shape = (1, self.kv_heads, capacity, self.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.token = torch.zeros((1, 1), device=device, dtype=torch.long)
        self.pos = torch.zeros((1, 1), device=device, dtype=torch.long)
        self.index = torch.arange(capacity, device=device)
        # the graph covers the layer stack only; embedding lookup and lm_head run outside it
        # (the int8 repack's embedding/lm_head kernels invalidate stream capture)
        self.x_in = torch.zeros((1, 1, cfg.hidden_size), device=device, dtype=dtype)
        self.h_out = torch.zeros((1, cfg.hidden_size), device=device, dtype=dtype)
        self.logits = torch.empty((1, cfg.vocab_size), device=device, dtype=dtype)
        self.graph = None

    def load_prefill(self, cache, length: int):
        for i, (k, v) in enumerate(cache):
            self.k[i][:, :, :length] = k
            self.v[i][:, :, :length] = v
        self.pos.fill_(length)

    def _layers(self):
        ar = self.ar
        x = self.x_in
        cos, sin = rope_cos_sin(self.pos, self.head_dim, ar.cfg.rope_theta)
        mask = (self.index <= self.pos)[None, None]  # [1, 1, 1, capacity]
        scale = self.head_dim ** -0.5
        slot = self.pos.view(-1)
        for i, layer in enumerate(ar.model.layers):
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), cos, sin)
            self.k[i].index_copy_(2, slot, k)
            self.v[i].index_copy_(2, slot, v)
            qg = q.view(1, self.kv_heads, self.groups, self.head_dim)
            scores = torch.matmul(qg, self.k[i].transpose(-1, -2)) * scale  # [1, kv, groups, capacity]
            scores = scores.float().masked_fill(~mask, float("-inf")).softmax(-1).to(q.dtype)
            out = torch.matmul(scores, self.v[i]).reshape(1, 1, -1)
            x = x + layer.self_attn.o_proj(out)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        self.h_out.copy_(ar.model.norm(x[:, -1]))

    def _step(self):
        ar = self.ar
        self.x_in.copy_(ar.model.embed_tokens(self.token).to(self.x_in.dtype))
        if self.graph is not None:
            self.graph.replay()
        else:
            self._layers()
        self.logits.copy_(ar.model.lm_head(self.h_out))

    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._layers()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._layers()

    def step(self, token: int):
        self.token.fill_(token)
        self._step()
        self.pos.add_(1)
        return self.logits[0]


class YuE2Pipeline:
    def __init__(self, model: YuE2Model, vae: YuE2VAE):
        self.model = model
        self.vae = vae
        self.do_tiled_decoding = True
        self.use_cuda_graphs = True

    @torch.no_grad()
    def generate_tokens(
        self,
        prefix_embeds: torch.Tensor,  # [1, L, H]
        phase: str,
        max_tokens: int,
        seed: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        penalty_window: int,
        min_tokens: int,
        progress: Optional[Callable[[int, int], None]] = None,
        legacy_off: bool = False,
    ) -> List[int]:
        """Sample AR tokens after ``prefix_embeds`` until the phase's end token (raw vocab ids)."""
        ar = self.model.ar
        device = prefix_embeds.device
        prefix_len = prefix_embeds.shape[1]
        if prefix_len + max_tokens > CONTEXT:
            raise ValueError("YuE2 prompt plus token budget exceeds the model context")
        # reference sampler: seeded generator on the compute device, multinomial over the filtered softmax
        generator = torch.Generator(device=device).manual_seed(seed)
        capacity = prefix_len + max_tokens
        graph = None
        if self.use_cuda_graphs and device.type == "cuda":
            try:
                graph = ARDecodeGraph(ar, capacity, device, ar.dtype)
                prefill_cache, hidden = ar.prefill(prefix_embeds.to(ar.dtype), return_hidden=True)
                logits = ar.model.lm_head(hidden[:, -1])[0]
                graph.load_prefill(prefill_cache, prefix_len)
                del prefill_cache, hidden
                graph.capture()
            except Exception as e:  # capture unsupported for these ops/kernels: eager path
                print(f"YuE2: CUDA graph decode unavailable ({type(e).__name__}: {e}); using eager decode")
                graph = None
                torch.cuda.empty_cache()
        if graph is None:
            cache = KVCache(self.model.cfg, 1, capacity, device, ar.dtype)
            logits = ar.prefill_into_cache(prefix_embeds.to(ar.dtype), cache)[0]
        history: List[int] = []
        min_tokens = min(min_tokens, max_tokens)
        end = ABC_END if phase == "abc" else MUSIC_END
        for step in range(max_tokens):
            token = sample_logits(
                logits, history, step, temperature, top_p, top_k, repetition_penalty, penalty_window, min_tokens, generator,
                phase=phase, legacy_off=legacy_off,
            )
            if token == end:
                break
            history.append(token)
            if progress is not None:
                progress(step + 1, max_tokens)
            if step + 1 < max_tokens:
                if graph is not None:
                    logits = graph.step(token)
                else:
                    emb = ar.embed(torch.tensor([[token]], device=device))
                    logits = ar.decode_step(emb, cache)[0]
        if graph is None:
            del cache
        del graph
        return history

    def generate_codec_tokens(
        self,
        prefix_embeds: torch.Tensor,
        max_tokens: int,
        seed: int,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 100,
        repetition_penalty: float = 1.2,
        penalty_window: int = 50,
        min_tokens: int = 200,
        progress: Optional[Callable[[int, int], None]] = None,
        legacy_off: bool = False,
    ) -> List[int]:
        """Music phase (ComfyUI defaults). Returns codec ids without the vocab offset."""
        ids = self.generate_tokens(
            prefix_embeds, "semantic", max_tokens, seed, temperature, top_p, top_k, repetition_penalty, penalty_window,
            min_tokens, progress, legacy_off=legacy_off,
        )
        return [t - CODEC_OFFSET for t in ids]

    def generate_abc_tokens(
        self,
        prefix_embeds: torch.Tensor,
        seed: int,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 30,
        repetition_penalty: float = 1.005,
        penalty_window: int = 100,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[int]:
        """ABC phase (ComfyUI "Generate ABC" defaults): text ids of the sheet, without ABC_END."""
        return self.generate_tokens(
            prefix_embeds, "abc", max_tokens, seed, temperature, top_p, top_k, repetition_penalty, penalty_window,
            min(32, max_tokens), progress,
        )

    @torch.no_grad()
    def synthesize(
        self,
        prefix_embeds: torch.Tensor,  # [1, L, H]
        codec: List[int],
        seed: int,
        steps: int = 32,
        step_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """-> latents [1, T, 64] in the NAR's dtype."""
        ar, nar = self.model.ar, self.model.nar
        device = prefix_embeds.device
        dtype = nar.dtype
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((len(codec), 64), generator=generator).to(device, dtype)
        ranges = chunk_ranges(len(codec), prefix_embeds.shape[1])
        outputs = []
        total_steps = steps * len(ranges)
        for ci, (start, end) in enumerate(ranges):
            ids = torch.tensor([[c + CODEC_OFFSET for c in codec[start:end]] + [MUSIC_END]], device=device)
            embeds = torch.cat([prefix_embeds.to(ar.dtype), ar.embed(ids)], dim=1)
            cache, _ = ar.prefill(embeds)
            prefix_len = embeds.shape[1]
            state = noise[start:end][None]
            dt = 1.0 / steps
            for step in range(steps):
                t = 1.0 - step * dt
                v1 = nar(state, torch.full((1,), t, device=device, dtype=dtype), cache, prefix_len)
                mid = state - v1 * (dt / 2)
                v2 = nar(mid, torch.full((1,), t - dt / 2, device=device, dtype=dtype), cache, prefix_len)
                state = state - v2 * dt
                if step_callback is not None:
                    step_callback(ci * steps + step, total_steps, state)
            outputs.append(state)
            del cache
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """[1, T, 64] -> waveform [1, 2, samples] float32."""
        z = latents.transpose(1, 2).to(self.vae.device, self.vae.dtype)
        if self.do_tiled_decoding:
            audio = self.vae.tiled_decode(z)
        else:
            audio = self.vae.decode(z)
        return audio.float().clamp(-1, 1)
