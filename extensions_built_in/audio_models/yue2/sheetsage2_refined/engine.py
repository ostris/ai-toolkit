"""Batched greedy decoding: static KV cache, on-device grammar, CUDA graphs.

``BatchDecoder`` holds ``batch`` independent sequences ("slots").  One step
feeds every slot its last token, writes that token's keys and values at its own
position, picks the next token under the grammar mask and advances the grammar,
all on the device.  On CUDA a step is captured once per cache-length bucket and
replayed, so the host only launches graphs and polls for finished slots every
few dozen steps.  Finished or empty slots keep computing harmlessly until they
are reloaded, which lets the caller refill them with new windows (continuous
batching).

Numerically a step matches running the decoder under bf16 autocast: Linear and
output-projection weights are bf16 copies of what autocast would cast, the
token-embedding lookup stays fp32, and single-query attention uses plain
matmuls (the fused SDPA kernels idle the GPU with one query per head).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import POSITION_OFFSET, Attention, TokenDecoder
from .grammar import AFTER_METER, ENDED, TokenGrammar
from .vocab import EOS, FULL_PROMPTS, METERS, NO_CHORD, NO_KEY, OUT, Vocab

# Terms a frame head can rescore, and the prompt whose tokens each one moves.
FRAME_TERMS = {
    "root": "chord_detail",
    "bass": "chord_detail",
    "detail": "chord_detail",  # root x quality (a head with a ``detail`` output)
    "quality": "chord_detail",  # the quality given the root, from the same output (the root left to ``root``)
    "key": "key",
    "key_boundary": "key",
    "downbeat": "downbeat_meter",
    "meter": "downbeat_meter",
}


def frame_terms(weights: dict[str, float] | None, prompts) -> tuple[float, ...]:
    """``weights`` (term -> weight) as a tuple in ``FRAME_TERMS`` order, zero for the terms whose
    tokens the prompts never emit."""

    weights = weights or {}
    unknown = set(weights) - set(FRAME_TERMS)
    if unknown:
        raise ValueError(f"unknown frame terms {sorted(unknown)}; known: {list(FRAME_TERMS)}")
    return tuple(float(weights.get(name, 0.0)) if prompt in prompts else 0.0 for name, prompt in FRAME_TERMS.items())


class BatchDecoder:
    BUCKETS = (256, 512, 1024, 2048, 4096)

    def __init__(
        self,
        decoder: TokenDecoder,
        vocab: Vocab,
        *,
        batch: int,
        frames: int,
        prompts=FULL_PROMPTS,
        mode_bias: list[float] | None = None,
        frame_weight: dict[str, float] | None = None,
        dtype=torch.bfloat16,
    ):
        config = decoder.config
        self.device = device = decoder.token_embedding.weight.device
        if device.type != "cuda":
            dtype = torch.float32  # no autocast on the CPU path
        self.decoder, self.dtype, self.batch, self.frames = decoder, dtype, batch, frames
        self.max_tokens = config.max_tokens
        self.versions = self._versions(decoder)
        self.grammar = TokenGrammar(vocab, device, prompts)
        # Optional additive bias on key_mode tokens by their mode (mode_bias[m] for theory.MODES[m]),
        # e.g. -tau * log P(mode): the decoder ranks modes well but rarely picks the rare ones.
        self.mode_bias = mode_bias if "key_mode" in prompts else None
        if self.mode_bias is not None:
            self.mode_span = (vocab.start["key_mode"], vocab.start["key_mode"] + NO_KEY)
            self.mode_table = torch.tensor(self.mode_bias, dtype=torch.float32, device=device).repeat_interleave(12)
        # Optional rescoring by frame heads (``heads.FrameHead``), one weight per FRAME_TERMS
        # entry, looked up at the frame of the slot's current time: root x quality gets
        # w_root * log P(root), the bass token w_bass * log P(bass), key tokens w_key * log P(key)
        # and, once the window's key is stated, w_key_boundary * logit P(key change); the
        # downbeat position (eighth position 0) gets w_downbeat * logit P(downbeat).  With a meter
        # weight, a change to a meter of n beats with the active meter's beat unit (x/4 after x/4)
        # gets w_meter * (log P(n beats) - log P(beats of the active meter)) from the head's bar
        # length, so that the decoder states a new meter where the head hears a bar of another length
        # (the head does not choose the beat unit: a switch it pushed at the wrong moment left the
        # beats half a beat off).  A change starts a bar, so it also gets the downbeat term and may
        # only be followed by eighth position 0.  Restating the active meter (a window's first
        # event) is left alone.
        # Each slot's tables come with its window (``load``).
        self.frame_weight = frame_terms(frame_weight, prompts)
        self.rescore = any(self.frame_weight)
        if self.rescore:
            sizes = {"root": 12, "bass": 12, "detail": NO_CHORD, "quality": NO_CHORD, "key": 24, "meter": 16}
            self.frame = {
                name: torch.zeros(batch, frames, *((sizes[name],) if name in sizes else ()), device=device)
                for name in FRAME_TERMS
            }
            self.frame_now = torch.zeros(batch, dtype=torch.long, device=device)
            self.key_seen = torch.zeros(batch, dtype=torch.bool, device=device)
            self.root_of_detail = torch.arange(NO_CHORD, device=device) % 12
            self.key_span = (vocab.start["key"], vocab.end["key"])
            self.eighth_span = (vocab.start["eighth_position"], vocab.end["eighth_position"])
            self.meter_span = (vocab.start["meter"], vocab.end["meter"])
            # bar-length class (beats - 1, at most 15) of every meter token
            self.meter_beats = torch.tensor([min(n, 16) - 1 for n, _ in METERS], device=device)
            self.meter_unit = torch.tensor([d for _, d in METERS], device=device)
            self.active_meter = torch.full((batch,), -1, dtype=torch.long, device=device)  # meter index, -1 none
            self.meter_changed = torch.zeros(batch, dtype=torch.bool, device=device)  # the last meter token changed it
            self.meter_index = torch.arange(len(METERS), device=device)
            self.time_hz = vocab.time_hz
        self.bass_span = (vocab.start["chord_bass"], vocab.end["chord_bass"])
        self.detail_span = (vocab.start["chord_detail"], vocab.start["chord_detail"] + NO_CHORD)

        self.weights = {
            id(module): (module.weight.detach().to(dtype), module.bias.detach().to(dtype))
            for module in decoder.modules()
            if isinstance(module, nn.Linear)
        }
        self.embedding = decoder.token_embedding.weight.detach().to(dtype)

        heads, head_dim = config.heads, config.width // config.heads
        zeros = lambda *shape: torch.zeros(*shape, dtype=dtype, device=device)  # noqa: E731
        self.self_kv = [
            (zeros(batch, heads, self.max_tokens, head_dim), zeros(batch, heads, self.max_tokens, head_dim))
            for _ in decoder.layers
        ]
        self.cross_kv = [
            (zeros(batch, heads, frames, head_dim), zeros(batch, heads, frames, head_dim)) for _ in decoder.layers
        ]
        # Additive cross-attention bias that hides each slot's padding frames.
        self.cross_bias = torch.zeros(batch, 1, 1, frames, device=device)
        self.slots = torch.arange(self.max_tokens, device=device)
        self.rows = torch.arange(batch, device=device)
        self.out = torch.zeros(batch, self.max_tokens, dtype=torch.long, device=device)
        # Mutable per-slot state, all updated in place by a step.
        self.state = TokenGrammar.new_state(batch, device)
        self.state["finished"].fill_(True)
        self.state["token"] = torch.full((batch,), EOS, dtype=torch.long, device=device)
        self.state["position"] = torch.zeros(batch, dtype=torch.long, device=device)
        self.stop = torch.full((batch,), float("inf"), dtype=torch.float64, device=device)

        self.bound = [0] * batch  # host-side upper bound of each slot's position
        self.loaded = [False] * batch
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle() if device.type == "cuda" else None

    @staticmethod
    def _versions(decoder: TokenDecoder) -> tuple[int, ...]:
        return tuple(parameter._version for parameter in decoder.parameters())

    def is_current(
        self,
        decoder: TokenDecoder,
        batch: int,
        frames: int,
        prompts,
        mode_bias=None,
        frame_weight=None,
    ) -> bool:
        """False if built for another decoder, shape, prompt set, rescoring, or weights that changed since."""

        return (
            decoder is self.decoder
            and (mode_bias if "key_mode" in prompts else None) == self.mode_bias
            and frame_terms(frame_weight, prompts) == self.frame_weight
            and batch == self.batch
            and frames == self.frames
            and tuple(sorted(set(prompts))) == self.grammar.prompts
            and self._versions(decoder) == self.versions
        )

    # ------------------------------------------------------------------ slots
    def free_slots(self) -> list[int]:
        return [slot for slot in range(self.batch) if not self.loaded[slot]]

    def load(
        self,
        slot: int,
        memory: Tensor,
        prefix: list[int],
        stop_seconds: float | None,
        frames: int | None = None,
        frame_logp: dict[str, Tensor] | None = None,
    ) -> None:
        """Start ``slot`` on a window: ``memory`` ``[1, frames, D]``, token ``prefix`` (ends after <out>).

        ``frames`` limits cross-attention to the first ``frames`` memory frames.  ``frame_logp``
        holds the window's per-frame tables (``transcribe.frame_tables``) when the engine
        rescores by a frame head.
        """

        length = len(prefix)
        frames = self.frames if frames is None else frames
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            cache = self.decoder.new_cache(memory, torch.tensor([frames], device=self.device))
            if length > 1:
                self.decoder(torch.tensor([prefix[:-1]], device=self.device), cache)
        for (key, value), (cross_key, cross_value), layer in zip(self.self_kv, self.cross_kv, cache):
            if length > 1:
                key[slot, :, : length - 1].copy_(layer["self_key"][0])
                value[slot, :, : length - 1].copy_(layer["self_value"][0])
            cross_key[slot].copy_(layer["cross_key"][0])
            cross_value[slot].copy_(layer["cross_value"][0])
        self.cross_bias[slot].zero_()
        self.cross_bias[slot, ..., frames:] = float("-inf")
        self.out[slot, :length] = torch.tensor(prefix, device=self.device)
        for name, value in self.grammar.prefix_state(prefix, prefix.index(OUT), self.device).items():
            self.state[name][slot] = value[0]
        self.state["token"][slot] = prefix[-1]
        self.state["position"][slot] = length - 1
        self.stop[slot] = float("inf") if stop_seconds is None else stop_seconds
        self.bound[slot], self.loaded[slot] = length - 1, True
        if self.rescore:
            for name, table in self.frame.items():
                if name not in frame_logp:  # a term without a head stays at zero
                    table[slot].zero_()
                    continue
                count = min(len(frame_logp[name]), self.frames)
                table[slot].zero_()
                table[slot, :count] = frame_logp[name][:count]
            seconds = self.grammar.time_seconds.cpu()
            times = [float(seconds[t]) for t in prefix if seconds[t] > float("-inf")]
            self.frame_now[slot] = min(int(times[-1] * self.time_hz) * 25 // self.time_hz, self.frames - 1) if times else 0
            self.key_seen[slot] = any(self.key_span[0] <= t < self.key_span[1] for t in prefix)
            m0, m1 = self.meter_span
            meters = [t - m0 for t in prefix if m0 <= t < m1]
            self.active_meter[slot] = meters[-1] if meters else -1
            self.meter_changed[slot] = False

    def run(self, steps: int) -> None:
        """Advance every slot by ``steps`` tokens (finished slots stay put)."""

        for _ in range(steps):
            top = max((b for b, on in zip(self.bound, self.loaded) if on), default=0)
            bucket = next((b for b in self.BUCKETS if b > top), self.max_tokens)
            if self.device.type == "cuda":
                if bucket not in self.graphs:
                    self.graphs[bucket] = self._capture(bucket)
                self.graphs[bucket].replay()
            else:
                self._forward(bucket)
            self.bound = [min(b + 1, self.max_tokens - 1) if on else b for b, on in zip(self.bound, self.loaded)]

    def finished(self) -> list[int]:
        """Loaded slots whose sequence has ended (synchronises with the device)."""

        done = self.state["finished"].tolist()
        return [slot for slot in range(self.batch) if self.loaded[slot] and done[slot]]

    def take(self, slot: int) -> tuple[list[int], str]:
        """Return a finished slot's tokens (always ending in <eos>) and why it ended; free the slot."""

        length = int(self.state["position"][slot]) + 1
        tokens = self.out[slot, :length].tolist()
        if tokens[-1] != EOS:
            tokens.append(EOS)
        self.loaded[slot] = False
        return tokens, ENDED[int(self.state["ended"][slot])]

    # ------------------------------------------------------------------- step
    def _capture(self, bucket: int) -> torch.cuda.CUDAGraph:
        # Warm-up runs execute the step for real, so restore the state afterwards.
        saved = {name: value.clone() for name, value in self.state.items()}
        saved_out = self.out.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._forward(bucket)
        torch.cuda.current_stream().wait_stream(stream)
        for name, value in saved.items():
            self.state[name].copy_(value)
        self.out.copy_(saved_out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            self._forward(bucket)
        return graph

    def _linear(self, module: nn.Linear, x: Tensor) -> Tensor:
        return F.linear(x, *self.weights[id(module)])

    def _attend(self, attn: Attention, x: Tensor, key: Tensor, value: Tensor, bias: Tensor | None = None) -> Tensor:
        query = attn.split(self._linear(attn.q, x))
        scores = query @ key.transpose(-1, -2) * query.shape[-1] ** -0.5
        if bias is not None:
            scores = scores + bias
        y = torch.softmax(scores, dim=-1).to(value.dtype) @ value
        return self._linear(attn.out, y.transpose(1, 2).reshape(x.shape[0], 1, -1))

    def _forward(self, bucket: int) -> None:
        decoder, state, rows = self.decoder, self.state, self.rows
        position, token = state["position"], state["token"]
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            x = (
                decoder.token_embedding(token[:, None])
                + decoder.position_embedding(position + POSITION_OFFSET)[:, None]
            )
            x = decoder.embedding_norm(x)
            slots = self.slots[:bucket]
            bias = torch.zeros(self.batch, bucket, device=x.device).masked_fill(
                slots[None] > position[:, None], float("-inf")
            )
            bias = bias[:, None, None, :]
            for layer, (key, value), (cross_key, cross_value) in zip(decoder.layers, self.self_kv, self.cross_kv):
                attn = layer.self_attn
                key[rows, :, position] = attn.split(self._linear(attn.k, x))[:, :, 0].to(self.dtype)
                value[rows, :, position] = attn.split(self._linear(attn.v, x))[:, :, 0].to(self.dtype)
                x = layer.self_attn_norm(x + self._attend(attn, x, key[:, :, :bucket], value[:, :, :bucket], bias))
                x = layer.cross_attn_norm(x + self._attend(layer.cross_attn, x, cross_key, cross_value, self.cross_bias))
                x = layer.final_norm(x + self._linear(layer.fc2, F.gelu(self._linear(layer.fc1, x))))
            logits = F.linear(x, self.embedding)[:, -1].float()
            if self.mode_bias is not None:
                logits[:, self.mode_span[0] : self.mode_span[1]] += self.mode_table
            if self.rescore:
                (b0, b1), (d0, d1), (k0, k1) = self.bass_span, self.detail_span, self.key_span
                w, now, frame = dict(zip(FRAME_TERMS, self.frame_weight)), self.frame_now, self.frame
                if w["root"]:
                    logits[:, d0:d1] += w["root"] * frame["root"][rows, now][:, self.root_of_detail]
                if w["detail"]:
                    logits[:, d0:d1] += w["detail"] * frame["detail"][rows, now]
                if w["quality"]:
                    logits[:, d0:d1] += w["quality"] * frame["quality"][rows, now]
                if w["bass"]:
                    logits[:, b0:b1] += w["bass"] * frame["bass"][rows, now]
                if w["key"]:
                    logits[:, k0:k1] += w["key"] * frame["key"][rows, now]
                if w["key_boundary"]:
                    stated = self.key_seen.float()
                    logits[:, k0:k1] += (w["key_boundary"] * frame["key_boundary"][rows, now] * stated)[:, None]
                if w["downbeat"]:
                    logits[:, self.eighth_span[0]] += w["downbeat"] * frame["downbeat"][rows, now]
                if w["meter"]:
                    (m0, m1), (e0, e1) = self.meter_span, self.eighth_span
                    table = frame["meter"][rows, now]
                    active = self.active_meter.clamp_min(0)
                    change = (self.active_meter >= 0)[:, None] & (self.meter_index[None] != active[:, None])
                    current = table.gather(1, self.meter_beats[active][:, None])
                    same_unit = self.meter_unit[None] == self.meter_unit[active][:, None]
                    logits[:, m0:m1] += w["meter"] * (table[:, self.meter_beats] - current) * (change & same_unit)
                    if w["downbeat"]:
                        logits[:, m0:m1] += w["downbeat"] * frame["downbeat"][rows, now][:, None] * change
                    after = (state["incomplete"] == AFTER_METER) & self.meter_changed
                    logits[:, e0 + 1 : e1].masked_fill_(after[:, None], float("-inf"))

        allowed = self.grammar.allowed(state)
        choice = logits.masked_fill(~allowed, float("-inf")).argmax(-1)
        active = ~state["finished"]
        choice = torch.where(active, choice, token)
        new_position = torch.where(active, position + 1, position)
        self.out[rows, new_position] = torch.where(active, choice, self.out[rows, new_position])
        self.grammar.update(state, choice, active, self.stop, new_position >= self.max_tokens - 1)
        token.copy_(choice)
        position.copy_(new_position)
        if self.rescore:
            seconds = self.grammar.time_seconds[choice]
            frame = (seconds * 25).long().clamp(0, self.frames - 1)
            self.frame_now.copy_(torch.where(active & (seconds > float("-inf")), frame, self.frame_now))
            k0, k1 = self.key_span
            self.key_seen.copy_(self.key_seen | (active & (choice >= k0) & (choice < k1)))
            m0, m1 = self.meter_span
            is_meter = active & (choice >= m0) & (choice < m1)
            meter = (choice - m0).clamp(0, m1 - m0 - 1)
            changed = (self.active_meter >= 0) & (meter != self.active_meter)
            self.meter_changed.copy_(torch.where(is_meter, changed, self.meter_changed))
            self.active_meter.copy_(torch.where(is_meter, meter, self.active_meter))
