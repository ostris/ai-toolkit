"""Launch-light replacement for SheetSage2's ``constrained_prompt_generate``.

Upstream decodes one token per Python iteration: a growing KV cache re-concatenated every step,
the grammar mask rebuilt from ~10 slice kernels, and a host sync per token, which leaves the
6-layer / 512-wide decoder at ~5 ms per token on an idle CPU and far worse under dataloader
load. Here the grammar (``PromptGrammarState``) is tabulated once into ``allowed[state, token]``
and ``next_state[state, token_type]`` lookups, the decoder runs on a ``StaticCache``, and the
whole per-token step (mask, argmax, state update, decoder forward for the chosen token) is one
CUDA graph replayed with a host sync every ``SYNC_EVERY`` tokens. Batch 1 only (what upstream's
``constrained_prompt_generate`` accepts); anything else falls back to the upstream function.
"""

import copy
import math
from typing import Optional

import torch

from .sheetsage_decoder import StaticCache

SYNC_EVERY = 32
TYPE_NAMES = ("subbeat_shift", "time", "meter", "eighth_position", "structure", "key", "chord_full", "pitch", "duration")
EOS_TYPE = len(TYPE_NAMES)
OTHER_TYPE = EOS_TYPE + 1
INCOMPLETE = (None, "rhythm_after_meter", "melody_after_pitch")
N_STATES = 2 * 2 * 5 * 7 * 3
NO_STOP = 1 << 40


def state_index(state) -> int:
    """PromptGrammarState -> row; ``allowed``/``update`` only read these fields (shift_run only as < 4)."""
    key = (int(state.payload_count > 0), int(state.in_shift), min(int(state.shift_run), 4), int(state.last_field_index) + 1, INCOMPLETE.index(state.incomplete))
    return (((key[0] * 2 + key[1]) * 5 + key[2]) * 7 + key[3]) * 3 + key[4]


def build_tables(tokenizer, state_cls):
    """Enumerate the whole grammar key space: allowed [S, V] bool, next_state [S, n_types] long, token type [V] long."""
    attrs = [{"chord_full": "full_chord"}.get(n, n) for n in TYPE_NAMES]
    ranges = [(getattr(tokenizer, f"{a}_token_start"), getattr(tokenizer, f"{a}_token_end")) for a in attrs]
    type_of = torch.full((tokenizer.n_tokens,), OTHER_TYPE, dtype=torch.long)
    for t, (s, e) in enumerate(ranges):
        type_of[s:e] = t
    type_of[tokenizer.eos_token] = EOS_TYPE
    allowed = torch.zeros(N_STATES, tokenizer.n_tokens, dtype=torch.bool)
    next_state = torch.arange(N_STATES, dtype=torch.long)[:, None].repeat(1, OTHER_TYPE + 1)
    for payload in (0, 1):
        for in_shift in (False, True):
            for run in range(5):
                for lfi in range(-1, 6):
                    for inc in INCOMPLETE:
                        st = state_cls(tokenizer)
                        st.payload_count, st.in_shift, st.shift_run, st.last_field_index, st.incomplete = payload, in_shift, run, lfi, inc
                        i = state_index(st)
                        allowed[i] = st.allowed(torch.device("cpu"))
                        for t, (s, e) in enumerate(ranges):
                            if s < e:
                                nxt = copy.copy(st)
                                nxt.update(s)
                                next_state[i, t] = state_index(nxt)
    return allowed, next_state, type_of


def stop_time_id(stop_seconds: Optional[float], time_hz: int) -> int:
    """Smallest time id with ``id / time_hz >= stop`` under float division, as upstream compares."""
    if stop_seconds is None:
        return NO_STOP
    t = max(0, math.ceil(float(stop_seconds) * time_hz))
    while t > 0 and (t - 1) / time_hz >= stop_seconds:
        t -= 1
    while t / time_hz < stop_seconds:
        t += 1
    return t


class FastConstrainedGenerate:
    """Callable with upstream's ``constrained_prompt_generate`` signature; ``upstream`` handles unsupported calls."""

    def __init__(self, upstream):
        self.upstream = upstream
        self.model = None
        self.key = None
        self.graph = None

    def __call__(self, model, audio, prompts, max_sequence_length, prefix_tokens=None, autocast_dtype=torch.bfloat16,
                 stop_time_seconds=None, progress_callback=None, memory=None, step_callback=None):
        if step_callback is not None or audio.device.type != "cuda" or audio.shape[0] != 1:
            return self.upstream(model, audio, prompts, max_sequence_length, prefix_tokens=prefix_tokens, autocast_dtype=autocast_dtype,
                                 stop_time_seconds=stop_time_seconds, progress_callback=progress_callback, memory=memory, step_callback=step_callback)
        tokenizer = model.tokenizer
        prefix = list(prefix_tokens) if prefix_tokens is not None else tokenizer.prompt_prefix(prompts)
        if not prefix or prefix[0] != tokenizer.sos_token:
            raise ValueError("generation prefix must begin with <|sos|>")
        if prefix[-1] == tokenizer.eos_token:
            prefix = prefix[:-1]
        max_len = int(max_sequence_length)
        if len(prefix) >= max_len:
            return torch.tensor(prefix + [tokenizer.eos_token], dtype=torch.long)
        autocast = torch.autocast(device_type="cuda", dtype=autocast_dtype, cache_enabled=False) if autocast_dtype is not None else torch.autocast(device_type="cuda", enabled=False)
        if memory is None:
            with autocast:
                memory = model.encode(audio)
        self._ensure_built(model, audio.device, max_len, memory.shape[1], autocast_dtype)

        state = self._state_cls(tokenizer)
        for token in prefix[prefix.index(tokenizer.out_token) + 1 :]:
            state.update(token)
        self.cache.fill_cross(self.dec, memory)
        hidden = self.dec.prefill_static(torch.tensor([prefix], dtype=torch.long, device=audio.device), self.cache)
        self.logits.copy_(self._project(hidden))
        self.state.fill_(state_index(state))
        self.stop_id.fill_(stop_time_id(stop_time_seconds, tokenizer.time_hz))
        self.gen_count.zero_()
        self.finish_pos.fill_(-1)
        self.done.zero_()

        budget = max_len - len(prefix)
        steps = 0
        while steps < budget:
            for _ in range(min(SYNC_EVERY, budget - steps)):
                self.graph.replay()
                steps += 1
            if progress_callback is not None:
                progress_callback(len(prefix) + steps)
            if self.done.item():
                break
        n = int(self.finish_pos.item()) + 1 if self.done.item() else steps
        generated = self.out[:n].tolist()
        if any(int(self.type_of[t]) == OTHER_TYPE for t in generated):
            raise RuntimeError("Unexpected prompt token type in constrained generation")
        tokens = prefix + generated
        if tokens[-1] != tokenizer.eos_token:
            tokens.append(tokenizer.eos_token)
        return torch.tensor(tokens, dtype=torch.long)

    def _project(self, hidden):
        """last position of [1, L, dim] -> fp32 logits [1, V]"""
        return torch.nn.functional.linear(hidden[:, -1].to(self.proj_w.dtype), self.proj_w).float()

    # ------------------------------------------------------------------
    def _ensure_built(self, model, device, max_len, memory_len, autocast_dtype):
        key = (id(model), str(device), max_len, memory_len, autocast_dtype)
        if self.key == key and self.graph is not None:
            return
        import importlib

        gen = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".generation_sheetsage2")
        self._state_cls = gen.PromptGrammarState
        self.model, self.key = model, key
        tok = model.tokenizer
        allowed, next_state, type_of = build_tables(tok, self._state_cls)
        self.allowed, self.next_state = allowed.to(device), next_state.to(device)
        self.type_of = type_of  # host copy for the post-check
        self.type_of_dev = type_of.to(device)
        cache_len = max_len + SYNC_EVERY  # replays may overshoot the budget by < SYNC_EVERY after finishing
        self.cache = StaticCache(model.decoder, cache_len, memory_len, device, autocast_dtype or torch.float32)
        self.logits = torch.zeros(1, tok.n_tokens, device=device)
        self.state = torch.zeros(1, dtype=torch.long, device=device)
        self.stop_id = torch.zeros(1, dtype=torch.long, device=device)
        self.gen_count = torch.zeros(1, dtype=torch.long, device=device)
        self.finish_pos = torch.full((1,), -1, dtype=torch.long, device=device)
        self.done = torch.zeros(1, dtype=torch.bool, device=device)
        self.out = torch.zeros(cache_len, dtype=torch.long, device=device)
        self.cur = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.eos = tok.eos_token
        self.time_start = tok.time_token_start
        # bf16 copies of the decoder linears and the tied projection table: what autocast would cast on every step
        dtype = autocast_dtype or torch.float32
        self.dec = copy.deepcopy(model.decoder, memo={id(model.decoder.embed_tokens): model.decoder.embed_tokens})
        for m in self.dec.modules():
            if isinstance(m, torch.nn.Linear):
                m.to(dtype)
        self.proj_w = model.output_projection.weight.detach().to(dtype)

        def step():
            masked = self.logits.masked_fill(~self.allowed[self.state], float("-inf"))
            tok_id = masked.argmax(-1)  # [1]
            ttype = self.type_of_dev[tok_id]
            stop_hit = (ttype == 1) & ((tok_id - self.time_start) >= self.stop_id)
            finished = (tok_id == self.eos) | stop_hit
            self.out.index_copy_(0, self.gen_count, tok_id)
            self.finish_pos.copy_(torch.where(finished & (self.finish_pos < 0), self.gen_count, self.finish_pos))
            self.done.logical_or_(finished)
            self.gen_count += 1
            self.state.copy_(self.next_state[self.state, ttype])
            self.cur.copy_(tok_id[:, None])
            self.logits.copy_(self._project(self.dec.step_static(self.cur, self.cache)))

        # warm up on a side stream (cuBLAS workspaces, autocast paths), then capture
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream(device).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            step()
        torch.cuda.synchronize(device)
