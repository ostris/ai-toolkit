"""Inference-time LoRA loading for the engine: any common LoRA file format ->
per-module low-rank deltas, applied either as forward hooks (dynamic strength,
removable) or merged into the weights (quantized weights re-quantized with
stochastic rounding so small deltas survive).

Independent of the training network classes on purpose.

Supported key formats (auto-detected per key):
  peft / diffusers   transformer.blocks.0.attn.to_q.lora_A.weight / lora_B
  comfy              diffusion_model.<...>.lora_A|lora_down (+ .alpha)
  kohya / lycoris    lora_unet_<flat_name>.lora_down.weight / lora_up + alpha,
                     lora_te_ / lora_te1_ / lora_te2_ for text encoders
  full diffs         <module>.diff (weight delta), <module>.diff_b (bias delta)
The holder's convert_lora_weights_before_load hook runs first, so archs with
their own conversions (anima, ltx2, hidream_o1) keep working.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

_A_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora_A.default.weight")
_B_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora_B.default.weight")
_TE_PREFIXES = {
    "lora_te1_": 0, "lora_te_": 0, "lora_te2_": 1, "lora_te3_": 2,
    "text_encoder.": 0, "text_encoder_2.": 1, "text_encoder_3.": 2,
    "te.": 0, "te1.": 0, "te2.": 1,
}
_DIT_PREFIXES = (
    "model.diffusion_model.", "diffusion_model.", "transformer.", "unet.",
    "lora_unet_", "lora_transformer_", "base_model.model.",
)


class LoRAEntry:
    def __init__(self, module: torch.nn.Module, name: str):
        self.module = module
        self.name = name
        self.A: Optional[torch.Tensor] = None  # (r, in)
        self.B: Optional[torch.Tensor] = None  # (out, r)
        self.alpha: Optional[float] = None
        self.diff: Optional[torch.Tensor] = None
        self.diff_b: Optional[torch.Tensor] = None

    @property
    def scale(self) -> float:
        if self.A is None or self.alpha is None:
            return 1.0
        r = self.A.shape[0]
        return float(self.alpha) / r if r else 1.0

    def delta(self, strength: float, device, dtype=torch.float32) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(weight delta, bias delta) at this strength, full precision."""
        dw = None
        if self.A is not None and self.B is not None:
            A = self.A.to(device=device, dtype=dtype)
            B = self.B.to(device=device, dtype=dtype)
            if A.dim() == 4:  # conv lora: (r, in, kh, kw) @ (out, r, 1, 1)
                dw = torch.einsum("or,rikl->oikl", B.flatten(1), A) * (self.scale * strength)
            else:
                dw = (B @ A) * (self.scale * strength)
        if self.diff is not None:
            d = self.diff.to(device=device, dtype=dtype) * strength
            dw = d if dw is None else dw + d
        db = self.diff_b.to(device=device, dtype=dtype) * strength if self.diff_b is not None else None
        return dw, db


class InferenceLoRA:
    def __init__(self, path: str, strength: float = 1.0, name: Optional[str] = None):
        self.path = path
        self.strength = float(strength)
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.entries: List[LoRAEntry] = []
        self.unmatched: List[str] = []
        self._hooks: List = []

    # ---- loading / key resolution ----
    def load(self, holder) -> "InferenceLoRA":
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"LoRA not found: {self.path}")
        sd = load_file(self.path)
        convert = getattr(holder, "convert_lora_weights_before_load", None)
        if callable(convert):
            try:
                sd = convert(sd)
            except Exception:
                pass
        roots = self._roots(holder)
        grouped: Dict[Tuple[int, str], dict] = {}
        for key, tensor in sd.items():
            base, part = self._split(key)
            if part is None:
                continue
            grouped.setdefault(base, {})[part] = tensor
        self.entries = []
        self.unmatched = []
        for base, parts in grouped.items():
            module, name = self._resolve(base, roots)
            if module is None:
                self.unmatched.append(base)
                continue
            e = LoRAEntry(module, name)
            e.A = parts.get("A")
            e.B = parts.get("B")
            e.diff = parts.get("diff")
            e.diff_b = parts.get("diff_b")
            alpha = parts.get("alpha")
            if alpha is not None:
                e.alpha = float(alpha.flatten()[0])
            if e.A is not None and e.B is not None and e.A.dim() == 2:
                w_shape = getattr(getattr(module, "weight", None), "shape", None)
                if w_shape is not None and (e.B.shape[0] != w_shape[0] or e.A.shape[1] != w_shape[1]):
                    self.unmatched.append(f"{base} (shape {tuple(e.B.shape[0:1]) + tuple(e.A.shape[1:])} vs {tuple(w_shape)})")
                    continue
            if e.A is None and e.diff is None and e.diff_b is None:
                continue
            self.entries.append(e)
        return self

    @staticmethod
    def _split(key: str):
        for suf in _A_SUFFIXES:
            if key.endswith(suf):
                return key[: -len(suf)], "A"
        for suf in _B_SUFFIXES:
            if key.endswith(suf):
                return key[: -len(suf)], "B"
        if key.endswith(".alpha"):
            return key[: -len(".alpha")], "alpha"
        if key.endswith(".diff"):
            return key[: -len(".diff")], "diff"
        if key.endswith(".diff_b"):
            return key[: -len(".diff_b")], "diff_b"
        return key, None

    @staticmethod
    def _roots(holder) -> List[Tuple[str, torch.nn.Module, dict, dict]]:
        """(kind, root module, dotted-name map, flat-name map) for the DiT and TEs."""
        roots = []
        dit = getattr(holder, "model", None)
        if isinstance(dit, torch.nn.Module):
            names = dict(dit.named_modules())
            roots.append(("dit", dit, names, {n.replace(".", "_"): n for n in names}))
        tes = getattr(holder, "text_encoder", None)
        tes = tes if isinstance(tes, list) else [tes]
        for i, te in enumerate(tes):
            if isinstance(te, torch.nn.Module) and not type(te).__name__.startswith("Fake"):
                names = dict(te.named_modules())
                roots.append((f"te{i}", te, names, {n.replace(".", "_"): n for n in names}))
        return roots

    @classmethod
    def _resolve(cls, base: str, roots):
        te_index = None
        rest = base
        for pref, idx in _TE_PREFIXES.items():
            if base.startswith(pref):
                te_index = idx
                rest = base[len(pref):]
                break
        if te_index is None:
            for pref in _DIT_PREFIXES:
                if base.startswith(pref):
                    rest = base[len(pref):]
                    break
        candidates = [r for r in roots if (r[0] == f"te{te_index}" if te_index is not None else r[0] == "dit")]
        if not candidates:
            candidates = roots
        for _, root, names, flat in candidates:
            if rest in names:
                return names[rest], rest
            flat_key = rest.replace(".", "_")
            if flat_key in flat:
                n = flat[flat_key]
                return names[n], n
        return None, None

    # ---- hook mode ----
    def attach(self):
        """Forward hooks that add strength * delta(x) to each module's output."""
        self.detach()
        for e in self.entries:
            if e.A is None or e.A.dim() != 2:
                if e.diff is None and e.diff_b is None:
                    continue
            lora = self

            def make(entry: LoRAEntry):
                cache = {}

                def hook(module, inputs, output):
                    if lora.strength == 0:
                        return output
                    x = inputs[0] if inputs else None
                    if x is None:
                        return output
                    key = (x.device, x.dtype)
                    if key not in cache:
                        cache.clear()
                        cache[key] = (
                            entry.A.to(device=x.device, dtype=x.dtype) if entry.A is not None and entry.A.dim() == 2 else None,
                            entry.B.to(device=x.device, dtype=x.dtype) if entry.B is not None and entry.B.dim() == 2 else None,
                            entry.diff.to(device=x.device, dtype=x.dtype) if entry.diff is not None and entry.diff.dim() == 2 else None,
                            entry.diff_b.to(device=x.device, dtype=x.dtype) if entry.diff_b is not None else None,
                        )
                    A, B, diff, diff_b = cache[key]
                    add = None
                    if A is not None and B is not None:
                        add = ((x @ A.t()) @ B.t()) * (entry.scale * lora.strength)
                    if diff is not None:
                        d = (x @ diff.t()) * lora.strength
                        add = d if add is None else add + d
                    if diff_b is not None:
                        add = (diff_b * lora.strength) if add is None else add + diff_b * lora.strength
                    if add is None:
                        return output
                    if isinstance(output, tuple):
                        return (output[0] + add.to(output[0].dtype),) + tuple(output[1:])
                    return output + add.to(output.dtype)

                return hook

            self._hooks.append(e.module.register_forward_hook(make(e)))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ---- merge mode ----
    @torch.no_grad()
    def merge(self, stochastic: bool = True) -> int:
        """Add strength * delta into each module's weight. Quantized (Ostris)
        weights: dequantize -> add -> requantize on the stored grid with
        stochastic rounding (int8 backends), nearest otherwise. Plain weights
        in bf16/fp16 on cuda: stochastic rounding via copy_stochastic."""
        from toolkit.optimizers.optimizer_utils import copy_stochastic

        merged = 0
        for e in self.entries:
            m = e.module
            w = getattr(m, "weight", None)
            if w is None:
                continue
            device = w.device
            dw, db = e.delta(self.strength, device)
            if dw is not None:
                if getattr(m, "is_ostris_quantized", False):
                    fp = m.dequantize_weight().to(torch.float32) + dw.reshape(w.shape)
                    q = m.ostris_quantizer
                    rq_codes = getattr(q, "requantize_codes_", None)
                    if rq_codes is not None and stochastic:
                        try:
                            rq_codes(m, fp, stochastic=True)
                        except TypeError:
                            rq_codes(m, fp)
                    elif rq_codes is not None:
                        rq_codes(m, fp)
                    else:
                        m.requantize_(fp)
                elif isinstance(w, torch.nn.Parameter) or isinstance(w, torch.Tensor):
                    fp = w.data.to(torch.float32) + dw.reshape(w.shape)
                    if stochastic and w.is_cuda and w.dtype in (torch.bfloat16, torch.float16):
                        copy_stochastic(w.data, fp)
                    else:
                        w.data.copy_(fp.to(w.dtype))
                else:
                    continue
            if db is not None and getattr(m, "bias", None) is not None:
                m.bias.data.add_(db.to(m.bias.dtype))
            merged += 1
        m_marker = getattr(self, "_marker", None)
        return merged

    def summary(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "strength": self.strength,
            "modules": len(self.entries),
            "unmatched": len(self.unmatched),
        }


class LoRAStack:
    """The LoRAs applied to one holder, in one mode."""

    def __init__(self, holder, mode: str = "hook"):
        self.holder = holder
        self.mode = mode
        self.loras: List[InferenceLoRA] = []

    def key(self) -> str:
        return "|".join(f"{l.path}@{l.strength:.4f}" for l in self.loras) + f"#{self.mode}"

    def load(self, specs: List[dict], status_fn=None):
        for spec in specs:
            lora = InferenceLoRA(spec["path"], spec.get("strength", 1.0), spec.get("name")).load(self.holder)
            if status_fn:
                status_fn(
                    f"LoRA {lora.name}: {len(lora.entries)} modules"
                    + (f", {len(lora.unmatched)} unmatched" if lora.unmatched else "")
                )
            if not lora.entries:
                raise ValueError(f"LoRA {lora.name} matched no modules of this model (unmatched: {lora.unmatched[:3]})")
            self.loras.append(lora)
        return self

    def apply(self, status_fn=None):
        touched = set()
        for lora in self.loras:
            if self.mode == "merge":
                n = lora.merge(stochastic=True)
                if status_fn:
                    status_fn(f"Merged LoRA {lora.name} into {n} modules (stochastic rounding)")
            else:
                lora.attach()
            for e in lora.entries:
                touched.add(id(e.module))
        return touched

    def remove(self):
        """Hook mode: detach every hook. Merge mode: cannot un-merge; the
        engine reloads the affected components instead."""
        for lora in self.loras:
            lora.detach()
        self.loras = []

    def set_strengths(self, specs: List[dict]) -> bool:
        """Hook mode only: update strengths in place for the same paths."""
        if self.mode != "hook" or len(specs) != len(self.loras):
            return False
        for lora, spec in zip(self.loras, specs):
            if spec["path"] != lora.path:
                return False
        for lora, spec in zip(self.loras, specs):
            lora.strength = float(spec.get("strength", 1.0))
        return True
