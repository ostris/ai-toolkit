"""YuE2 instrumentation: is the training signal any good?

Training conditions the AR expert on codec tokens from the *community* audio tokenizer (the
official one is unreleased). Everything here measures how far that approximation is from what
the base model actually speaks, without running the trainer:

  score        base-model next-token CE / rank / entropy on a token stream (what ``ar_ce`` is at step 0)
  round_trip   AR tokens -> NAR render -> community tokenizer -> tokens': agreement and CE gap
  render       NAR render of real-song tokens from the community tokenizer (the quality ceiling)
  token_stats  diversity / repetition of a token stream (real vs generated vs LoRA)

``YuE2Lab`` loads the same modules the trainer does and builds AR inputs through the trainer's own
``_ar_inputs`` so the numbers match training exactly. Drive it from a notebook or script.
"""

import math
from typing import Dict, List, Optional, Sequence

import torch
from safetensors.torch import load_file

from toolkit.models.v2.resolver import resolve_named_file

from .src.model import CODEC_OFFSET, CODEC_SIZE, CONTEXT, FRAMES_PER_SECOND, YuE2Model, merge_nar_lora
from .src.pipeline import YuE2Pipeline
from .src.tokenizer import HEAD_FILE, HEAD_REPO, NAR_LORA_FILE, SHEETSAGE_FILE, SemanticTokenizer, SheetSage2Transcriber, YuE2TextTokenizer
from .src.vae import HOP, SAMPLE_RATE, YuE2VAE
from .yue2_model import (DEFAULT_CHECKPOINT, LEVEL_FIT_DEFAULT, YuE2AudioModel, apply_level_fit, level_fit_mode, load_semantic_tokenizer,
                         parse_caption, phase_tta_shifts, resolve_ai_toolkit_file, resolve_local_folder)

# saved-LoRA key prefix -> module path under YuE2Model (see convert_lora_weights_before_save)
LORA_ROOTS = {"diffusion_model.": "nar.", "text_encoders.": "ar.", "transformer.nar.": "nar.", "transformer.ar.": "ar."}
LORA_PAIRS = (("lora_A", "lora_B"), ("lora_down", "lora_up"))


def token_stats(tokens: Sequence[int], ngram: int = 4) -> Dict[str, float]:
    """Diversity / repetition of one codec stream. A healthy stream has high unigram entropy and few
    repeated n-grams; a collapsed one (same notes over and over) shows the opposite."""
    t = torch.as_tensor(list(tokens) if not torch.is_tensor(tokens) else tokens).long().flatten().cpu()
    n = t.numel()
    counts = torch.bincount(t, minlength=1).float()
    p = counts[counts > 0] / n
    grams = t.unfold(0, ngram, 1) if n >= ngram else t[None]
    runs = torch.diff(torch.nonzero(torch.cat([torch.tensor([True]), t[1:] != t[:-1], torch.tensor([True])])).flatten())
    return {
        "frames": n,
        "seconds": n / FRAMES_PER_SECOND,
        "unique": int((counts > 0).sum()),
        "unique_per_100": float((counts > 0).sum()) / n * 100,
        "unigram_entropy_bits": float(-(p * p.log2()).sum()),
        "top10_mass": float(p.sort(descending=True).values[:10].sum()),
        "repeat_prev": float((t[1:] == t[:-1]).float().mean()) if n > 1 else 0.0,
        f"repeated_{ngram}grams": 1.0 - torch.unique(grams, dim=0).shape[0] / grams.shape[0],
        "mean_run": float(runs.float().mean()),
        "max_run": int(runs.max()),
    }


def agreement(a: torch.Tensor, b: torch.Tensor, max_shift: int = 3) -> Dict[int, float]:
    """Per-frame id agreement of two streams at frame shifts -max..+max (shift s compares a[t] with
    b[t+s]). A peak away from 0 means the tokenizer is time-misaligned rather than simply wrong."""
    a, b = a.long().flatten().cpu(), b.long().flatten().cpu()
    out = {}
    for s in range(-max_shift, max_shift + 1):
        x, y = (a[: a.numel() - s], b[s:]) if s >= 0 else (a[-s:], b[: b.numel() + s])
        n = min(x.numel(), y.numel())
        out[s] = float((x[:n] == y[:n]).float().mean()) if n else 0.0
    return out


def shift_map(a: torch.Tensor, b: torch.Tensor, window: int = 125, hop: int = 25, max_shift: int = 25) -> torch.Tensor:
    """Agreement per time window and frame shift -> [windows, 2 * max_shift + 1]. A ridge that walks away
    from shift 0 over time = clock drift between the streams; uniformly low = the tokenizer is just wrong."""
    a, b = a.long().flatten().cpu(), b.long().flatten().cpu()
    n = min(a.numel(), b.numel())
    rows = []
    for s0 in range(max_shift, n - window - max_shift + 1, hop):
        ref = a[s0 : s0 + window]
        rows.append(torch.stack([(ref == b[s0 + s : s0 + s + window]).float().mean() for s in range(-max_shift, max_shift + 1)]))
    return torch.stack(rows) if rows else torch.zeros(0, 2 * max_shift + 1)


def save_run(folder: str, label: str, result: Optional[Dict] = None, **wavs: torch.Tensor) -> str:
    """Write one labelled test result: ``<folder>/<label>.json`` (every scalar, dict and 1-D tensor of
    ``result``) and ``<folder>/<label>__<name>.wav`` per waveform (``result["wav"]`` is saved as ``render``)."""
    import json
    import os

    import soundfile as sf

    os.makedirs(folder, exist_ok=True)
    label = "".join(c if c.isalnum() or c in "-_.=" else "_" for c in label)
    result = dict(result or {})
    if torch.is_tensor(result.get("wav")):
        wavs.setdefault("render", result.pop("wav"))

    def plain(v):
        if torch.is_tensor(v):
            return v.tolist() if v.dim() <= 1 else None
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        return v if isinstance(v, (int, float, str, bool, list, type(None))) else None

    with open(os.path.join(folder, f"{label}.json"), "w", encoding="utf-8") as f:
        json.dump(plain(result), f, ensure_ascii=False)
    for name, wav in wavs.items():
        sf.write(os.path.join(folder, f"{label}__{name}.wav"), wav.float().cpu().T.numpy(), SAMPLE_RATE)
    return os.path.join(folder, label)


class YuE2Lab:
    # audio-side models hold no YuE2 weights, so every lab in the process (e.g. base + merged NAR) shares them
    _shared: Dict[tuple, object] = {}

    @property
    def tag(self) -> str:
        """File-name label of this instance's NAR variant."""
        return f"nar={'merged' if self.merged_nar else 'base'}"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        head_path: str = f"{HEAD_REPO}/{HEAD_FILE}",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        merge_nar: bool = False,
        nar_lora_path: str = f"{HEAD_REPO}/{NAR_LORA_FILE}",
        cuda_graphs: bool = True,
        mert: Optional[str] = None,  # a community head's MERT-v2-FullSong folder, as the trainer's model_kwargs.mert_path (None = its default)
        phase_tta=False,  # as the trainer's model_kwargs.phase_tta (default false: one reading, as the latent cache reads)
    ):
        self.mert = mert
        self.phase_tta = phase_tta_shifts(phase_tta)
        self.device_torch = torch.device(device)
        self.torch_dtype = dtype
        sd = load_file(resolve_named_file(checkpoint, component="yue2 checkpoint"))
        self.tokenizer = YuE2TextTokenizer(sd.pop("text_encoders.yue2_tokenizer_json").numpy().tobytes())
        self.model = YuE2Model.load_from_state_dict(sd, dtype=dtype)
        if merge_nar:
            merge_nar_lora(self.model, resolve_ai_toolkit_file(nar_lora_path, component="yue2 nar lora"))
        self.merged_nar = merge_nar
        self.vae = YuE2VAE.load_from_state_dict({k[len("vae.") :]: v for k, v in sd.items() if k.startswith("vae.")}, dtype=torch.float32)
        del sd
        self.model.to(self.device_torch).eval().requires_grad_(False)
        self.vae.to(self.device_torch).eval().requires_grad_(False)
        self.pipeline = YuE2Pipeline(self.model, self.vae)
        self.pipeline.use_cuda_graphs = cuda_graphs
        self.head_path = head_path
        self.semantic: Optional[SemanticTokenizer] = None
        self.transcriber: Optional[SheetSage2Transcriber] = None
        self._lora: Optional[tuple] = None

    # ------------------------------------------------------------------ audio <-> tokens
    @staticmethod
    def load_audio(path: str, start: float = 0.0, seconds: Optional[float] = None, level_fit=LEVEL_FIT_DEFAULT) -> torch.Tensor:
        """-> [2, samples] at 48 kHz, trimmed to whole VAE hops. ``level_fit`` as the trainer's model_kwargs.level_fit (default
        LEVEL_FIT_DEFAULT, the trainer's): the WHOLE song's loudness is fitted before the excerpt is cut, as the latent cache reads it."""
        import torchaudio
        from toolkit.audio.decode import load_audio

        wav, sr = load_audio(path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav[:2] if wav.shape[0] >= 2 else wav.repeat(2, 1)
        wav = apply_level_fit(wav, level_fit_mode(level_fit))
        a = int(start * SAMPLE_RATE)
        b = wav.shape[-1] if seconds is None else a + int(seconds * SAMPLE_RATE)
        wav = wav[:, a:b]
        return wav[:, : wav.shape[-1] - wav.shape[-1] % HOP]

    def tokenize(self, wav: torch.Tensor, return_conf: bool = False):
        """Community tokenizer: [2, samples] @48k -> codec ids [T] (and the head's confidence [T])."""
        if self.semantic is None:
            key = ("semantic", self.head_path, self.mert, str(self.device_torch), self.phase_tta)
            if key not in self._shared:
                self._shared[key] = load_semantic_tokenizer(self.head_path, self.mert, phase_tta=self.phase_tta).to(self.device_torch)
            self.semantic = self._shared[key]
        return self.semantic.tokenize(wav, SAMPLE_RATE, return_conf=return_conf)

    def transcribe(self, wav: torch.Tensor, cot: str = "full") -> List[int]:
        """SheetSage2 ABC sheet ids for ``wav`` (what cot full/melody training puts before the codec tokens)."""
        if self.transcriber is None:
            key = ("sheetsage", str(self.device_torch))
            if key not in self._shared:
                path = resolve_named_file(SHEETSAGE_FILE, component="yue2 sheetsage")
                self._shared[key] = SheetSage2Transcriber(path, resolve_local_folder("sheetsage_code")).to(self.device_torch)
            self.transcriber = self._shared[key]
        return self.tokenizer.encode_abc(self.transcriber.transcribe(wav, SAMPLE_RATE, cot=cot))

    @torch.no_grad()
    def vae_roundtrip(self, wav: torch.Tensor) -> torch.Tensor:
        """VAE encode -> decode only: the best any render can sound."""
        z = self.vae.encode(wav[None].to(self.device_torch, self.vae.dtype))
        return self.vae.tiled_decode(z).float().clamp(-1, 1)[0].cpu()

    # ------------------------------------------------------------------ AR
    def prefix(self, caption: str, cot: str = "off") -> torch.Tensor:
        """Prompt head embeds [L, H] (instruction + tags + lyrics + ABC_START), as the trainer caches them."""
        parsed = parse_caption(caption)
        ids = torch.tensor([self.tokenizer.prefix_head_ids(parsed["style"], parsed["lyrics"], cot=cot)], device=self.device_torch)
        return self.model.ar.embed(ids)[0].to(self.torch_dtype)

    def inspect_text(self, text: str) -> List[str]:
        """BPE pieces of ``text``: many single-byte pieces per character = the vocabulary barely covers the script."""
        return [self.tokenizer.tokenizer.id_to_token(i) for i in self.tokenizer.encode(text)]

    @torch.no_grad()
    def score(self, caption: str, tokens, abc_ids: Sequence[int] = (), cot: str = "off", end_token: bool = True, chunk: int = 512) -> Dict:
        """Teacher-forced next-token stats of the loaded model over a codec stream, per frame:
        ``nll`` (nats; its mean is the trainer's ``ar_ce``), ``rank`` of the true id (0 = top-1) and the
        predictive ``entropy``. Inputs are built by the trainer's own ``_ar_inputs``."""
        tokens = torch.as_tensor(tokens, device=self.device_torch).long()
        abc = torch.as_tensor(list(abc_ids), dtype=torch.long, device=self.device_torch)
        embeds, ids = YuE2AudioModel._ar_inputs(self, self.prefix(caption, cot), abc, tokens, end_token=end_token)
        if embeds.shape[1] > CONTEXT:
            raise ValueError(f"{embeds.shape[1]} positions exceed the {CONTEXT} context; score a shorter excerpt")
        n = ids.shape[0]
        _, hidden = self.model.ar.prefill(embeds, return_hidden=True)
        hidden = hidden[0, -n - 1 : -1]
        nll, rank, ent = [], [], []
        for s0 in range(0, n, chunk):
            logp = torch.log_softmax(self.model.ar.model.lm_head(hidden[s0 : s0 + chunk]).float(), -1)
            tgt = logp.gather(1, ids[s0 : s0 + chunk, None])
            nll.append(-tgt[:, 0])
            rank.append((logp > tgt).sum(-1))
            ent.append(-(logp.exp() * logp).sum(-1))
        nll, rank, ent = torch.cat(nll).cpu(), torch.cat(rank).cpu(), torch.cat(ent).cpu()
        lo = abc.numel() + 2  # codec targets only: skip the sheet and [ABC_END, MUSIC_START]
        hi = lo + tokens.numel()
        nll, rank, ent = nll[lo:hi], rank[lo:hi], ent[lo:hi]
        return {
            "ce": float(nll.mean()), "ppl": math.exp(float(nll.mean())), "uniform_ce": math.log(CODEC_SIZE),
            "top1": float((rank == 0).float().mean()), "top10": float((rank < 10).float().mean()), "top100": float((rank < 100).float().mean()),
            "entropy": float(ent.mean()), "nll": nll, "rank": rank, "entropy_per_frame": ent,
        }

    @torch.no_grad()
    def generate(self, caption: str, seconds: float = 30.0, seed: int = 0, cot: str = "off", temperature: float = 1.0,
                 repetition_penalty: float = 1.2, top_p: float = 0.95, top_k: int = 100, abc_max_tokens: int = 2048) -> Dict:
        """Sample like the trainer's sampler / ComfyUI: optional ABC sheet, then codec tokens.
        -> {"tokens": [T], "abc_ids": [...], "prefix": [1, L, H] (head + sheet + ABC_END + MUSIC_START)}."""
        head = self.prefix(caption, cot)[None]
        abc_ids = self.pipeline.generate_abc_tokens(head, seed=seed, max_tokens=abc_max_tokens) if cot != "off" else []
        tail = torch.tensor([self.tokenizer.abc_tail_ids(abc_ids)], device=self.device_torch)
        prefix = torch.cat([head, self.model.ar.embed(tail).to(head.dtype)], dim=1)
        max_tokens = min(int(round(seconds * FRAMES_PER_SECOND)), CONTEXT - prefix.shape[1] - 5)
        codec = self.pipeline.generate_codec_tokens(
            prefix, max_tokens=max_tokens, seed=seed, temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, min_tokens=min(200, max_tokens), legacy_off=cot == "off",
        )
        return {"tokens": torch.tensor(codec), "abc_ids": abc_ids, "abc": self.tokenizer.decode(abc_ids), "prefix": prefix}

    @torch.no_grad()
    def render(self, tokens, caption: Optional[str] = None, prefix: Optional[torch.Tensor] = None, abc_ids: Sequence[int] = (),
               cot: str = "off", steps: int = 32, seed: int = 0) -> torch.Tensor:
        """NAR render of a codec stream -> waveform [2, samples]. Pass ``prefix`` from ``generate`` or a caption."""
        if prefix is None:
            tail = torch.tensor([self.tokenizer.abc_tail_ids(list(abc_ids))], device=self.device_torch)
            prefix = torch.cat([self.prefix(caption, cot)[None], self.model.ar.embed(tail).to(self.torch_dtype)], dim=1)
        latents = self.pipeline.synthesize(prefix, [int(t) for t in tokens], seed=seed, steps=steps)
        return self.pipeline.decode(latents)[0].cpu()

    # ------------------------------------------------------------------ the tests
    def round_trip(self, caption: str, seconds: float = 30.0, seed: int = 0, cot: str = "off", steps: int = 32) -> Dict:
        """AR tokens (ground truth: the base model wrote them) -> render -> community tokenizer.
        ``agreement[0]`` is the tokenizer's per-frame accuracy on the model's own audio; ``ce_true`` vs
        ``ce_retok`` is how much less the base model believes the tokenizer's version of the same song."""
        gen = self.generate(caption, seconds, seed, cot)
        wav = self.render(gen["tokens"], prefix=gen["prefix"], steps=steps, seed=seed)
        retok, conf = self.tokenize(wav, return_conf=True)
        true = gen["tokens"]
        n = min(true.numel(), retok.numel())
        true, retok, conf = true[:n], retok[:n].cpu(), conf[:n].cpu()
        hit = true == retok
        s_true = self.score(caption, true, gen["abc_ids"], cot, end_token=False)
        s_retok = self.score(caption, retok, gen["abc_ids"], cot, end_token=False)
        return {
            "wav": wav, "true": true, "retok": retok, "conf": conf, "hit": hit, "abc": gen["abc"],
            "agreement": agreement(true, retok),
            "conf_when_right": float(conf[hit].mean()) if hit.any() else float("nan"),
            "conf_when_wrong": float(conf[~hit].mean()) if (~hit).any() else float("nan"),
            "ce_true": s_true["ce"], "ce_retok": s_retok["ce"], "score_true": s_true, "score_retok": s_retok,
            "stats_true": token_stats(true), "stats_retok": token_stats(retok),
        }

    def real_song(self, wav: torch.Tensor, caption: str, cot: str = "off", steps: int = 32, seed: int = 0, render: bool = True,
                  reuse: Optional[Dict] = None) -> Dict:
        """A real training song through the training path: community tokens, their step-0 CE, and
        (``render``) what the NAR makes of them -- if this render is already mangled, that is the
        ceiling no AR training can beat. ``self_agreement`` re-tokenizes the render. ``reuse`` = the
        result of another lab for the same song: everything that does not depend on the NAR (tokens,
        score, VAE reference) is taken from it, so a base/merged pair only renders twice."""
        ar_keys = ("tokens", "conf", "abc_ids", "mean_conf", "score", "stats", "mel_l1_vae")
        if reuse is not None:
            out = {k: reuse[k] for k in ar_keys if k in reuse}
            tokens, abc_ids = out["tokens"], out["abc_ids"]
        else:
            tokens, conf = self.tokenize(wav, return_conf=True)
            tokens, conf = tokens.cpu(), conf.cpu()
            abc_ids = self.transcribe(wav, cot) if cot != "off" else []
            out = {"tokens": tokens, "conf": conf, "abc_ids": abc_ids, "mean_conf": float(conf.mean()),
                   "score": self.score(caption, tokens, abc_ids, cot), "stats": token_stats(tokens)}
        if render:
            out["wav"] = self.render(tokens, caption=caption, abc_ids=abc_ids, cot=cot, steps=steps, seed=seed)
            out["self_agreement"] = agreement(tokens, self.tokenize(out["wav"]).cpu())
            out["mel_l1"] = self.mel_distance(wav, out["wav"])
            if "mel_l1_vae" not in out:
                out["mel_l1_vae"] = self.mel_distance(wav, self.vae_roundtrip(wav))
        return out

    def survey(self, folder: str, start: float = 0.0, seconds: Optional[float] = 60.0, cot: str = "off", render: bool = False,
               steps: int = 32, extensions: Sequence[str] = (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".opus"), save_dir: Optional[str] = None) -> List[Dict]:
        """``real_song`` over every audio file in ``folder`` (caption = the ``.txt`` next to it, as in an
        ai-toolkit dataset) -> one flat row per song, ready for a DataFrame. ``render`` adds the NAR
        render metrics (slow) and keeps the waveform under ``row["wav"]``. ``save_dir`` writes each song's
        full result (and render) through ``save_run``, labelled with the song name."""
        import os

        rows = []
        for name in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in extensions:
                continue
            txt = os.path.join(folder, stem + ".txt")
            caption = open(txt, encoding="utf-8").read() if os.path.exists(txt) else ""
            try:
                r = self.real_song(self.load_audio(os.path.join(folder, name), start, seconds), caption, cot, steps, render=render)
            except Exception as e:  # one bad file must not end a long survey
                rows.append({"song": name, "error": f"{type(e).__name__}: {e}"})
                continue
            if save_dir:
                save_run(save_dir, f"{stem}__{self.tag}_cot={cot}", r)
            row = {"song": name, "captioned": bool(caption), "ce": r["score"]["ce"], "top1": r["score"]["top1"], "top10": r["score"]["top10"],
                   "top100": r["score"]["top100"], "entropy": r["score"]["entropy"], "head_conf": r["mean_conf"], **r["stats"]}
            if render:
                row.update(self_agreement=r["self_agreement"][0], mel_l1=r["mel_l1"], mel_l1_vae=r["mel_l1_vae"], wav=r["wav"])
            rows.append(row)
        return rows

    @staticmethod
    def mel_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        """Mean |log-mel| difference of two waveforms (lower = closer). Compare against ``mel_l1_vae``."""
        import torchaudio

        mel = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, n_fft=2048, hop_length=HOP // 4, n_mels=128)
        n = min(a.shape[-1], b.shape[-1])
        la, lb = (mel(x[..., :n].float().mean(0)).clamp_min(1e-5).log() for x in (a, b))
        return float((la - lb).abs().mean())

    # ------------------------------------------------------------------ trained LoRA on / off
    @torch.no_grad()
    def set_lora(self, path: Optional[str], scale: float = 1.0) -> int:
        """Fold a saved ai-toolkit LoRA into the weights (``None`` removes it). Needs the bf16 checkpoint."""
        if getattr(self.model, "aitk_is_quantized", False):
            raise ValueError("LoRA merging needs the bf16 checkpoint; the int8 repack cannot take merged weights")
        if self._lora is not None:
            old_path, old_scale = self._lora
            self._lora = None
            self._fold(old_path, -old_scale)
        if path is None:
            return 0
        n = self._fold(path, scale)
        self._lora = (path, scale)
        return n

    def _fold(self, path: str, scale: float) -> int:
        sd = load_file(path)
        modules = dict(self.model.named_modules())
        n = 0
        for key in sd:
            for down, up in LORA_PAIRS:
                if not key.endswith(f".{down}.weight"):
                    continue
                base = key[: -len(f".{down}.weight")]
                root = next((r for r in LORA_ROOTS if base.startswith(r)), None)
                if root is None:
                    raise KeyError(f"unrecognised LoRA key layout: {key}")
                A, B = sd[key].float(), sd[f"{base}.{up}.weight"].float()
                alpha = sd.get(f"{base}.alpha")
                w = modules[LORA_ROOTS[root] + base[len(root) :]].weight
                delta = (B @ A) * (scale * (float(alpha) / A.shape[0] if alpha is not None else 1.0))
                w.add_(delta.to(w.device, w.dtype))
                n += 1
        if n == 0:
            raise KeyError(f"no LoRA pairs found in {path}")
        return n
