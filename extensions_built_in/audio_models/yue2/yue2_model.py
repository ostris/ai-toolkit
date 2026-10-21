"""YuE2 (m-a-p/YuE2-3B) for ai-toolkit.

Weights: the Comfy-Org all-in-one checkpoint (default ``checkpoints/yue2_3b_int8_convrot.safetensors``,
shipped int8 convrot layers attached as-is; ``yue2_3b_bf16.safetensors`` also loads)
resolved under MODELS_PATH in ComfyUI's folder layout and downloaded there when missing.
The official audio -> semantic-token tokenizer is unreleased; training conditioning comes
from the community head (MERT-v2-FullSong + classifier) in
``Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4``. Its companion NAR adapter can
be merged into the base NAR on load (``model_kwargs.merge_nar_lora``, default false).
The default is the v4 pair; the repo's later pairs (v5 / v8 / v9, safetensors only) load through
``model_kwargs.semantic_head_path`` / ``nar_lora_path``, e.g.
``Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/tokenizer_head_joint_v9.safetensors``
(or a local file). A non-default head gets its own latent cache key, since the cache holds its tokens.

Training: per song the latent cache carries VAE latents plus the head's codec tokens; the
"text embedding" is the AR prompt prefix (instruction + tags + lyrics) as token embeddings.
Each step crops one window (``model_kwargs.train_window_frames``, 25 fps, 0 = whole song),
prefills the AR expert over ``prefix + tokens[window] + MUSIC_END`` and trains the NAR
flow loss on the real latents. With ``model_kwargs.ar_loss_weight`` > 0 the same prefill
also trains the AR expert with next-token cross entropy over the codec tokens, so one LoRA
network on both experts learns composition and rendering together.

With ``model_kwargs.do_separation`` MelBandRoformer (toolkit/audio/melbandroformer) splits every song into
vocals and instrumental at cache time; their codec tokens (and sheets) ride the latent cache next to the
mix, and the prompt cache holds three prefixes: the full prompt, lyrics only and tags only. Every step
still trains the NAR flow loss on the mix window; the AR next-token loss becomes a weighted mean of three
terms, (full prompt, mix), (lyrics only, vocals) and (tags only, instrumental), each from the song start,
with ``separation_vocals_weight`` (0.5) and ``separation_music_weight`` (1.0) against the mix's 1. Both
cache versions get a ``_sep`` suffix so existing caches are rebuilt.

Prompt modes follow ComfyUI (``model_kwargs.cot``): "full" (default, Generate ABC path: the AR
writes a chord-annotated ABC sheet, then the codec tokens), "melody" (sheet without chords) or
"off" (no sheet; what Comfy runs when the ABC input is empty). For full/melody the training sheet
comes from SheetSage2 at cache time and rides the latent cache as ``abc_ids``.
"""

import os
import random
import re
from typing import List, Optional

import torch
import torch.utils.checkpoint
from safetensors.torch import load_file
from tqdm import tqdm

from extensions_built_in.audio_models.base_audio_model import BaseAudioModel
from toolkit.audio.melbandroformer import load_melbandroformer, separate as separate_vocals
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig
from toolkit.dto import DTO
from toolkit.models.v2.resolver import resolve_named_file
from toolkit.paths import MODELS_PATH
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

from .src.model import ABC_END, CODEC_OFFSET, CONTEXT, FRAMES_PER_SECOND, INSTRUCTIONS, MUSIC_END, MUSIC_START, YuE2Model, merge_nar_lora
from .src.pipeline import YuE2Pipeline
from .src.tokenizer import HEAD_FILE, HEAD_REPO, NAR_LORA_FILE, SHEETSAGE_FILE, SemanticTokenizer, SheetSage2Transcriber, YuE2TextTokenizer
from .src.vae import SAMPLE_RATE, YuE2VAE

COT_CODES = {"off": 0, "melody": 1, "full": 2}
COT_NAMES = {v: k for k, v in COT_CODES.items()}

# do_separation trains the AR on three (prompt, stem) pairs; prompt-mask segment ids follow this order (1, 2, 3)
SEP_VARIANTS = ("full", "vocals", "music")


def _variant_prompt(parsed: dict, variant: str):
    """(style, lyrics) the AR sees per stem: the vocal stem gets just the lyrics, the music stem just the tags."""
    if variant == "vocals":
        return "", parsed["lyrics"]
    if variant == "music":
        return parsed["style"], ""
    return parsed["style"], parsed["lyrics"]

DEFAULT_CHECKPOINT = "Comfy-Org/YuE2/checkpoints/yue2_3b_int8_convrot.safetensors"
# community tokenizer files have no ComfyUI folder; they live under <MODELS_PATH>/ai_toolkit/
AI_TOOLKIT_DIR = "ai_toolkit"


def resolve_ai_toolkit_file(path: str, component: str) -> str:
    """Local file, or 'org/repo/file' downloaded into <MODELS_PATH>/ai_toolkit/<file>."""
    if os.path.exists(path):
        return path
    splits = path.split("/")
    if len(splits) < 3:
        raise ValueError(f"Invalid {component} path: {path}. Must be a local file or 'org/repo/filename'.")
    local_dir = os.path.join(MODELS_PATH, AI_TOOLKIT_DIR)
    candidate = os.path.join(local_dir, splits[-1])
    if os.path.exists(candidate):
        return candidate
    import huggingface_hub

    return huggingface_hub.hf_hub_download(repo_id="/".join(splits[:2]), filename="/".join(splits[2:]), local_dir=local_dir)

scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": False,
}


def _tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


_SECTION = re.compile(r"^\s*\[(Tags|Lyrics|Duration)\]\s*$", re.IGNORECASE | re.MULTILINE)
_SONG_SECTION = re.compile(r"^\s*\[[^\]]+\]\s*$")


def _normalize_section(line: str) -> str:
    """YuE2 section tags are Title case ([Verse 1], [Pre-Chorus]); uppercase the first letter and the
    letter after each hyphen of a bracketed section line, leaving the rest (e.g. descriptors) alone."""
    if not _SONG_SECTION.match(line):
        return line
    lead, body, trail = line[: line.index("[") + 1], line[line.index("[") + 1 : line.rindex("]")], line[line.rindex("]") :]
    body = " ".join(w.capitalize() if w.isupper() and len(w) > 1 else w for w in body.split(" "))  # [BRIDGE] -> [Bridge]
    body = body[:1].upper() + body[1:]
    body = re.sub(r"-([a-z])", lambda m: "-" + m.group(1).upper(), body)
    return lead + body + trail


def _number(value: str):
    try:
        return float(value) if value.strip() else None
    except ValueError:
        return None


def parse_caption(text: str) -> dict:
    """Caption / prompt -> style, lyrics, duration.

    Native form (YuE2's own prompt layout): style text, then a ``[Lyrics]`` line and the
    lyrics; an optional leading ``[Tags]`` line and an optional trailing ``[Duration]``
    line (max seconds, samples only). Plain text with no sections is the style.
    Legacy ``<CAPTION>``/``<LYRICS>``/``<DURATION>`` tags are still accepted."""
    if not isinstance(text, str):
        text = ""
    if "<CAPTION>" in text or "<LYRICS>" in text:
        lyrics = "\n".join(_normalize_section(l) for l in _tag(text, "LYRICS").splitlines())
        return {"style": _tag(text, "CAPTION"), "lyrics": lyrics, "duration": _number(_tag(text, "DURATION"))}
    sections = {"tags": []}
    current = "tags"
    for line in text.splitlines():
        m = _SECTION.match(line)
        if m:
            current = m.group(1).lower()
            sections.setdefault(current, [])
            continue
        # no [Lyrics] header: the first song-section line ([Intro], [Verse 1], ...) starts the lyrics
        if current == "tags" and "lyrics" not in sections and _SONG_SECTION.match(line):
            current = "lyrics"
            sections["lyrics"] = []
        sections.setdefault(current, []).append(_normalize_section(line) if current == "lyrics" else line)
    style = "\n".join(sections["tags"]).strip()
    lyrics = "\n".join(sections.get("lyrics", [])).strip()
    duration = _number("\n".join(sections.get("duration", [])))
    return {"style": style, "lyrics": lyrics, "duration": duration}


class YuE2TextEncoder(torch.nn.Module):
    """Prompt -> AR token embeddings of the prefix head (instruction, tags, lyrics, ABC_START).
    Holds no weights of its own; the embedding table lives in the AR expert."""

    def __init__(self, model: YuE2Model, tokenizer: YuE2TextTokenizer, cot: str = "full"):
        super().__init__()
        self._model = [model]
        self.tokenizer = tokenizer
        self.cot = cot
        self.register_buffer("_anchor", torch.zeros(1), persistent=False)

    @property
    def device(self):
        return self._anchor.device

    @property
    def dtype(self):
        return self._model[0].ar.dtype

    @torch.no_grad()
    def forward(self, style: str, lyrics: str) -> torch.Tensor:
        ids = torch.tensor([self.tokenizer.prefix_head_ids(style, lyrics, cot=self.cot)], dtype=torch.long)
        return self._model[0].ar.embed(ids)


class YuE2AudioModel(BaseAudioModel):
    arch = "yue2"
    sample_rate = SAMPLE_RATE

    def __init__(self, device, model_config, dtype="bf16", custom_pipeline=None, noise_scheduler=None, **kwargs):
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["YuE2AR", "YuE2NAR"]
        kw = self.model_config.model_kwargs
        self.train_window_frames = int(kw.get("train_window_frames", 1500))
        self.ar_loss_weight = float(kw.get("ar_loss_weight", 1.0))
        # AR next-token loss always runs on the song from its start (lyric order); 0 = whole song
        self.ar_max_tokens = int(kw.get("ar_max_tokens", 0))
        # optional separate learning rate for the AR expert's LoRA (Adam ignores loss scale)
        self.ar_lr_multiplier = float(kw.get("ar_lr_multiplier", 1.0))
        # trust region: KL(base || lora) on the AR next-token distributions, base = LoRA switched off
        self.ar_kl_weight = float(kw.get("ar_kl_weight", 0.0))
        self._loss_log_every = int(kw.get("loss_log_every", 25))
        # extra per-readout AR diagnostics (targets, unique tokens, p(END), p(target))
        self.debug = bool(kw.get("debug", False))
        self._loss_log_step = 0
        self.merge_nar_lora = bool(kw.get("merge_nar_lora", False))
        self.nar_lora_path = kw.get("nar_lora_path", f"{HEAD_REPO}/{NAR_LORA_FILE}")
        self.semantic_head_path = kw.get("semantic_head_path", f"{HEAD_REPO}/{HEAD_FILE}")
        # ComfyUI's Generate ABC path: "full" (chords) or "melody"; "off" skips the sheet (empty ABC in Comfy)
        self.cot = str(kw.get("cot", "full"))
        if self.cot not in INSTRUCTIONS:
            raise ValueError(f"model_kwargs.cot must be one of {list(INSTRUCTIONS)}")
        self.abc_max_tokens = int(kw.get("abc_max_tokens", 8192))
        # fraction of training items fed without the sheet (off-mode prefix), so one LoRA serves both Comfy paths
        self.abc_dropout = float(kw.get("abc_dropout", 0.5))
        self.sheetsage_path = kw.get("sheetsage_path", SHEETSAGE_FILE)
        self.transcriber: Optional[SheetSage2Transcriber] = None
        # do_separation: MelBandRoformer stems at cache time; the AR loss covers mix + vocals + instrumental every step
        self.do_separation = bool(kw.get("do_separation", False))
        # stem term weights in the AR loss (the mix term is 1); the vocal stem alone pulls samples toward lyrics-only output
        self.separation_weights = {
            "vocals": float(kw.get("separation_vocals_weight", 0.5)),
            "music": float(kw.get("separation_music_weight", 1.0)),
        }
        self.separator = None
        # encoders are only needed while caching; once training steps run without encode calls they are released
        self._train_calls = 0
        self._last_encode_call = 0
        self.sample_max_seconds = float(kw.get("sample_max_seconds", 120.0))
        self.sample_ar_temperature = float(kw.get("sample_ar_temperature", 1.0))
        # reference sampler value; 1.0 lets a memorized song replay (the penalty knocks it off path)
        self.sample_ar_repetition_penalty = float(kw.get("sample_ar_repetition_penalty", 1.1))
        self.semantic_tokenizer: Optional[SemanticTokenizer] = None
        self._pending_aux_loss = None
        self._pending_ar_ce = None
        self._pending_ar_kl = None
        self._pending_stem_ce = {}
        self.additional_loss_logs = {}

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load_model(self):
        dtype = self.torch_dtype
        device = self.device_torch
        name_or_path = self.model_config.name_or_path or DEFAULT_CHECKPOINT
        ckpt_path = resolve_named_file(name_or_path, component="yue2 checkpoint")
        self.print_and_status_update(f"Loading YuE2 from {ckpt_path}")
        sd = load_file(ckpt_path)
        tok_json = sd.pop("text_encoders.yue2_tokenizer_json", None)
        if tok_json is None:
            raise ValueError("YuE2 checkpoint has no embedded tokenizer (expected the Comfy-Org repack)")
        self.tokenizer = YuE2TextTokenizer(tok_json.numpy().tobytes())

        if self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0:
            raise NotImplementedError("Layer offloading not yet implemented for YuE2")

        self.model = YuE2Model.load_from_state_dict(sd, dtype=dtype)
        if self.merge_nar_lora:
            lora_path = resolve_ai_toolkit_file(self.nar_lora_path, component="yue2 nar lora")
            self.print_and_status_update(f"Merging NAR adapter {os.path.basename(lora_path)}")
            merge_nar_lora(self.model, lora_path)
        vae_sd = {k[len("vae.") :]: v for k, v in sd.items() if k.startswith("vae.")}
        # reference runs the VAE in fp32; keep the trainer's dtype check from downcasting it
        self.vae_torch_dtype = torch.float32
        self.vae = YuE2VAE.load_from_state_dict(vae_sd, dtype=torch.float32)
        del sd
        flush()

        self.model.aitk_post_load(**self.component_load_kwargs("transformer"))
        flush()
        self.model.to(device)
        self.vae.to(self.vae_device_torch)
        self.text_encoder = YuE2TextEncoder(self.model, self.tokenizer, cot=self.cot)
        self.pipeline = YuE2Pipeline(self.model, self.vae)
        self.pipeline.do_tiled_decoding = True
        self.pipeline.use_cuda_graphs = bool(self.model_config.model_kwargs.get("ar_cuda_graphs", True))

    def _get_semantic_tokenizer(self) -> SemanticTokenizer:
        if self.semantic_tokenizer is None:
            head_path = resolve_ai_toolkit_file(self.semantic_head_path, component="yue2 semantic head")
            # log only: this can run mid-training and a status update would replace "Training" in the UI
            print_acc("Loading YuE2 semantic tokenizer (MERT-v2-FullSong + head)")
            self.semantic_tokenizer = SemanticTokenizer(head_path, debug=self.debug)
        return self.semantic_tokenizer

    def _get_transcriber(self) -> SheetSage2Transcriber:
        if self.transcriber is None:
            print_acc("Loading SheetSage2 (audio -> ABC)")
            self.transcriber = SheetSage2Transcriber(resolve_named_file(self.sheetsage_path, component="sheetsage2"))
        return self.transcriber

    def _get_separator(self):
        if self.separator is None:
            print_acc("Loading MelBandRoformer (vocals / instrumental separation)")
            self.separator = load_melbandroformer(device=self.device_torch)
        return self.separator

    def get_latent_space_version(self):
        version = super().get_latent_space_version()
        # the cached codec tokens come from the semantic head: a non-default head needs its own cache
        # (the default keeps the plain key so existing caches stay valid)
        if self.semantic_head_path != f"{HEAD_REPO}/{HEAD_FILE}":
            head_name = os.path.splitext(os.path.basename(self.semantic_head_path))[0]
            version = f"{version}_{re.sub(r'[^A-Za-z0-9]+', '-', head_name)}"
        # stem tokens/sheets only ride the latent cache in separation mode
        return f"{version}_sep" if self.do_separation else version

    def get_text_embedding_space_version(self):
        version = super().get_text_embedding_space_version()
        # separation mode caches three prefixes per prompt
        return f"{version}_sep" if self.do_separation else version

    def pop_encode_warnings(self) -> List[str]:
        """Repair notes from the last encode_audio call; the latent cacher prints them with the file path."""
        if self.transcriber is None:
            return []
        warnings, self.transcriber.warnings = self.transcriber.warnings, []
        return warnings

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["model.layers"]

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # conditioning
    # ------------------------------------------------------------------
    def get_prompt_embeds(self, prompt) -> PromptEmbeds:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        embeds, masks = [], []
        for p in prompts:
            parsed = parse_caption(p)
            if self.do_separation:
                # full, lyrics-only and tags-only prefixes back to back; the mask holds the segment id (1, 2, 3), 0 = pad
                segments = [self.text_encoder(*_variant_prompt(parsed, v)) for v in SEP_VARIANTS]
                e = torch.cat(segments, 1)  # [1, L1+L2+L3, H]
                m = torch.cat([torch.full((1, seg.shape[1]), k + 1, dtype=torch.long, device=e.device) for k, seg in enumerate(segments)], 1)
            else:
                e = self.text_encoder(parsed["style"], parsed["lyrics"])  # [1, L, H]
                m = torch.ones(1, e.shape[1], dtype=torch.long, device=e.device)
            embeds.append(e)
            masks.append(m)
        max_len = max(e.shape[1] for e in embeds)
        embeds = [torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[1])) for e in embeds]
        masks = [torch.nn.functional.pad(m, (0, max_len - m.shape[1])) for m in masks]
        return PromptEmbeds(torch.cat(embeds, 0), attention_mask=torch.cat(masks, 0))

    def _transcribe_sheets(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        """SheetSage2 sheets for [C, samples] waveforms -> text ids [B, N], -1 padded."""
        transcriber = self._get_transcriber()
        if transcriber.device != self.device_torch:
            transcriber.to(self.device_torch)
        sheets = [self.tokenizer.encode_abc(transcriber.transcribe(wav, SAMPLE_RATE, cot=self.cot)) for wav in wavs]
        width = max(len(x) for x in sheets)
        abc = torch.full((len(sheets), width), -1, dtype=torch.int32)
        for i, ids in enumerate(sheets):
            abc[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        return abc

    def encode_audio(self, audio_tensor: torch.Tensor, device=None, dtype=None):
        """[B, 2, samples] at 48 kHz -> DTO [B, T, 64] with codec ``tokens`` [B, T] and, unless cot is
        "off", the SheetSage2 sheet as ``abc_ids`` [B, N] (text ids, -1 padded). With do_separation the
        MelBandRoformer stems add ``tokens_vocals`` / ``tokens_music`` and their ``abc_ids_*`` sheets."""
        if device is None:
            device = self.vae_device_torch
        self._last_encode_call = self._train_calls
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        latents = self.vae.encode(audio_tensor.to(device=device, dtype=self.vae.dtype))  # [B, 64, T]
        latents = latents.transpose(1, 2).contiguous()
        tokenizer = self._get_semantic_tokenizer()
        if tokenizer.device != self.device_torch:
            tokenizer.to(self.device_torch)
        tokens = torch.stack([tokenizer.tokenize(wav, SAMPLE_RATE) for wav in audio_tensor])  # [B, T']
        n = min(latents.shape[1], tokens.shape[1])
        latents = latents[:, :n].to(dtype=self.torch_dtype if dtype is None else dtype)
        extras = {"tokens": tokens[:, :n].to(latents.device, torch.int32)}
        if self.cot != "off":
            extras["abc_ids"] = self._transcribe_sheets(list(audio_tensor)).to(latents.device)
            # which sheet flavor the cache holds; training refuses a mismatch instead of silently using it
            extras["abc_mode"] = torch.full((audio_tensor.shape[0],), COT_CODES[self.cot], dtype=torch.int32, device=latents.device)
        if self.do_separation:
            separator = self._get_separator()
            stems = [separate_vocals(separator, wav.float(), SAMPLE_RATE) for wav in audio_tensor]  # (vocals, instrumental)
            for idx, name in ((0, "vocals"), (1, "music")):
                wavs = [stem[idx] for stem in stems]
                stem_tokens = torch.stack([tokenizer.tokenize(wav, SAMPLE_RATE) for wav in wavs])
                extras[f"tokens_{name}"] = stem_tokens[:, :n].to(latents.device, torch.int32)
                if self.cot != "off":
                    extras[f"abc_ids_{name}"] = self._transcribe_sheets(wavs).to(latents.device)
        return DTO(latents, **extras)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def condition_noisy_latents(self, latents: torch.Tensor, batch):
        frames = latents.shape[1]
        window = self.train_window_frames
        if window <= 0 or frames <= window:
            batch.yue2_window = (0, frames)
            return latents
        start = random.randint(0, frames - window)
        batch.yue2_window = (start, start + window)
        return latents[:, start : start + window]

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        start, end = getattr(batch, "yue2_window", (0, noise.shape[1]))
        return (noise - batch.latents)[:, start:end].detach()

    def get_additional_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """AR next-token CE from the last forward, added to the flow loss by the trainer and
        logged as its own term (`loss/ar_ce`, unweighted)."""
        aux = self._pending_aux_loss
        self._pending_aux_loss = None
        if aux is None:
            self.additional_loss_logs = {}
            return None
        ce = self._pending_ar_ce.detach().float().item()
        self.additional_loss_logs = {"loss/ar_ce": ce}
        for variant, stem_ce in self._pending_stem_ce.items():
            self.additional_loss_logs[f"loss/ar_ce_{variant}"] = stem_ce.detach().float().item()
        kl = self._pending_ar_kl
        if kl is not None:
            self.additional_loss_logs["loss/ar_kl"] = kl.detach().float().item()
        self._loss_log_step += 1
        if self.debug and self._loss_log_every > 0 and self._loss_log_step % self._loss_log_every == 0:
            info = getattr(self, "_pending_aux_info", "")
            kl_txt = f"  ar_kl {kl.detach().float().item():.4f}" if kl is not None else ""
            tqdm.write(f"yue2 step {self._loss_log_step}: ar_ce {ce:.4f}{kl_txt}" + (f"  {info}" if info else ""))
        return aux

    @staticmethod
    def _prefix_segment(prefix_embeds: torch.Tensor, prefix_mask: torch.Tensor, segment: int = 1) -> torch.Tensor:
        """Unpadded prefix embeds for one item. The mask holds segment ids: all 1 normally; with do_separation
        1 = full prompt, 2 = lyrics only, 3 = tags only (SEP_VARIANTS order); 0 = padding."""
        # the mask may arrive cast to bf16; compare rounded values, never sum them
        return prefix_embeds[prefix_mask.float().round() == segment]

    def _check_abc_cache(self, latents: DTO):
        if self.cot == "off":
            return
        mode = latents.get("abc_mode")
        cached = None if mode is None else int(mode.reshape(-1)[0].item())
        if latents.get("abc_ids") is None or cached != COT_CODES[self.cot]:
            raise ValueError(
                f"YuE2 cot={self.cot!r} needs a matching ABC sheet in the latent cache "
                f"(cache has {'none' if cached is None else COT_NAMES.get(cached, cached)!r}): "
                "delete the dataset's _latent_cache folder so it is rebuilt"
            )

    def _note_train_call(self):
        self._train_calls += 1
        if self._train_calls - self._last_encode_call > 2 and (
            self.semantic_tokenizer is not None or self.transcriber is not None or self.separator is not None
        ):
            # latents are cached: MERT + SheetSage2 (~4 GB) and the separator are dead weight for the rest of training
            self.semantic_tokenizer = None
            self.transcriber = None
            self.separator = None
            flush()

    def _item_prefix_and_abc(self, prefix: torch.Tensor, abc_all: Optional[torch.Tensor], i: int, caption: Optional[str], variant: str = "full"):
        """(prefix, sheet ids) for one item; the sheet is empty when cot is off. With probability ``abc_dropout``
        at train time the sheet is dropped and the item trains exactly as an off-mode prompt (off instruction,
        empty ABC block) rebuilt from the caption."""
        device = self.device_torch
        if self.cot == "off" or abc_all is None:
            return prefix, torch.zeros(0, dtype=torch.long, device=device)
        if torch.is_grad_enabled() and caption is not None and random.random() < self.abc_dropout:
            style, lyrics = _variant_prompt(parse_caption(caption), variant)
            ids = torch.tensor([self.tokenizer.prefix_head_ids(style, lyrics, cot="off")], device=device)
            return self.model.ar.embed(ids)[0].to(self.torch_dtype), torch.zeros(0, dtype=torch.long, device=device)
        row = abc_all[i].to(device)
        return prefix, row[row >= 0].long()

    def _stem_ar_loss(self, latents: DTO, i: int, variant: str, segment: int, prefix_embeds: torch.Tensor, prefix_mask: torch.Tensor, caption: Optional[str]):
        """do_separation: next-token CE (and KL) of one stem from the song start, prompted by its own prefix segment."""
        song = latents.get(f"tokens_{variant}")
        if song is None:
            raise ValueError("YuE2 do_separation needs stem tokens in the latent cache: delete the dataset's _latent_cache folder so it is rebuilt")
        song = song[i].to(self.device_torch)
        total = song.shape[0]
        prefix = self._prefix_segment(prefix_embeds, prefix_mask, segment)
        prefix, abc = self._item_prefix_and_abc(prefix, latents.get(f"abc_ids_{variant}"), i, caption, variant)
        limit = total if self.ar_max_tokens <= 0 else min(total, self.ar_max_tokens)
        embeds, ids = self._ar_inputs(prefix, abc, song[:limit], end_token=limit == total)
        ce, kl, _ = self._ar_losses(embeds, ids, prefix, total, limit == total)
        return ce, kl

    def _ar_losses(self, ar_embeds: torch.Tensor, ids: torch.Tensor, prefix: torch.Tensor, total: int, whole_song: bool):
        """Next-token CE over ``ids`` (and KL(base || lora) when ``ar_kl_weight`` > 0), computed in
        position chunks with recomputed logits so the fp32 [n, vocab] tensors never exist at once.
        Returns (ce, kl_or_None, info_str)."""
        model = self.model
        n = ids.shape[0]
        _, hidden = model.ar.prefill(ar_embeds, return_hidden=True)
        hidden = hidden[0, -n - 1 : -1]
        base_hidden = None
        if self.ar_kl_weight > 0:
            net = getattr(self, "_network", None)
            was_active = getattr(net, "is_active", None)
            if net is not None:
                net.is_active = False
            try:
                with torch.no_grad():
                    _, base_hidden = model.ar.prefill(ar_embeds, return_hidden=True)
                    base_hidden = base_hidden[0, -n - 1 : -1]
            finally:
                if net is not None:
                    net.is_active = was_active
        want_info = self.debug and self._loss_log_every > 0 and (self._loss_log_step + 1) % self._loss_log_every == 0
        lm_head = model.ar.model.lm_head
        want_kl = base_hidden is not None

        def chunk_losses(h, tgt, bh):
            # everything vocab-sized lives inside this checkpointed function, so autograd keeps only
            # the [chunk, hidden] inputs and recomputes the fp32 logits/log-probs in backward
            logits = lm_head(h).float()
            ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
            if bh is not None:
                with torch.no_grad():
                    base_logp = torch.log_softmax(lm_head(bh).float(), -1)
                kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), base_logp, log_target=True, reduction="sum")
            else:
                kl = ce.new_zeros(())
            if want_info:
                with torch.no_grad():
                    p = logits.softmax(-1)
                    p_first_end = p[0, MUSIC_END]
                    p_tgt = p.gather(1, tgt[:, None]).squeeze(1)
            else:
                p_first_end, p_tgt = ce.new_zeros(()), ce.new_zeros((0,))
            return ce, kl, p_first_end, p_tgt

        ce_sum = hidden.new_zeros((), dtype=torch.float32)
        kl_sum = hidden.new_zeros((), dtype=torch.float32)
        p_target, p_end_first = [], None
        chunk = 512  # vocab-sized fp32 transients scale with this (~1.9 GB per 512 positions incl. KL)
        for s0 in range(0, n, chunk):
            h = hidden[s0 : s0 + chunk]
            tgt = ids[s0 : s0 + chunk]
            bh = base_hidden[s0 : s0 + chunk] if want_kl else None
            if torch.is_grad_enabled():
                ce, kl, p_first, p_tgt = torch.utils.checkpoint.checkpoint(chunk_losses, h, tgt, bh, use_reentrant=False)
            else:
                ce, kl, p_first, p_tgt = chunk_losses(h, tgt, bh)
            ce_sum = ce_sum + ce
            kl_sum = kl_sum + kl
            if want_info:
                if s0 == 0:
                    p_end_first = p_first.item()
                p_target.append(p_tgt)
        info = ""
        if want_info:
            p_target = torch.cat(p_target)
            info = (f"[ar targets {n} unique {ids.unique().numel()} (song {total}, whole {whole_song}) "
                    f"prefix {prefix.shape[0]} p(END|prefix) {p_end_first:.3f} p(target) median {p_target.median().item():.3f}]")
        return ce_sum / n, (kl_sum / n if want_kl else None), info

    def _ar_inputs(self, prefix: torch.Tensor, abc_ids: torch.Tensor, tokens: torch.Tensor, end_token: bool = True):
        """One item: prefix head (unpadded [L, H]) + abc + [ABC_END, MUSIC_START] + codec tokens [+ MUSIC_END]
        -> embeds [1, L', H] and the ids after the head [n] (the AR targets)."""
        device = tokens.device
        parts = [abc_ids.long().to(device), torch.tensor([ABC_END, MUSIC_START], device=device), tokens.long() + CODEC_OFFSET]
        if end_token:
            parts.append(torch.tensor([MUSIC_END], device=device))
        ids = torch.cat(parts)
        embeds = torch.cat([prefix.to(self.model.ar.dtype), self.model.ar.embed(ids)], dim=0)[None]
        return embeds, ids

    # ------------------------------------------------------------------
    # per-expert learning rate: the trainer hands us the LoRA network before it
    # builds optimizer groups, so split the AR expert's modules into their own group
    # ------------------------------------------------------------------
    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value
        mult = getattr(self, "ar_lr_multiplier", 1.0)
        if value is None or mult == 1.0 or getattr(value, "_yue2_lr_split", False):
            return
        orig = value.prepare_optimizer_params

        def prepare_optimizer_params(text_encoder_lr=None, unet_lr=None, default_lr=None):
            groups = orig(text_encoder_lr, unet_lr, default_lr)
            ar_ids = set()
            for lora in getattr(value, "unet_loras", []):
                if ".ar." in lora.lora_name.replace("$$", "."):
                    ar_ids.update(id(p) for p in lora.parameters())
            out = []
            n_ar = 0
            for group in groups:
                base_lr = group.get("lr", default_lr)
                ar = [p for p in group["params"] if id(p) in ar_ids]
                rest = [p for p in group["params"] if id(p) not in ar_ids]
                if rest:
                    out.append({**group, "params": rest})
                if ar:
                    n_ar += len(ar)
                    if base_lr is not None:
                        out.append({**group, "params": ar, "lr": base_lr * mult})
                    else:
                        out.append({**group, "params": ar})
            if n_ar:
                self.print_and_status_update(f"YuE2: AR expert LoRA learning rate x{mult} ({n_ar} tensors)")
            return out

        value.prepare_optimizer_params = prepare_optimizer_params
        value._yue2_lr_split = True

    def get_noise_prediction(self, latent_model_input: torch.Tensor, timestep: torch.Tensor, text_embeddings: PromptEmbeds, batch=None, **kwargs):
        # `batch` must be a named parameter: predict_noise only forwards it when it sees it in the signature
        if batch is None or not isinstance(batch.latents, DTO) or batch.latents.get("tokens") is None:
            raise ValueError("YuE2 training needs codec tokens in the latent cache; enable latent caching")
        abc_all = batch.latents.get("abc_ids")
        self._check_abc_cache(batch.latents)
        start, end = getattr(batch, "yue2_window", (0, latent_model_input.shape[1]))
        tokens_all = batch.latents.tokens
        model = self.model
        if model.device == torch.device("cpu"):
            model.to(self.device_torch)
        device = self.device_torch
        t = (timestep.to(device, dtype=torch.float32) / 1000.0).to(self.torch_dtype)
        prefix_embeds = text_embeddings.text_embeds.to(device, self.torch_dtype)
        prefix_mask = text_embeddings.attention_mask.to(device)
        train_ar = self.ar_loss_weight > 0 and torch.is_grad_enabled()
        if torch.is_grad_enabled():
            self._note_train_call()
        captions = batch.get_caption_list() if hasattr(batch, "get_caption_list") else None
        preds, lm_losses, kl_losses = [], [], []
        stem_losses = {v: [] for v in SEP_VARIANTS[1:]}
        for i in range(latent_model_input.shape[0]):
            song = tokens_all[i].to(device)
            total = song.shape[0]
            prefix = self._prefix_segment(prefix_embeds[i], prefix_mask[i])
            prefix, abc = self._item_prefix_and_abc(prefix, abc_all, i, None if captions is None else captions[i])
            # NAR conditioning follows the released chunk protocol: prefix + abc + window + MUSIC_END
            embeds, ids = self._ar_inputs(prefix, abc, song[start:end])
            whole_song = start == 0 and end == total
            if train_ar and not whole_song:
                # the language-model loss must see the song from its beginning, or the AR
                # learns that lyrics and tokens can start anywhere relative to each other
                limit = total if self.ar_max_tokens <= 0 else min(total, self.ar_max_tokens)
                ar_embeds, ar_ids = self._ar_inputs(prefix, abc, song[:limit], end_token=limit == total)
                ce, kl, info = self._ar_losses(ar_embeds, ar_ids, prefix, total, whole_song)
                lm_losses.append(ce)
                if kl is not None:
                    kl_losses.append(kl)
                self._pending_aux_info = info
                with torch.no_grad():
                    cache, _ = model.ar.prefill(embeds)
            elif train_ar:
                ce, kl, info = self._ar_losses(embeds, ids, prefix, total, whole_song)
                lm_losses.append(ce)
                if kl is not None:
                    kl_losses.append(kl)
                self._pending_aux_info = info
                with torch.no_grad():
                    cache, _ = model.ar.prefill(embeds)
            else:
                cache, _ = model.ar.prefill(embeds)
            if train_ar and self.do_separation:
                for segment, variant in enumerate(SEP_VARIANTS[1:], start=2):
                    ce, kl = self._stem_ar_loss(batch.latents, i, variant, segment, prefix_embeds[i], prefix_mask[i], None if captions is None else captions[i])
                    stem_losses[variant].append(ce)
                    if kl is not None:
                        kl_losses.append(kl)
            # the flow loss must not train the AR through its KV cache: only next-token CE shapes the AR
            cache = [(k.detach(), v.detach()) for k, v in cache]
            pred = model.nar(latent_model_input[i : i + 1].to(device, self.torch_dtype), t[i : i + 1], cache, embeds.shape[1])
            preds.append(pred)
        if lm_losses:
            self._pending_ar_ce = torch.stack(lm_losses).mean()
            ar_ce = self._pending_ar_ce
            self._pending_stem_ce = {}
            if self.do_separation:
                # weighted mean over the three (prompt, target) pairs, mix weight 1; logs keep the unweighted terms
                self._pending_stem_ce = {v: torch.stack(terms).mean() for v, terms in stem_losses.items()}
                weights = self.separation_weights
                ar_ce = (ar_ce + sum(weights[v] * ce for v, ce in self._pending_stem_ce.items())) / (1.0 + sum(weights.values()))
            self._pending_ar_kl = torch.stack(kl_losses).mean() if kl_losses else None
            self._pending_aux_loss = ar_ce * self.ar_loss_weight
            if self._pending_ar_kl is not None:
                self._pending_aux_loss = self._pending_aux_loss + self._pending_ar_kl * self.ar_kl_weight
        return torch.cat(preds, 0)

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return self.pipeline


    def generate_single_audio(self, pipeline: YuE2Pipeline, gen_config: GenerateImageConfig, conditional_embeds: PromptEmbeds, unconditional_embeds, generator, extra):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        if gen_config.output_ext not in ["mp3", "wav", "flac"]:
            gen_config.output_ext = "mp3"
        parsed = parse_caption(gen_config.prompt)
        # sample-config duration field first, then a [Duration] prompt line, then the model default
        max_seconds = getattr(gen_config, "duration", None) or parsed["duration"] or self.sample_max_seconds
        max_tokens = max(1, int(round(max_seconds * FRAMES_PER_SECOND)))
        head = self._prefix_segment(conditional_embeds.text_embeds[0], conditional_embeds.attention_mask[0])[None]
        head = head.to(self.device_torch, self.torch_dtype)

        # stage 1 (cot full/melody): the AR writes the ABC sheet after ABC_START, as Comfy's Generate ABC does
        abc_ids: List[int] = []
        if self.cot != "off":
            self._status_update("YuE2: generating ABC sheet")
            abc_bar = tqdm(total=self.abc_max_tokens, desc="  abc tokens", unit="tok", position=1, leave=False, dynamic_ncols=True)
            try:
                abc_ids = pipeline.generate_abc_tokens(
                    head, seed=gen_config.seed, max_tokens=self.abc_max_tokens, progress=lambda done, total: abc_bar.update(done - abc_bar.n)
                )
            finally:
                abc_bar.close()
        tail = torch.tensor([self.tokenizer.abc_tail_ids(abc_ids)], device=self.device_torch)
        prefix = torch.cat([head, self.model.ar.embed(tail).to(head.dtype)], dim=1)
        # one acoustic frame needs two positions plus the boundary tokens (Comfy's budget rule)
        max_tokens = min(max_tokens, CONTEXT - prefix.shape[1] - 5)
        if max_tokens < 1:
            raise ValueError("YuE2 prompt leaves no room for music; shorten the style, lyrics, or ABC")

        # nested bars under generate_images' "Generating Samples" bar (position 0)
        self._status_update(f"YuE2: generating codec tokens (up to {max_tokens / FRAMES_PER_SECOND:.0f}s)")
        token_bar = tqdm(total=max_tokens, desc="  codec tokens", unit="tok", position=1, leave=False, dynamic_ncols=True)

        def token_progress(done, total):
            token_bar.update(done - token_bar.n)
            token_bar.set_postfix_str(f"{done / FRAMES_PER_SECOND:.1f}s", refresh=False)

        try:
            codec = pipeline.generate_codec_tokens(
                prefix, max_tokens=max_tokens, seed=gen_config.seed, temperature=self.sample_ar_temperature,
                repetition_penalty=self.sample_ar_repetition_penalty, progress=token_progress, legacy_off=self.cot == "off",
            )
        finally:
            token_bar.close()
        if len(codec) == 0:
            codec = [0]
        self._status_update(f"YuE2: rendering {len(codec) / FRAMES_PER_SECOND:.1f}s")
        render_bar = tqdm(desc=f"  render {len(codec) / FRAMES_PER_SECOND:.0f}s", unit="step", position=1, leave=False, dynamic_ncols=True)

        def render_step(i, n, x):
            if render_bar.total != n:
                render_bar.total = n
            render_bar.update(i + 1 - render_bar.n)
            self._emit_sample_step(x, i, n)

        try:
            latents = pipeline.synthesize(
                prefix,
                codec,
                seed=gen_config.seed,
                steps=gen_config.num_inference_steps,
                step_callback=render_step,
            )
        finally:
            render_bar.close()
        audio = pipeline.decode(latents)  # [1, 2, samples]
        return audio.cpu()

    # ------------------------------------------------------------------
    # LoRA key layout: transformer.nar.* -> diffusion_model.*, transformer.ar.* -> text_encoders.*
    # ------------------------------------------------------------------
    def convert_lora_weights_before_save(self, state_dict):
        out = {}
        for k, v in state_dict.items():
            if k.startswith("transformer.nar."):
                k = "diffusion_model." + k[len("transformer.nar.") :]
            elif k.startswith("transformer.ar."):
                k = "text_encoders." + k[len("transformer.ar.") :]
            out[k] = v
        return out

    def convert_lora_weights_before_load(self, state_dict):
        out = {}
        for k, v in state_dict.items():
            if k.startswith("diffusion_model."):
                k = "transformer.nar." + k[len("diffusion_model.") :]
            elif k.startswith("text_encoders."):
                k = "transformer.ar." + k[len("text_encoders.") :]
            out[k] = v
        return out
