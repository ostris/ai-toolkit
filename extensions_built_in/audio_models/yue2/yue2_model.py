"""YuE2 (m-a-p/YuE2-3B) for ai-toolkit.

Weights: the Comfy-Org all-in-one checkpoint (``checkpoints/yue2_3b_bf16.safetensors``)
resolved under MODELS_PATH in ComfyUI's folder layout and downloaded there when missing.
The official audio -> semantic-token tokenizer is unreleased; training conditioning comes
from the community head (MERT-v2-FullSong + classifier) in
``Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4``. Its companion NAR adapter is
merged into the base NAR on load (``model_kwargs.merge_nar_lora``, default true) since the
head's tokens are a dialect the stock renderer was not trained on.

Training: per song the latent cache carries VAE latents plus the head's codec tokens; the
"text embedding" is the AR prompt prefix (instruction + tags + lyrics) as token embeddings.
Each step crops one window (``model_kwargs.train_window_frames``, 25 fps, 0 = whole song),
prefills the AR expert over ``prefix + tokens[window] + MUSIC_END`` and trains the NAR
flow loss on the real latents. With ``model_kwargs.ar_loss_weight`` > 0 the same prefill
also trains the AR expert with next-token cross entropy over the codec tokens, so one LoRA
network on both experts learns composition and rendering together.

Captions use the ace-step tag format: <CAPTION> (style), <LYRICS>, <DURATION> (max seconds
for samples).
"""

import os
import random
import re
from typing import List, Optional

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from extensions_built_in.audio_models.base_audio_model import BaseAudioModel
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig
from toolkit.dto import DTO
from toolkit.models.v2.resolver import resolve_named_file
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

from .src.model import CODEC_OFFSET, FRAMES_PER_SECOND, MUSIC_END, YuE2Model, merge_nar_lora
from .src.pipeline import YuE2Pipeline
from .src.tokenizer import HEAD_FILE, HEAD_REPO, NAR_LORA_FILE, SemanticTokenizer, YuE2TextTokenizer
from .src.vae import SAMPLE_RATE, YuE2VAE

DEFAULT_CHECKPOINT = "Comfy-Org/YuE2/checkpoints/yue2_3b_bf16.safetensors"

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
        return {"style": _tag(text, "CAPTION"), "lyrics": _tag(text, "LYRICS"), "duration": _number(_tag(text, "DURATION"))}
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
        sections.setdefault(current, []).append(line)
    style = "\n".join(sections["tags"]).strip()
    lyrics = "\n".join(sections.get("lyrics", [])).strip()
    duration = _number("\n".join(sections.get("duration", [])))
    return {"style": style, "lyrics": lyrics, "duration": duration}


class YuE2TextEncoder(torch.nn.Module):
    """Prompt prefix -> AR token embeddings. Holds no weights of its own; the
    embedding table lives in the AR expert (looked up on whatever device it is on)."""

    def __init__(self, model: YuE2Model, tokenizer: YuE2TextTokenizer, cot: str = "off"):
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
        return self._model[0].ar.model.embed_tokens.weight.dtype

    def prefix_ids(self, style: str, lyrics: str) -> List[int]:
        return self.tokenizer.prefix_ids(style, lyrics, cot=self.cot)

    @torch.no_grad()
    def forward(self, style: str, lyrics: str) -> torch.Tensor:
        ids = torch.tensor([self.prefix_ids(style, lyrics)], dtype=torch.long)
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
        self._loss_log_every = int(kw.get("loss_log_every", 25))
        self._loss_log_step = 0
        self.merge_nar_lora = bool(kw.get("merge_nar_lora", True))
        self.nar_lora_path = kw.get("nar_lora_path", f"{HEAD_REPO}/{NAR_LORA_FILE}")
        self.semantic_head_path = kw.get("semantic_head_path", f"{HEAD_REPO}/{HEAD_FILE}")
        self.sample_max_seconds = float(kw.get("sample_max_seconds", 120.0))
        self.sample_ar_temperature = float(kw.get("sample_ar_temperature", 1.0))
        self.semantic_tokenizer: Optional[SemanticTokenizer] = None
        self._pending_aux_loss = None

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
            lora_path = resolve_named_file(self.nar_lora_path, component="yue2 nar lora")
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
        self.text_encoder = YuE2TextEncoder(self.model, self.tokenizer)
        self.pipeline = YuE2Pipeline(self.model, self.vae)
        self.pipeline.do_tiled_decoding = True
        self.pipeline.use_cuda_graphs = bool(self.model_config.model_kwargs.get("ar_cuda_graphs", True))

    def _get_semantic_tokenizer(self) -> SemanticTokenizer:
        if self.semantic_tokenizer is None:
            head_path = resolve_named_file(self.semantic_head_path, component="yue2 semantic head")
            # log only: this can run mid-training and a status update would replace "Training" in the UI
            print_acc("Loading YuE2 semantic tokenizer (MERT-v2-FullSong + head)")
            self.semantic_tokenizer = SemanticTokenizer(head_path)
        return self.semantic_tokenizer

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
            e = self.text_encoder(parsed["style"], parsed["lyrics"])  # [1, L, H]
            embeds.append(e)
            masks.append(torch.ones(1, e.shape[1], dtype=torch.long, device=e.device))
        max_len = max(e.shape[1] for e in embeds)
        embeds = [torch.nn.functional.pad(e, (0, 0, 0, max_len - e.shape[1])) for e in embeds]
        masks = [torch.nn.functional.pad(m, (0, max_len - m.shape[1])) for m in masks]
        return PromptEmbeds(torch.cat(embeds, 0), attention_mask=torch.cat(masks, 0))

    def encode_audio(self, audio_tensor: torch.Tensor, device=None, dtype=None):
        """[B, 2, samples] at 48 kHz -> DTO [B, T, 64] with codec ``tokens`` [B, T]."""
        if device is None:
            device = self.vae_device_torch
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
        return DTO(latents, tokens=tokens[:, :n].to(latents.device, torch.int32))

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

    def scale_loss(self, loss):
        aux = self._pending_aux_loss
        self._pending_aux_loss = None
        if aux is None:
            return loss
        self._loss_log_step += 1
        if self._loss_log_every > 0 and self._loss_log_step % self._loss_log_every == 0:
            # the trainer's bar shows the sum; the two terms move on different scales
            info = getattr(self, "_pending_aux_info", "")
            tqdm.write(f"yue2 step {self._loss_log_step}: flow {loss.detach().float().mean().item():.4f}  ar_ce {aux.detach().float().item() / max(self.ar_loss_weight, 1e-8):.4f}  {info}")
        return loss + aux

    @staticmethod
    def _train_prefix(prefix_embeds: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        """Unpadded prefix embeds for one item."""
        # the mask may arrive cast to bf16; count entries, never sum them
        length = int((prefix_mask > 0.5).sum().item())
        return prefix_embeds[:length]

    def _ar_info(self, logits: torch.Tensor, ids: torch.Tensor, prefix: torch.Tensor, total: int, whole_song: bool) -> str:
        """Readout diagnostics: what the AR loss was computed over."""
        if self._loss_log_every <= 0 or (self._loss_log_step + 1) % self._loss_log_every != 0:
            return ""
        with torch.no_grad():
            p = logits.softmax(-1)
            p_target = p.gather(1, ids[:, None]).squeeze(1)
            return (f"[ar targets {ids.shape[0]} unique {ids.unique().numel()} (song {total}, whole {whole_song}) "
                    f"prefix {prefix.shape[0]} p(END|prefix) {p[0, MUSIC_END].item():.3f} p(target) median {p_target.median().item():.3f}]")

    def _ar_inputs(self, prefix: torch.Tensor, tokens: torch.Tensor, end_token: bool = True):
        """One item: prefix (unpadded [L, H]) + codec tokens [+ MUSIC_END] -> embeds [1, L', H], ids [n]."""
        ids = tokens.long() + CODEC_OFFSET
        if end_token:
            ids = torch.cat([ids, torch.tensor([MUSIC_END], device=tokens.device)])
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
        preds, lm_losses = [], []
        for i in range(latent_model_input.shape[0]):
            song = tokens_all[i].to(device)
            total = song.shape[0]
            prefix = self._train_prefix(prefix_embeds[i], prefix_mask[i])
            # NAR conditioning follows the released chunk protocol: prefix + window + MUSIC_END
            embeds, ids = self._ar_inputs(prefix, song[start:end])
            whole_song = start == 0 and end == total
            if train_ar and not whole_song:
                # the language-model loss must see the song from its beginning, or the AR
                # learns that lyrics and tokens can start anywhere relative to each other
                limit = total if self.ar_max_tokens <= 0 else min(total, self.ar_max_tokens)
                ar_embeds, ar_ids = self._ar_inputs(prefix, song[:limit], end_token=limit == total)
                _, hidden = model.ar.prefill(ar_embeds, return_hidden=True)
                logits = model.ar.model.lm_head(hidden[0, -ar_ids.shape[0] - 1 : -1]).float()
                lm_losses.append(torch.nn.functional.cross_entropy(logits, ar_ids))
                self._pending_aux_info = self._ar_info(logits, ar_ids, prefix, total, whole_song)
                with torch.no_grad():
                    cache, _ = model.ar.prefill(embeds)
            else:
                cache, hidden = model.ar.prefill(embeds, return_hidden=train_ar)
                if train_ar:
                    logits = model.ar.model.lm_head(hidden[0, -ids.shape[0] - 1 : -1]).float()
                    lm_losses.append(torch.nn.functional.cross_entropy(logits, ids))
                    self._pending_aux_info = self._ar_info(logits, ids, prefix, total, whole_song)
            # the flow loss must not train the AR through its KV cache: only next-token CE shapes the AR
            cache = [(k.detach(), v.detach()) for k, v in cache]
            pred = model.nar(latent_model_input[i : i + 1].to(device, self.torch_dtype), t[i : i + 1], cache, embeds.shape[1])
            preds.append(pred)
        if lm_losses:
            self._pending_aux_loss = torch.stack(lm_losses).mean() * self.ar_loss_weight
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
        length = int((conditional_embeds.attention_mask[0] > 0.5).sum().item())
        prefix = conditional_embeds.text_embeds[:1, :length].to(self.device_torch, self.torch_dtype)

        # nested bars under generate_images' "Generating Samples" bar (position 0)
        self._status_update(f"YuE2: generating codec tokens (up to {max_seconds:.0f}s)")
        token_bar = tqdm(total=max_tokens, desc="  codec tokens", unit="tok", position=1, leave=False, dynamic_ncols=True)

        def token_progress(done, total):
            token_bar.update(done - token_bar.n)
            token_bar.set_postfix_str(f"{done / FRAMES_PER_SECOND:.1f}s", refresh=False)

        try:
            codec = pipeline.generate_codec_tokens(
                prefix, max_tokens=max_tokens, seed=gen_config.seed, temperature=self.sample_ar_temperature, progress=token_progress,
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
