from typing import List, Optional

import torch
import time
import os
from .model import (
    SAMPLE_RATE,
    AceStep15,
    OobleckVAE,
    TextEncoder,
    get_silence_latent,
    compute_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer

SFT_PROMPT = """# Instruction
{instruction}

# Caption
{caption}

# Metas
{metas}<|endoftext|>
"""


class AceStep15Pipeline:
    SAMPLE_RATE = 48000
    LATENT_RATE = 25  # 48000 / 1920
    SFT_PROMPT = SFT_PROMPT

    def __init__(self, transformer, vae, text_encoder, tokenizer, scheduler):
        self.transformer: AceStep15 = transformer
        self.vae: OobleckVAE = vae
        self.text_encoder: TextEncoder = text_encoder
        self.tokenizer: AutoTokenizer = tokenizer
        self.scheduler = scheduler

    def to(self, *args, **kwargs):
        self.transformer.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)

    def get_text_embedings(
        self, prompt, lyrics, bpm, key, time_sig, duration, language
    ):
        metas = f"- bpm: {bpm}\n- timesignature: {time_sig}\n- keyscale: {key}\n- duration: {int(duration)} seconds\n"
        caption = self.SFT_PROMPT.format(
            instruction="Fill the audio semantic mask based on the given conditions:",
            caption=prompt,
            metas=metas,
        )
        lyrics_text = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

        cap_tok = self.tokenizer(
            caption, truncation=True, max_length=256, return_tensors="pt"
        )
        lyr_tok = self.tokenizer(
            lyrics_text, truncation=True, max_length=2048, return_tensors="pt"
        )

        text_embeddings = self.text_encoder.encode_text(
            cap_tok.input_ids.to(self.text_encoder.device)
        ).to(self.transformer.dtype)
        text_mask = cap_tok.attention_mask.to(self.text_encoder.device).bool()
        lyric_embeddings = self.text_encoder.encode_lyrics(
            lyr_tok.input_ids.to(self.text_encoder.device)
        ).to(self.transformer.dtype)
        lyric_mask = lyr_tok.attention_mask.to(self.text_encoder.device).bool()

        return text_embeddings, text_mask, lyric_embeddings, lyric_mask

    def __call__(
        self,
        prompt="",
        lyrics="",
        encoder_embeddings: Optional[List[torch.Tensor]] = None,
        encoder_mask: Optional[List[torch.Tensor]] = None,
        # uses a null conditional for unconditional if not provided, which is what we want for CFG
        num_inference_steps=50,
        duration=30.0,
        generator: torch.Generator = None,
        bpm="N/A",
        key="N/A",
        time_sig="N/A",
        language="en",
        guidance_scale=1.0,
    ):
        t_sched = compute_timesteps(num_inference_steps, 3.0)
        latent_len = int(duration * self.LATENT_RATE)
        device = self.transformer.device
        dtype = self.transformer.dtype

        # Text encoding
        if encoder_embeddings is not None and encoder_mask is not None:
            enc_h = encoder_embeddings
            enc_m = encoder_mask
            sil = get_silence_latent(latent_len, device, dtype)  # [1, 64, T]
            src = sil.transpose(1, 2)  # [1, T, 64]
            chunk_masks = torch.ones_like(src)
            ctx = torch.cat([src, chunk_masks.to(src.dtype)], dim=-1)
        else:
            text_h, text_m, lyric_h, lyric_m = self.get_text_embedings(
                prompt, lyrics, bpm, key, time_sig, duration, language
            )

            # Silence as source latent [1, 64, T] -> [1, T, 64] for DiT
            sil = get_silence_latent(latent_len, device, dtype)  # [1, 64, T]
            src = sil.transpose(1, 2)  # [1, T, 64]
            chunk_masks = torch.ones_like(src)

            # Reference audio (silence)
            ref = sil[:, :, :750].transpose(1, 2)  # [1, 750, 64]
            ref_order = torch.zeros(1, device=device, dtype=torch.long)

            # Prepare conditions (conditional)
            enc_h, enc_m, ctx = self.transformer.prepare_condition(
                text_h, text_m, lyric_h, lyric_m, ref, ref_order, src, chunk_masks
            )

        # Prepare unconditional conditions for CFG
        use_cfg = guidance_scale > 1.0
        enc_h_uncond = None
        if use_cfg:
            enc_h_uncond = self.transformer.null_condition_emb.expand_as(enc_h)

        # Noise
        if generator is None:
            generator = torch.Generator(device=device)
        noise_ch = ctx.shape[-1] // 2
        xt = randn_tensor(
            (1, latent_len, noise_ch), generator=generator, device=device, dtype=dtype
        )
        # xt = torch.randn(1, latent_len, noise_ch, generator=generator, device=device, dtype=dtype)

        # Diffusion
        t_sched_t = torch.tensor(t_sched, device=device, dtype=dtype)
        attn = torch.ones(1, latent_len, device=device, dtype=dtype)

        for i in range(len(t_sched_t)):
            tv = t_sched_t[i].item()
            tt = torch.full((1,), tv, device=device, dtype=dtype)

            vt_cond = self.transformer.decoder(xt, tt, tt, attn, enc_h, enc_m, ctx)

            if use_cfg:
                vt_uncond = self.transformer.decoder(
                    xt, tt, tt, attn, enc_h_uncond, enc_m, ctx
                )
                vt = vt_uncond + guidance_scale * (vt_cond - vt_uncond)
            else:
                vt = vt_cond

            if i == len(t_sched_t) - 1:
                xt = xt - vt * tv
            else:
                xt = xt - vt * (tv - t_sched_t[i + 1].item())

        # VAE decode
        wav = self.vae.decode(xt.transpose(1, 2))  # [1, 2, samples]
        wav = wav[0, :, : int(duration * SAMPLE_RATE)]
        return wav
