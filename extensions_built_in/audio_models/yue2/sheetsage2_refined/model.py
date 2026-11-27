"""SheetSage2 seq2seq model: MERT-v2 encoder -> layer mix -> token decoder."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from .decoder import DecoderConfig, TokenDecoder
from .encoder import EncoderConfig, MERTEncoder
from .vocab import Vocab

WINDOW_SECONDS = 300.0  # decoder window; see ``SheetSage.pad_encoder`` for windows shorter than this


class SheetSage(nn.Module):
    def __init__(self, encoder_config: EncoderConfig = EncoderConfig(), decoder_config: DecoderConfig | None = None):
        super().__init__()
        self.vocab = Vocab(WINDOW_SECONDS)
        decoder_config = decoder_config or DecoderConfig(vocab_size=self.vocab.size)
        if self.vocab.size != decoder_config.vocab_size:
            raise ValueError(f"vocabulary has {self.vocab.size} tokens, decoder expects {decoder_config.vocab_size}")
        self.encoder = MERTEncoder(encoder_config)
        # Softmax weights over the encoder input state and every Conformer block.
        self.layer_weights = nn.Parameter(torch.zeros(encoder_config.layers + 1))
        self.encoder_projection = nn.Linear(encoder_config.width, decoder_config.width)
        self.decoder = TokenDecoder(decoder_config)
        # SheetSage2 zero-pads a short window to WINDOW_SECONDS before encoding (the encoder attends
        # to the padding); the fine-tuned decoders learnt on windows encoded at their own length.
        self.pad_encoder = True
        self.bar_features = None  # heads.BarFeatures, for decoders trained with it

    @property
    def sample_rate(self) -> int:
        return self.encoder.config.sample_rate

    def encode(self, waveform: Tensor) -> Tensor:
        """``[B, samples]`` at 24 kHz -> decoder memory ``[B, frames, 512]``."""

        return self.memory(self.encoder(waveform))

    def memory(self, states: list[Tensor]) -> Tensor:
        """Encoder hidden states -> decoder memory (the layer mix and projection)."""

        weights = torch.softmax(self.layer_weights, dim=0)
        mixed = states[0] * weights[0]
        for state, weight in zip(states[1:], weights[1:]):
            mixed = mixed + state * weight
        return self.encoder_projection(mixed)

    @classmethod
    def load(cls, path: Path, device: str | torch.device = "cpu", decoder: Path | None = None) -> "SheetSage":
        """SheetSage2's weights; ``decoder`` replaces the decoder with a fine-tuned one (and stops padding
        short windows for the encoder, as the fine-tuned decoders were trained)."""

        model = cls()
        state = load_file(str(path), device=str(device))
        # SheetSage2 knows 31678 tokens; the rows added after them (chord_detail, key_mode, ...) are only
        # ever produced by a fine-tuned decoder, so for SheetSage2 itself they stay zero (never allowed).
        name = "decoder.token_embedding.weight"
        missing = model.vocab.size - state[name].shape[0]
        state[name] = torch.cat((state[name], state[name].new_zeros(missing, state[name].shape[1])))
        model.load_state_dict(state)
        if decoder is not None:
            weights = load_file(str(decoder), device=str(device))
            # ``chord_pitch``: an auxiliary head used only in training
            model.decoder.load_state_dict({k: v for k, v in weights.items() if not k.startswith("chord_pitch.")})
            model.pad_encoder = False
        return model.to(device).eval()
