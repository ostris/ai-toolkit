from typing import Any, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from .configuration import MossMusicConfig, MossMusicEncoderConfig


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        max_timescale = 10000.0
        log_timescale_increment = math.log(max_timescale) / (embedding_dim // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(embedding_dim // 2).float()
        )
        self.register_buffer("inv_timescales", inv_timescales, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        scaled_time = torch.arange(
            seq_len, device=device, dtype=self.inv_timescales.dtype
        ).unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        sin_emb = torch.sin(scaled_time)
        cos_emb = torch.cos(scaled_time)
        pos_emb = torch.cat([sin_emb, cos_emb], dim=1)
        return pos_emb.unsqueeze(0)


class MossMusicEncoder(nn.Module):
    """Audio encoder with conv-stem downsampling and Whisper transformer layers."""

    def __init__(self, config: MossMusicEncoderConfig):
        super().__init__()
        self.config = config
        self.gelu = nn.GELU()

        self.conv1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.conv3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        # 128 mel bins / 8 = 16 after 3 convs with stride=2
        self.stem_proj = nn.Linear(config.downsample_hidden_size * 16, config.d_model)
        self.embed_positions = SinusoidsPositionEmbedding(
            config.max_source_positions, config.d_model
        )
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.out_proj = (
            nn.Linear(config.d_model, config.output_dim, bias=False)
            if config.output_dim != config.d_model
            else nn.Identity()
        )

        self.deepstack_encoder_layer_indexes = list(
            config.deepstack_encoder_layer_indexes or []
        )
        self._deepstack_capture_map = {
            layer_idx: capture_idx
            for capture_idx, layer_idx in enumerate(self.deepstack_encoder_layer_indexes)
        }

        self.n_window = int(config.n_window)
        self.chunk_frames = int(self.n_window * 2)
        self.conv_chunksize = int(config.conv_chunksize)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @staticmethod
    def _compute_downsampled_length(lengths: torch.Tensor) -> torch.Tensor:
        def conv_out_len(L):
            return (L - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(lengths)))

    def _encode_chunk_batch(
        self,
        input_features: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode a batch of (already padded) chunks through the conv stem and
        transformer layers. Returns (last_hidden, ordered_deepstack_hidden_states).
        """
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        downsampled_lengths = self._compute_downsampled_length(seq_lengths)

        # [B, n_mels, T] -> [B, 1, n_mels, T]
        x = input_features.unsqueeze(1)
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))

        # [B, C, F, T] -> [B, T, C*F]
        x = x.permute(0, 3, 1, 2).contiguous().flatten(2)
        x = self.stem_proj(x)

        max_len = int(downsampled_lengths.max().item())
        if x.size(1) > max_len:
            x = x[:, :max_len, :]

        positions = self.embed_positions(x.shape[1], x.device)
        x = x + positions.to(x.dtype)

        padding_mask = (
            torch.arange(x.size(1), device=x.device)[None, :]
            >= downsampled_lengths[:, None]
        )
        attention_mask = (1.0 - (~padding_mask).to(dtype=x.dtype)) * torch.finfo(
            x.dtype
        ).min
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        deepstack_hidden_states: List[Optional[torch.Tensor]] = [None] * len(
            self.deepstack_encoder_layer_indexes
        )
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
            capture_idx = self._deepstack_capture_map.get(layer_idx)
            if capture_idx is not None:
                deepstack_hidden_states[capture_idx] = x

        x = self.layer_norm(x)
        x = self.out_proj(x)

        ordered_deepstack_hidden_states = [
            h for h in deepstack_hidden_states if h is not None
        ]
        if not isinstance(self.out_proj, nn.Identity):
            ordered_deepstack_hidden_states = [
                self.out_proj(h) for h in ordered_deepstack_hidden_states
            ]
        return x, ordered_deepstack_hidden_states

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: Optional[torch.Tensor] = None,
        output_deepstack_hidden_states: bool = True,
    ) -> BaseModelOutputWithPast:
        if input_features.dim() == 3:
            if feature_lens is None:
                feature_lens = torch.full(
                    (input_features.size(0),),
                    input_features.size(-1),
                    dtype=torch.long,
                    device=input_features.device,
                )
            else:
                feature_lens = feature_lens.to(
                    device=input_features.device, dtype=torch.long
                )
            valid_chunks = [
                input_features[i, :, : int(feature_lens[i].item())]
                for i in range(int(input_features.shape[0]))
            ]
            input_features = torch.cat(valid_chunks, dim=1)
        elif input_features.dim() != 2:
            raise ValueError(
                f"Expected [n_mels, T] or [B, n_mels, T], got {tuple(input_features.shape)}."
            )

        if feature_lens is None:
            feature_lens = torch.tensor(
                [int(input_features.shape[1])],
                device=input_features.device,
                dtype=torch.long,
            )
        else:
            feature_lens = feature_lens.to(
                device=input_features.device, dtype=torch.long
            )

        chunk_frames = int(self.chunk_frames)
        chunk_num = torch.ceil(
            feature_lens.to(torch.float32) / float(chunk_frames)
        ).long()
        chunk_lengths = torch.full(
            (int(chunk_num.sum().item()),),
            chunk_frames,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % chunk_frames
        chunk_lengths[chunk_lengths == 0] = chunk_frames

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(
            chunk_list, batch_first=True
        ).transpose(1, 2)

        feature_lens_after_cnn = self._compute_downsampled_length(chunk_lengths)
        t_down_max = (
            int(feature_lens_after_cnn.max().item())
            if feature_lens_after_cnn.numel() > 0
            else 0
        )
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [
                torch.ones(int(L.item()), dtype=torch.bool, device=padded_feature.device)
                for L in feature_lens_after_cnn
            ],
            batch_first=True,
        )
        if padded_mask_after_cnn.shape[1] < t_down_max:
            padded_mask_after_cnn = F.pad(
                padded_mask_after_cnn,
                (0, t_down_max - padded_mask_after_cnn.shape[1]),
                value=False,
            )

        num_deepstack = len(self.deepstack_encoder_layer_indexes)
        padded_embeds: List[torch.Tensor] = []
        deepstack_padded_embeds: List[List[torch.Tensor]] = [
            [] for _ in range(num_deepstack)
        ]
        for feat_chunk, len_chunk in zip(
            padded_feature.split(self.conv_chunksize, dim=0),
            chunk_lengths.split(self.conv_chunksize, dim=0),
        ):
            out, deepstack_outs = self._encode_chunk_batch(feat_chunk, len_chunk)
            if out.shape[1] < t_down_max:
                out = F.pad(out, (0, 0, 0, t_down_max - out.shape[1]))
            padded_embeds.append(out)
            if output_deepstack_hidden_states and num_deepstack > 0:
                if len(deepstack_outs) != num_deepstack:
                    raise RuntimeError(
                        "Deepstack output count does not match configured layer indexes."
                    )
                for capture_idx, ds in enumerate(deepstack_outs):
                    if ds.shape[1] < t_down_max:
                        ds = F.pad(ds, (0, 0, 0, t_down_max - ds.shape[1]))
                    deepstack_padded_embeds[capture_idx].append(ds)

        if padded_embeds:
            padded_embed = torch.cat(padded_embeds, dim=0)
        else:
            padded_embed = torch.empty(
                (0, t_down_max, self.config.output_dim),
                device=padded_feature.device,
            )

        valid_tokens = padded_embed[padded_mask_after_cnn]  # [N_valid, D]
        last_hidden_state = valid_tokens.unsqueeze(0)  # [1, N_valid, D]

        deepstack_states: Optional[Tuple[torch.Tensor, ...]] = None
        if output_deepstack_hidden_states and num_deepstack > 0:
            collected: List[torch.Tensor] = []
            for chunks_list in deepstack_padded_embeds:
                if chunks_list:
                    ds = torch.cat(chunks_list, dim=0)
                    collected.append(ds[padded_mask_after_cnn].unsqueeze(0))
                else:
                    collected.append(
                        torch.empty(
                            (1, 0, self.config.output_dim),
                            device=padded_feature.device,
                            dtype=padded_embed.dtype,
                        )
                    )
            deepstack_states = tuple(collected)

        return BaseModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            hidden_states=deepstack_states,
        )


class GatedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MossMusicPreTrainedModel(PreTrainedModel):
    config_class = MossMusicConfig
    config: MossMusicConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {"hidden_states": Qwen3DecoderLayer}


class MossMusicModel(MossMusicPreTrainedModel, GenerationMixin):
    config_class = MossMusicConfig
    _tied_weights_keys: List[str] = []

    def __init__(self, config: MossMusicConfig):
        super().__init__(config)

        self.audio_encoder = MossMusicEncoder(config.audio_config)
        self.language_model = Qwen3Model(config.language_config)

        self.audio_adapter = GatedMLP(
            input_size=config.audio_config.output_dim,
            hidden_size=config.adapter_hidden_size,
            output_size=config.language_config.hidden_size,
        )

        deepstack_k = len(
            getattr(config.audio_config, "deepstack_encoder_layer_indexes", []) or []
        )
        if config.deepstack_num_inject_layers is not None:
            deepstack_k = min(deepstack_k, int(config.deepstack_num_inject_layers))
        self.deepstack_audio_merger_list = nn.ModuleList(
            [
                GatedMLP(
                    input_size=config.audio_config.output_dim,
                    hidden_size=config.adapter_hidden_size,
                    output_size=config.language_config.hidden_size,
                )
                for _ in range(deepstack_k)
            ]
        )

        self.vocab_size = config.language_config.vocab_size
        self.lm_head = nn.Linear(
            config.language_config.hidden_size, self.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_audio_features(self, input_features, feature_lens):
        audio_outputs = self.audio_encoder(
            input_features=input_features,
            feature_lens=feature_lens,
            output_deepstack_hidden_states=True,
        )
        deepstack = (
            list(audio_outputs.hidden_states)
            if audio_outputs.hidden_states is not None
            else None
        )
        return audio_outputs.last_hidden_state, deepstack

    def _apply_deepstack_to_hidden_states(
        self,
        hidden_states: torch.Tensor,
        audio_input_mask: torch.Tensor,
        deepstack_embeds: torch.Tensor,
    ) -> torch.Tensor:
        audio_input_mask = audio_input_mask.to(hidden_states.device)
        deepstack_embeds = deepstack_embeds.to(hidden_states.device, hidden_states.dtype)
        flat = deepstack_embeds.reshape(-1, deepstack_embeds.shape[-1])
        hs = hidden_states.clone()
        hs[audio_input_mask] = hs[audio_input_mask] + flat
        return hs

    def _register_llm_deepstack_hooks(
        self,
        audio_input_mask: torch.Tensor,
        deepstack_audio_embeds: List[torch.Tensor],
    ):
        if deepstack_audio_embeds is None or len(deepstack_audio_embeds) == 0:
            return []

        layers = getattr(self.language_model, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Qwen3Model does not expose `.layers`; cannot register DeepStack hooks."
            )

        num_inject = len(deepstack_audio_embeds)
        handles = []

        for layer_idx, layer in enumerate(layers):
            if layer_idx >= num_inject:
                break

            def _make_llm_hook(k: int):
                def _hook(_module, _inputs, _output):
                    if isinstance(_output, (tuple, list)):
                        hs = _output[0]
                        new_hs = self._apply_deepstack_to_hidden_states(
                            hs, audio_input_mask, deepstack_audio_embeds[k]
                        )
                        return (new_hs,) + tuple(_output[1:])
                    else:
                        return self._apply_deepstack_to_hidden_states(
                            _output, audio_input_mask, deepstack_audio_embeds[k]
                        )

                return _hook

            handles.append(layer.register_forward_hook(_make_llm_hook(layer_idx)))

        return handles

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        # generate() still passes return_dict; transformers 5 models ignore it
        kwargs.pop("return_dict", None)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        hook_handles = []
        if audio_data is not None:
            if audio_input_mask is None:
                raise ValueError(
                    "audio_input_mask is required when audio_data is provided."
                )

            audio_embeds, deepstack = self.get_audio_features(audio_data, audio_data_seqlens)
            audio_embeds = self.audio_adapter(audio_embeds)

            audio_token_count = int(audio_input_mask.to(torch.int32).sum().item())
            if audio_token_count != int(audio_embeds.shape[1]):
                raise ValueError(
                    f"Audio token count mismatch: audio_input_mask has {audio_token_count} audio tokens, "
                    f"but audio_embeds has length {int(audio_embeds.shape[1])}."
                )

            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds)

            if deepstack is not None and len(self.deepstack_audio_merger_list) > 0:
                deepstack_audio_embeds = []
                for i, x in enumerate(deepstack[: len(self.deepstack_audio_merger_list)]):
                    ds = self.deepstack_audio_merger_list[i](x)
                    if int(ds.shape[1]) != audio_token_count:
                        raise ValueError(
                            f"DeepStack audio seq_len mismatch at index {i}: "
                            f"expected {audio_token_count}, got {int(ds.shape[1])}."
                        )
                    deepstack_audio_embeds.append(ds)

                try:
                    hook_handles = self._register_llm_deepstack_hooks(
                        audio_input_mask, deepstack_audio_embeds
                    )
                except Exception:
                    for h in hook_handles:
                        h.remove()
                    raise

        try:
            outputs = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        finally:
            for h in hook_handles:
                h.remove()

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
            shift_logits = shift_logits.view(-1, self.config.language_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # audio is encoded once into the prompt embeds; re-sending it on decode
        # steps would re-run the encoder against a stale audio_input_mask
        if not is_first_iteration:
            for key in ("audio_data", "audio_data_seqlens", "audio_input_mask"):
                kwargs.pop(key, None)
        return super().prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )


__all__ = [
    "MossMusicEncoderConfig",
    "MossMusicConfig",
    "MossMusicModel",
]
