from typing import List, Optional
from dataclasses import dataclass, field

from transformers import PretrainedConfig, Qwen3Config


@dataclass
class MossMusicEncoderConfig:
    d_model: int = 1280
    output_dim: int = 1280
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    downsample_rate: int = 8
    downsample_hidden_size: int = 480
    encoder_attention_window_size: int = 100
    max_source_positions: int = 1500
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    layer_norm_eps: float = 1e-5
    _attn_implementation: str = "eager"
    pretrained_path: str = ""

    n_window: int = 200
    conv_chunksize: int = 64

    deepstack_encoder_layer_indexes: List[int] = field(
        default_factory=lambda: [8, 16, 24]
    )

    @classmethod
    def from_dict(cls, config_dict):
        if config_dict is None:
            return cls()
        allowed_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in allowed_keys}
        return cls(**filtered)

    def to_dict(self):
        return {
            "d_model": self.d_model,
            "output_dim": self.output_dim,
            "num_mel_bins": self.num_mel_bins,
            "encoder_layers": self.encoder_layers,
            "encoder_attention_heads": self.encoder_attention_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "downsample_rate": self.downsample_rate,
            "downsample_hidden_size": self.downsample_hidden_size,
            "encoder_attention_window_size": self.encoder_attention_window_size,
            "max_source_positions": self.max_source_positions,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "activation_dropout": self.activation_dropout,
            "activation_function": self.activation_function,
            "layer_norm_eps": self.layer_norm_eps,
            "_attn_implementation": self._attn_implementation,
            "pretrained_path": self.pretrained_path,
            "n_window": self.n_window,
            "conv_chunksize": self.conv_chunksize,
            "deepstack_encoder_layer_indexes": list(
                self.deepstack_encoder_layer_indexes or []
            ),
        }


class MossMusicConfig(PretrainedConfig):
    model_type = "moss_music"
    is_composition = True

    def __init__(
        self,
        audio_config=None,
        language_config=None,
        adapter_hidden_size=8192,
        ignore_index=-100,
        deepstack_num_inject_layers: Optional[int] = None,
        **kwargs,
    ):
        if isinstance(audio_config, dict):
            audio_config = MossMusicEncoderConfig.from_dict(audio_config)
        elif audio_config is None:
            audio_config = MossMusicEncoderConfig()

        if isinstance(language_config, dict):
            language_config = Qwen3Config(**language_config)
        elif language_config is None:
            language_config = Qwen3Config()

        self.audio_config = audio_config
        self.language_config = language_config
        self.adapter_hidden_size = adapter_hidden_size
        self.ignore_index = ignore_index
        self.deepstack_num_inject_layers = deepstack_num_inject_layers

        _propagate_keys = {
            "num_hidden_layers",
            "eos_token_id",
            "bos_token_id",
            "vocab_size",
            "tie_word_embeddings",
        }
        for key in ("num_hidden_layers", "eos_token_id", "bos_token_id", "vocab_size"):
            kwargs.setdefault(key, getattr(language_config, key, None))
        kwargs.setdefault("tie_word_embeddings", False)

        if hasattr(language_config, "to_dict"):
            _lang_keys = set(language_config.to_dict().keys())
            for key in list(kwargs.keys()):
                if key in _lang_keys and key not in _propagate_keys:
                    kwargs.pop(key)

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output["audio_config"] = (
            self.audio_config.to_dict()
            if hasattr(self.audio_config, "to_dict")
            else self.audio_config
        )
        output["language_config"] = (
            self.language_config.to_dict()
            if hasattr(self.language_config, "to_dict")
            else self.language_config
        )
        output["adapter_hidden_size"] = self.adapter_hidden_size
        output["ignore_index"] = self.ignore_index
        output["deepstack_num_inject_layers"] = self.deepstack_num_inject_layers
        return output


__all__ = ["MossMusicEncoderConfig", "MossMusicConfig"]
