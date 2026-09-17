import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import BatchEncoding, Qwen2Tokenizer


@dataclass
class MelConfig:
    mel_sr: int = 16000
    mel_dim: int = 128
    mel_n_fft: int = 400
    mel_hop_length: int = 160
    mel_dtype: torch.dtype = torch.bfloat16


class MossMusicProcessor:
    """Builds MOSS-Music inputs: Whisper log-mel features plus a Qwen chat
    prompt whose audio span is one placeholder token per 80ms of audio, with
    the elapsed seconds written in as digit tokens every 2s (time markers)."""

    _AUDIO_SPAN_RE = re.compile(r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>")

    def __init__(
        self,
        tokenizer,
        *,
        mel_config: Optional[MelConfig] = None,
        enable_time_marker: bool = True,
        audio_token_id: int = 151654,
        audio_start_id: int = 151669,
        audio_end_id: int = 151670,
    ):
        self.tokenizer = tokenizer
        self.audio_token_id = int(audio_token_id)
        self.audio_start_id = int(audio_start_id)
        self.audio_end_id = int(audio_end_id)
        self.enable_time_marker = bool(enable_time_marker)
        self.config = mel_config or MelConfig()
        self._whisper_feature_extractor = None

        self._digit_token_ids = {str(d): 15 + d for d in range(10)}
        self.audio_tokens_per_second = 12.5
        self.time_marker_every_seconds = 2
        self.time_marker_every_audio_tokens = int(
            self.audio_tokens_per_second * self.time_marker_every_seconds
        )
        self.model_input_names = [
            "input_ids",
            "attention_mask",
            "audio_data",
            "audio_data_seqlens",
        ]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer_kwargs = {}
        for key in ["cache_dir", "revision", "token", "local_files_only"]:
            if key in kwargs:
                tokenizer_kwargs[key] = kwargs[key]
        # the checkpoint's tokenizer_config names Qwen2Tokenizer; loading it
        # directly skips AutoTokenizer's remote-code prompt for the auto_map
        tokenizer = Qwen2Tokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        return cls(
            tokenizer,
            mel_config=kwargs.pop("mel_config", None),
            enable_time_marker=kwargs.pop("enable_time_marker", True),
            audio_token_id=kwargs.pop("audio_token_id", 151654),
            audio_start_id=kwargs.pop("audio_start_id", 151669),
            audio_end_id=kwargs.pop("audio_end_id", 151670),
        )

    @staticmethod
    def _conv3_downsample_len(raw_mel_len: int) -> int:
        def conv_out_len(length: int) -> int:
            return (length - 1) // 2 + 1

        return conv_out_len(conv_out_len(conv_out_len(int(raw_mel_len))))

    def _get_whisper_feature_extractor(self):
        if self._whisper_feature_extractor is not None:
            return self._whisper_feature_extractor

        from transformers.models.whisper.feature_extraction_whisper import (
            WhisperFeatureExtractor,
        )

        self._whisper_feature_extractor = WhisperFeatureExtractor(
            feature_size=int(self.config.mel_dim),
            sampling_rate=int(self.config.mel_sr),
            hop_length=int(self.config.mel_hop_length),
            n_fft=int(self.config.mel_n_fft),
        )
        return self._whisper_feature_extractor

    def _extract_mel(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        wav_np = np.ascontiguousarray(audio, dtype=np.float32)
        if wav_np.ndim == 2:
            wav_np = wav_np[0]
        # the raw fbank pass has no 30s chunking, so full songs come through whole
        fe = self._get_whisper_feature_extractor()
        feats = fe._np_extract_fbank_features(wav_np[None, ...], device="cpu")
        return torch.from_numpy(feats[0]).to(dtype=self.config.mel_dtype)

    def _get_time_marker_token_ids(self, second: int) -> List[int]:
        return [self._digit_token_ids[digit] for digit in str(second)]

    def _build_audio_tokens_with_time_markers(self, audio_seq_len: int) -> List[int]:
        total_duration_seconds = audio_seq_len / self.audio_tokens_per_second
        num_full_seconds = int(total_duration_seconds)

        token_ids: List[int] = []
        audio_tokens_consumed = 0
        for second in range(
            self.time_marker_every_seconds,
            num_full_seconds + 1,
            self.time_marker_every_seconds,
        ):
            marker_pos = (
                second // self.time_marker_every_seconds
            ) * self.time_marker_every_audio_tokens
            audio_segment_len = marker_pos - audio_tokens_consumed
            if audio_segment_len > 0:
                token_ids.extend([self.audio_token_id] * audio_segment_len)
                audio_tokens_consumed += audio_segment_len
            token_ids.extend(self._get_time_marker_token_ids(second))

        remaining = audio_seq_len - audio_tokens_consumed
        if remaining > 0:
            token_ids.extend([self.audio_token_id] * remaining)
        return token_ids

    def _build_audio_placeholder_ids(self, num_audio_tokens: int) -> List[int]:
        if self.enable_time_marker:
            return self._build_audio_tokens_with_time_markers(num_audio_tokens)
        return [self.audio_token_id] * num_audio_tokens

    def _build_default_prompt(self, text: str, has_audio: bool) -> str:
        if has_audio:
            return (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                f"{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _build_input_from_prompt(self, prompt: str, token_lens: List[int]) -> List[int]:
        spans = list(self._AUDIO_SPAN_RE.finditer(prompt))
        if len(spans) != len(token_lens):
            raise ValueError(
                f"Audio placeholder count mismatch: found {len(spans)} spans in text, "
                f"but got {len(token_lens)} audio inputs."
            )

        input_ids: List[int] = []
        cursor = 0
        for index, match in enumerate(spans):
            prefix = prompt[cursor : match.start()]
            if prefix:
                input_ids.extend(self.tokenizer.encode(prefix, add_special_tokens=False))

            input_ids.append(self.audio_start_id)
            input_ids.extend(self._build_audio_placeholder_ids(int(token_lens[index])))
            input_ids.append(self.audio_end_id)
            cursor = match.end()

        suffix = prompt[cursor:]
        if suffix:
            input_ids.extend(self.tokenizer.encode(suffix, add_special_tokens=False))
        return input_ids

    def __call__(
        self,
        *,
        text: Union[str, Sequence[str]],
        audios: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        audio: Optional[Sequence[Union[np.ndarray, torch.Tensor]]] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError(f"Expected text batch size 1, got {len(text)}")
            prompt_text = text[0]
        else:
            prompt_text = text

        audio_list = audios if audios is not None else audio
        audio_list = [] if audio_list is None else list(audio_list)

        mels: List[torch.Tensor] = []
        raw_lengths: List[int] = []
        token_lens: List[int] = []
        for one_audio in audio_list:
            mel = self._extract_mel(one_audio)
            raw_len = int(mel.shape[-1])
            mels.append(mel)
            raw_lengths.append(raw_len)
            token_lens.append(self._conv3_downsample_len(raw_len))

        if mels:
            max_length = max(raw_lengths)
            audio_batch = torch.zeros(
                (len(mels), self.config.mel_dim, max_length),
                dtype=self.config.mel_dtype,
            )
            for index, mel in enumerate(mels):
                audio_batch[index, :, : mel.shape[-1]] = mel
            seqlens_tensor = torch.tensor(raw_lengths, dtype=torch.long)
        else:
            audio_batch = None
            seqlens_tensor = None

        if self._AUDIO_SPAN_RE.search(prompt_text) is None:
            prompt_text = self._build_default_prompt(prompt_text, has_audio=bool(audio_list))
        input_ids_list = self._build_input_from_prompt(prompt_text, token_lens)

        input_ids_tensor = torch.tensor([input_ids_list], dtype=torch.long)
        data = {
            "input_ids": input_ids_tensor,
            "attention_mask": torch.ones_like(input_ids_tensor),
        }
        if audio_batch is not None:
            data["audio_data"] = audio_batch
            data["audio_data_seqlens"] = seqlens_tensor
        return BatchEncoding(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["MelConfig", "MossMusicProcessor"]
