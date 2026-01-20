from typing import Optional

import torch

DEFAULT_AUDIO_MODEL_ID = "mispeech/midashenglm-7b-0804-fp8"
DEFAULT_AUDIO_PROMPT = (
    "Provide a concise but detailed caption of the audio. "
    "Describe any speech (content summary if clear), speaker count, "
    "perceived gender/age, tone/emotion, language/accent, and notable non-speech sounds "
    "such as music (genre/instruments/tempo/mood) or environmental noises. "
    "If there is no speech, focus on the audio events and atmosphere."
)


class AudioCaptioner:
    def __init__(self, device: str = "cuda", model_path: Optional[str] = None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path or DEFAULT_AUDIO_MODEL_ID
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self, model_path: Optional[str] = None) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        if model_path:
            self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=self.device if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def generate_caption(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        num_beams: int = 1,
        do_sample: bool = True,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt_text = prompt or DEFAULT_AUDIO_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]

        model_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
        ).to(device=self.model.device, dtype=self.model.dtype)

        input_ids = model_inputs.get("input_ids")
        prompt_length = input_ids.shape[1] if input_ids is not None else 0

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
        }
        if repetition_penalty is not None:
            generation_kwargs["repetition_penalty"] = repetition_penalty
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            generation = self.model.generate(**model_inputs, **generation_kwargs)

        if prompt_length and generation.shape[1] > prompt_length:
            generation = generation[:, prompt_length:]

        decoded = self.tokenizer.batch_decode(generation, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def unload_model(self) -> None:
        if self.model is not None:
            if self.device == "cuda":
                self.model.to("cpu")
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
