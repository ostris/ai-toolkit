from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from collections import OrderedDict

from optimum.quanto import freeze
from toolkit.basic import flush
from toolkit.util.quantize import quantize, get_qtype

from .BaseCaptioner import BaseCaptioner
import transformers
import logging
import warnings

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


class Qwen3VLCaptioner(BaseCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Qwen3VLCaptioner, self).__init__(process_id, job, config, **kwargs)

    def load_model(self):
        self.print_and_status_update("Loading Qwen3VL model")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.caption_config.model_name_or_path,
            dtype=self.torch_dtype,
            device_map="cpu",
        )
        if not self.caption_config.low_vram:
            self.model.to(self.device_torch)
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing Qwen3VL model")
            quantize(self.model, weights=get_qtype(self.caption_config.qtype))
            freeze(self.model)
            flush()
        self.processor = AutoProcessor.from_pretrained(
            self.caption_config.model_name_or_path
        )
        if self.caption_config.low_vram:
            self.model.to(self.device_torch)
        flush()

    def get_caption_for_file(self, file_path: str) -> str:
        img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": self.caption_config.caption_prompt},
                    ],
                }
            ]

            # Preparation for inference
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device_torch)

            # Inference: Generation of the output
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=self.caption_config.max_new_tokens
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            return output_text[0].strip()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
