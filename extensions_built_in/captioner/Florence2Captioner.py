from transformers import AutoModelForCausalLM, AutoProcessor
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


VALID_FLORENCE2_TASKS = (
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
)


class Florence2Captioner(BaseCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Florence2Captioner, self).__init__(process_id, job, config, **kwargs)

    def load_model(self):
        self.print_and_status_update("Loading Florence-2 model")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.caption_config.model_name_or_path,
            dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map="cpu",
        )
        if not self.caption_config.low_vram:
            self.model.to(self.device_torch)
        if self.caption_config.quantize:
            self.print_and_status_update("Quantizing Florence-2 model")
            try:
                quantize(self.model, weights=get_qtype(self.caption_config.qtype))
                freeze(self.model)
                flush()
            except Exception as e:
                # Florence-2 ships custom modules via trust_remote_code that
                # quanto cannot always introspect; fall back to unquantized.
                print(f"Florence-2 quantization skipped: {e}")
                flush()
        self.processor = AutoProcessor.from_pretrained(
            self.caption_config.model_name_or_path,
            trust_remote_code=True,
        )
        if self.caption_config.low_vram:
            self.model.to(self.device_torch)
        flush()

    def _resolve_task(self) -> str:
        prompt = (self.caption_config.caption_prompt or "").strip()
        if prompt in VALID_FLORENCE2_TASKS:
            return prompt
        return "<MORE_DETAILED_CAPTION>"

    def get_caption_for_file(self, file_path: str) -> str:
        img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
        try:
            task = self._resolve_task()
            inputs = self.processor(
                text=task,
                images=img,
                return_tensors="pt",
            ).to(self.device_torch, self.torch_dtype)

            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.caption_config.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]

            parsed = self.processor.post_process_generation(
                generated_text,
                task=task,
                image_size=(img.width, img.height),
            )

            caption = parsed.get(task, "")
            if not isinstance(caption, str):
                caption = str(caption)
            return caption.strip()
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
