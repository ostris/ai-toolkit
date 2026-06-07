from transformers import  CLIPImageProcessor, BitsAndBytesConfig, AutoTokenizer

from .caption import default_long_prompt, default_short_prompt, default_replacements, clean_caption
import torch
from PIL import Image


class FuyuImageProcessor:
    def __init__(self, device='cuda'):
        from transformers import FuyuProcessor, FuyuForCausalLM
        self.device = device
        self.model: FuyuForCausalLM = None
        self.processor: FuyuProcessor = None
        self.dtype = torch.bfloat16
        self.tokenizer: AutoTokenizer
        self.is_loaded = False

    def load_model(self):
        from transformers import FuyuProcessor, FuyuForCausalLM
        model_path = "adept/fuyu-8b"
        kwargs = {"device_map": self.device}
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        self.processor = FuyuProcessor.from_pretrained(model_path)
        self.model = FuyuForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        self.is_loaded = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = FuyuForCausalLM.from_pretrained(model_path, torch_dtype=self.dtype, **kwargs)
        self.processor = FuyuProcessor(image_processor=FuyuImageProcessor(), tokenizer=self.tokenizer)

    def generate_caption(
            self, image: Image,
            prompt: str = default_long_prompt,
            replacements=default_replacements,
            max_new_tokens=512
    ):
        # prepare inputs for the model
        # text_prompt = f"{prompt}\n"

        # image = image.convert('RGB')
        model_inputs = self.processor(text=prompt, images=[image])
        model_inputs = {k: v.to(dtype=self.dtype if torch.is_floating_point(v) else v.dtype, device=self.device) for k, v in
                        model_inputs.items()}

        generation_output = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        prompt_len = model_inputs["input_ids"].shape[-1]
        output = self.tokenizer.decode(generation_output[0][prompt_len:], skip_special_tokens=True)
        output = clean_caption(output, replacements=replacements)
        return output

        # inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        # for k, v in inputs.items():
        #     inputs[k] = v.to(self.device)

        # # autoregressively generate text
        # generation_output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # generation_text = self.processor.batch_decode(generation_output[:, -max_new_tokens:], skip_special_tokens=True)
        # output = generation_text[0]
        #
        # return clean_caption(output, replacements=replacements)
