
from .caption import default_long_prompt, default_short_prompt, default_replacements, clean_caption

import torch
from PIL import Image, ImageOps

from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

img_ext = ['.jpg', '.jpeg', '.png', '.webp', '.avif']


class LLaVAImageProcessor:
    def __init__(self, device='cuda'):
        try:
            from llava.model import LlavaLlamaForCausalLM
        except ImportError:
            # print("You need to manually install llava -> pip install --no-deps  git+https://github.com/haotian-liu/LLaVA.git")
            print(
                "You need to manually install llava -> pip install --no-deps  git+https://github.com/haotian-liu/LLaVA.git")
            raise
        self.device = device
        self.model: LlavaLlamaForCausalLM = None
        self.tokenizer: AutoTokenizer = None
        self.image_processor: CLIPImageProcessor = None
        self.is_loaded = False

    def load_model(self):
        from llava.model import LlavaLlamaForCausalLM

        model_path = "4bit/llava-v1.5-13b-3GB"
        # kwargs = {"device_map": "auto"}
        kwargs = {"device_map": self.device}
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        self.model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=self.device)
        self.image_processor = vision_tower.image_processor
        self.is_loaded = True

    def generate_caption(
            self, image:
            Image, prompt: str = default_long_prompt,
            replacements=default_replacements,
            max_new_tokens=512
    ):
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.utils import disable_torch_init
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        # question = "how many dogs are in the picture?"
        disable_torch_init()
        conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        image_tensor = self.image_processor.preprocess([image], return_tensors='pt')['pixel_values'].half().cuda()

        inp = f"{roles[0]}: {prompt}"
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        raw_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(raw_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, do_sample=True, temperature=0.1,
                max_new_tokens=max_new_tokens, use_cache=True, stopping_criteria=[stopping_criteria],
                top_p=0.8
            )
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        output = outputs.rsplit('</s>', 1)[0]
        return clean_caption(output, replacements=replacements)
