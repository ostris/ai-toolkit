try:
    from llava.model import LlavaLlamaForCausalLM
except ImportError:
    # print("You need to manually install llava -> pip install --no-deps  git+https://github.com/haotian-liu/LLaVA.git")
    print("You need to manually install llava -> pip install --no-deps  git+https://github.com/haotian-liu/LLaVA.git")
    raise

long_prompt = 'caption this image. describe every single thing in the image in detail. Do not include any unnecessary words in your description for the sake of good grammar. I want many short statements that serve the single purpose of giving the most thorough description if items as possible in the smallest, comma separated way possible. be sure to describe people\'s moods, clothing, the environment, lighting, colors, and everything.'
short_prompt = 'caption this image in less than ten words'

prompts = [
    long_prompt,
    short_prompt,
]

replacements = [
    ("the image features", ""),
    ("the image shows", ""),
    ("the image depicts", ""),
    ("the image is", ""),
]

import torch
from PIL import Image, ImageOps
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

img_ext = ['.jpg', '.jpeg', '.png', '.webp']


class LLaVAImageProcessor:
    def __init__(self, device='cuda'):
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

    def clean_caption(self, cap):
        # remove any newlines
        cap = cap.replace("\n", ", ")
        cap = cap.replace("\r", ", ")
        cap = cap.replace(".", ",")
        cap = cap.replace("\"", "")

        # remove unicode characters
        cap = cap.encode('ascii', 'ignore').decode('ascii')

        # make lowercase
        cap = cap.lower()
        # remove any extra spaces
        cap = " ".join(cap.split())

        for replacement in replacements:
            cap = cap.replace(replacement[0], replacement[1])

        cap_list = cap.split(",")
        # trim whitespace
        cap_list = [c.strip() for c in cap_list]
        # remove empty strings
        cap_list = [c for c in cap_list if c != ""]
        # remove duplicates
        cap_list = list(dict.fromkeys(cap_list))
        # join back together
        cap = ", ".join(cap_list)
        return cap

    def generate_caption(self, image: Image, prompt: str = long_prompt):
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
                max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria],
                top_p=0.9
            )
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        output = outputs.rsplit('</s>', 1)[0]
        return self.clean_caption(output)

    def generate_captions(self, image: Image):

        responses = []
        for prompt in prompts:
            output = self.generate_caption(image, prompt)
            responses.append(output)
        # replace all . with ,
        return responses
