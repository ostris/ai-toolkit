
from .caption import default_long_prompt, default_short_prompt, default_replacements, clean_caption

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Union, List
import tempfile
import os

img_ext = ['.jpg', '.jpeg', '.png', '.webp']
video_ext = ['.mp4', '.avi', '.mov', '.mkv', '.webm']


class JoyCaptionImageProcessor:
    def __init__(self, device='cuda', model_path=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.torch_dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        self.model_path = model_path or "fancyfeast/llama-joycaption-beta-one-hf-llava"

    def load_model(self, model_path=None):
        from transformers import AutoProcessor, LlavaForConditionalGeneration


        if model_path:
            self.model_path = model_path

        print(f"Loading JoyCaption model from {self.model_path} on {self.device}...")


        self.processor = AutoProcessor.from_pretrained(self.model_path)



        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == 'cuda' else None,
            attn_implementation="sdpa"  # Use torch.nn.functional.scaled_dot_product_attention
        )

        if self.device != 'cuda':
            self.model = self.model.to(self.device)

        self.model.eval()

        self.is_loaded = True
        print(f"JoyCaption model loaded successfully on {self.device}")

    def generate_caption(
        self,
        image: Image,
        prompt: str = None,
        replacements=default_replacements,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.9,
        system_prompt="You are a helpful image captioner."
    ):

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")


        if image.mode != 'RGB':
            image = image.convert('RGB')


        if prompt is None:
            prompt = "Write a long descriptive caption for this image in a formal tone."


        convo = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]




        convo_string = self.processor.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=True
        )
        assert isinstance(convo_string, str)


        inputs = self.processor(
            text=[convo_string],
            images=[image],
            return_tensors="pt"
        ).to(self.device)


        if self.device == 'cuda':
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)


        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=temperature,
                top_k=None,
                top_p=top_p,
            )[0]


        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]


        caption = self.processor.tokenizer.decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        caption = caption.strip()


        return clean_caption(caption, replacements=replacements)

    def extract_video_frames(
        self,
        video_path: str,
        num_frames: int = 8,
        sample_method: str = 'uniform'
    ) -> List[Image.Image]:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")

        frames = []

        if sample_method == 'uniform':

            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif sample_method == 'first':

            frame_indices = list(range(min(num_frames, total_frames)))
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

        cap.release()

        if not frames:
            raise ValueError(f"No frames could be extracted from video: {video_path}")

        print(f"Extracted {len(frames)} frames from video")
        return frames

    def generate_video_caption(
        self,
        video_path: str,
        num_frames: int = 8,
        sample_method: str = 'uniform',
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.9,
        combine_method: str = "first",
        prompt: str = None
    ) -> str:

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")


        frames = self.extract_video_frames(video_path, num_frames, sample_method)


        frame_captions = []
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}...")
            caption = self.generate_caption(
                image=frame,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            frame_captions.append(caption)


        if combine_method == "first":

            final_caption = frame_captions[0]
        elif combine_method == "longest":

            final_caption = max(frame_captions, key=len)
        elif combine_method == "combined":


            all_elements = set()
            for caption in frame_captions:
                elements = [e.strip() for e in caption.split(',')]
                all_elements.update(elements)
            final_caption = ', '.join(sorted(all_elements))
        else:
            final_caption = frame_captions[0]

        return final_caption

    def unload_model(self):
        if self.model is not None:
            self.model.to('cpu')
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        self.is_loaded = False


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Model unloaded and GPU memory freed")
