
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


class Florence2ImageProcessor:
    def __init__(self, device='cuda', model_path=None):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model_path = model_path or "microsoft/Florence-2-large-ft"

    def load_model(self, model_path=None):
        from transformers import AutoModelForCausalLM, AutoProcessor


        if model_path:
            self.model_path = model_path

        print(f"Loading Florence-2 model from {self.model_path} on {self.device}...")


        import warnings
        warnings.filterwarnings('ignore', category=SyntaxWarning)


        try:
            import timm
            from timm.models.davit import DaViT
            if not hasattr(DaViT, '_initialize_weights'):
                def _initialize_weights(self):

                    pass
                DaViT._initialize_weights = _initialize_weights
        except Exception:

            pass

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.is_loaded = True
        print(f"Florence-2 model loaded successfully on {self.device}")

    def generate_caption(
        self,
        image: Image,
        prompt: str = default_long_prompt,
        replacements=default_replacements,
        max_new_tokens=1024,
        num_beams=3,
        task="<DETAILED_CAPTION>"
    ):

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")


        if image.mode != 'RGB':
            image = image.convert('RGB')


        inputs = self.processor(
            text=task,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)


        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )


        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )


        caption_text = parsed_answer.get(task, "")


        caption_text = caption_text.replace("The image shows ", "")
        caption_text = caption_text.replace("The image depicts ", "")


        return clean_caption(caption_text, replacements=replacements)

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
        max_new_tokens: int = 1024,
        num_beams: int = 3,
        task: str = "<DETAILED_CAPTION>",
        combine_method: str = "first"
    ) -> str:

        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")


        frames = self.extract_video_frames(video_path, num_frames, sample_method)


        frame_captions = []
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}...")
            caption = self.generate_caption(
                image=frame,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                task=task
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
