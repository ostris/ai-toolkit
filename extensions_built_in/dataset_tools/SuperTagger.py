import copy
import json
import os
from collections import OrderedDict
import gc
import traceback
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from .tools.dataset_tools_config_modules import RAW_DIR, TRAIN_DIR, Step, ImgInfo
from .tools.fuyu_utils import FuyuImageProcessor
from .tools.image_tools import load_image, ImageProcessor, resize_to_max
from .tools.llava_utils import LLaVAImageProcessor
from .tools.caption import default_long_prompt, default_short_prompt, default_replacements
from jobs.process import BaseExtensionProcess
from .tools.sync_tools import get_img_paths

img_ext = ['.jpg', '.jpeg', '.png', '.webp', '.avif']


def flush():
    torch.cuda.empty_cache()
    gc.collect()


VERSION = 2


class SuperTagger(BaseExtensionProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        parent_dir = config.get('parent_dir', None)
        self.dataset_paths: list[str] = config.get('dataset_paths', [])
        self.device = config.get('device', 'cuda')
        self.steps: list[Step] = config.get('steps', [])
        self.caption_method = config.get('caption_method', 'llava:default')
        self.caption_prompt = config.get('caption_prompt', default_long_prompt)
        self.caption_short_prompt = config.get('caption_short_prompt', default_short_prompt)
        self.force_reprocess_img = config.get('force_reprocess_img', False)
        self.caption_replacements = config.get('caption_replacements', default_replacements)
        self.caption_short_replacements = config.get('caption_short_replacements', default_replacements)
        self.master_dataset_dict = OrderedDict()
        self.dataset_master_config_file = config.get('dataset_master_config_file', None)
        if parent_dir is not None and len(self.dataset_paths) == 0:
            # find all folders in the patent_dataset_path
            self.dataset_paths = [
                os.path.join(parent_dir, folder)
                for folder in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, folder))
            ]
        else:
            # make sure they exist
            for dataset_path in self.dataset_paths:
                if not os.path.exists(dataset_path):
                    raise ValueError(f"Dataset path does not exist: {dataset_path}")

        print(f"Found {len(self.dataset_paths)} dataset paths")

        self.image_processor: ImageProcessor = self.get_image_processor()

    def get_image_processor(self):
        if self.caption_method.startswith('llava'):
            return LLaVAImageProcessor(device=self.device)
        elif self.caption_method.startswith('fuyu'):
            return FuyuImageProcessor(device=self.device)
        else:
            raise ValueError(f"Unknown caption method: {self.caption_method}")

    def process_image(self, img_path: str):
        root_img_dir = os.path.dirname(os.path.dirname(img_path))
        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]
        train_dir = os.path.join(root_img_dir, TRAIN_DIR)
        train_img_path = os.path.join(train_dir, filename)
        json_path = os.path.join(train_dir, f"{filename_no_ext}.json")

        # check if json exists, if it does load it as image info
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                img_info = ImgInfo(**json.load(f))
        else:
            img_info = ImgInfo()

        # always send steps first in case other processes need them
        img_info.add_steps(copy.deepcopy(self.steps))
        img_info.set_version(VERSION)
        img_info.set_caption_method(self.caption_method)

        image: Image = None
        caption_image: Image = None

        did_update_image = False

        # trigger reprocess of steps
        if self.force_reprocess_img:
            img_info.trigger_image_reprocess()

        # set the image as updated if it does not exist on disk
        if not os.path.exists(train_img_path):
            did_update_image = True
            image = load_image(img_path, force_rgb=True)
        if img_info.force_image_process:
            did_update_image = True
            image = load_image(img_path, force_rgb=True)

        # go through the needed steps
        for step in copy.deepcopy(img_info.state.steps_to_complete):
            if step == 'caption':
                # load image
                if image is None:
                    image = load_image(img_path, force_rgb=True)
                if caption_image is None:
                    caption_image = resize_to_max(image, 1024, 1024)

                if not self.image_processor.is_loaded:
                    print('Loading Model. Takes a while, especially the first time')
                    self.image_processor.load_model()

                img_info.caption = self.image_processor.generate_caption(
                    image=caption_image,
                    prompt=self.caption_prompt,
                    replacements=self.caption_replacements
                )
                img_info.mark_step_complete(step)
            elif step == 'caption_short':
                # load image
                if image is None:
                    image = load_image(img_path, force_rgb=True)

                if caption_image is None:
                    caption_image = resize_to_max(image, 1024, 1024)

                if not self.image_processor.is_loaded:
                    print('Loading Model. Takes a while, especially the first time')
                    self.image_processor.load_model()
                img_info.caption_short = self.image_processor.generate_caption(
                    image=caption_image,
                    prompt=self.caption_short_prompt,
                    replacements=self.caption_short_replacements
                )
                img_info.mark_step_complete(step)
            elif step == 'contrast_stretch':
                # load image
                if image is None:
                    image = load_image(img_path, force_rgb=True)
                image = ImageOps.autocontrast(image, cutoff=(0.1, 0), preserve_tone=True)
                did_update_image = True
                img_info.mark_step_complete(step)
            else:
                raise ValueError(f"Unknown step: {step}")

        os.makedirs(os.path.dirname(train_img_path), exist_ok=True)
        if did_update_image:
            image.save(train_img_path)

        if img_info.is_dirty:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(img_info.to_dict(), f, indent=4, ensure_ascii=False)

        if self.dataset_master_config_file:
            # add to master dict
            self.master_dataset_dict[train_img_path] = img_info.to_dict()

    def run(self):
        super().run()
        imgs_to_process = []
        # find all images
        for dataset_path in self.dataset_paths:
            raw_dir = os.path.join(dataset_path, RAW_DIR)
            raw_image_paths = get_img_paths(raw_dir)
            for raw_image_path in raw_image_paths:
                imgs_to_process.append(raw_image_path)

        if len(imgs_to_process) == 0:
            print(f"No images to process")
        else:
            print(f"Found {len(imgs_to_process)} to process")

            for img_path in tqdm(imgs_to_process, desc="Processing images"):
                try:
                    self.process_image(img_path)
                except Exception:
                    # print full stack trace
                    print(traceback.format_exc())
                    continue
                # self.process_image(img_path)

        if self.dataset_master_config_file is not None:
            # save it as json
            with open(self.dataset_master_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.master_dataset_dict, f, indent=4, ensure_ascii=False)

        del self.image_processor
        flush()
