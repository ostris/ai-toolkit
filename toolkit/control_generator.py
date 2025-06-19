import gc
import math
import os
import torch
from typing import Literal
from PIL import Image, ImageFilter, ImageOps
import pillow_avif
from extensions_built_in.dataset_tools.tools.image_tools import load_image
from tqdm import tqdm

from torchvision import transforms

# supress all warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def flush(garbage_collect=True):
    torch.cuda.empty_cache()
    if garbage_collect:
        gc.collect()


ControlTypes = Literal['depth', 'pose', 'line', 'inpaint', 'mask']

img_ext_list = ['.jpg', '.jpeg', '.png', '.webp', '.avif']


class ControlGenerator:
    def __init__(self, device, sd=None):
        self.device = device
        self.sd = sd  # optional. It will unload the model if not None
        self.has_unloaded = False
        self.control_depth_model = None
        self.control_pose_model = None
        self.control_line_model = None
        self.control_bg_remover = None
        self.debug = False
        self.regen = False

    def get_control_path(self, img_path, control_type: ControlTypes):
        if self.regen:
            return self._generate_control(img_path, control_type)
        coltrols_folder = os.path.join(os.path.dirname(img_path), '_controls')
        file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        file_name_no_ext_control = f"{file_name_no_ext}.{control_type}"
        for ext in img_ext_list:
            possible_path = os.path.join(
                coltrols_folder, file_name_no_ext_control + ext)
            if os.path.exists(possible_path):
                return possible_path
        # if we get here, we need to generate the control
        return self._generate_control(img_path, control_type)

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _generate_control(self, img_path, control_type):
        device = self.device
        image: Image = None

        coltrols_folder = os.path.join(os.path.dirname(img_path), '_controls')
        file_name_no_ext = os.path.splitext(os.path.basename(img_path))[0]

        # we need to generate the control. Unload model if not unloaded
        if not self.has_unloaded:
            if self.sd is not None:
                print("Unloading model to generate controls")
                self.sd.set_device_state_preset('unload')
            self.has_unloaded = True

        if image is None:
            # make sure image is loaded if we havent loaded it with another control
            image = load_image(img_path, force_rgb=True)

            # resize to a max of 1mp
            max_size = 1024 * 1024

            w, h = image.size
            if w * h > max_size:
                scale = math.sqrt(max_size / (w * h))
                w = int(w * scale)
                h = int(h * scale)
                image = image.resize((w, h), Image.BICUBIC)

        save_path = os.path.join(
            coltrols_folder, f"{file_name_no_ext}.{control_type}.jpg")
        os.makedirs(coltrols_folder, exist_ok=True)
        if control_type == 'depth':
            self.debug_print("Generating depth control")
            if self.control_depth_model is None:
                from transformers import pipeline
                self.control_depth_model = pipeline(
                    task="depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Large-hf",
                    device=device,
                    torch_dtype=torch.float16
                )
            img = image.copy()
            in_size = img.size
            output = self.control_depth_model(img)
            out_tensor = output["predicted_depth"]  # shape (1, H, W) 0 - 255
            out_tensor = out_tensor.clamp(0, 255)
            out_tensor = out_tensor.squeeze(0).cpu().numpy()
            img = Image.fromarray(out_tensor.astype('uint8'))
            img = img.resize(in_size, Image.LANCZOS)
            img.save(save_path)
            return save_path
        elif control_type == 'pose':
            self.debug_print("Generating pose control")
            if self.control_pose_model is None:
                try:
                    import onnxruntime
                    onnxruntime.set_default_logger_severity(3)
                except ImportError:
                    raise ImportError(
                        "onnxruntime is not installed. Please install it with pip install onnxruntime or onnxruntime-gpu")
                try:
                    from easy_dwpose import DWposeDetector
                    self.control_pose_model = DWposeDetector(
                        device=str(device))
                except ImportError:
                    raise ImportError(
                        "easy-dwpose is not installed. Please install it with pip install easy-dwpose")
            img = image.copy()

            detect_res = int(math.sqrt(img.size[0] * img.size[1]))
            img = self.control_pose_model(
                img, output_type="pil", include_hands=True, include_face=True, detect_resolution=detect_res)
            img = img.convert('RGB')
            img.save(save_path)
            return save_path

        elif control_type == 'line':
            self.debug_print("Generating line control")
            if self.control_line_model is None:
                from controlnet_aux import TEEDdetector
                self.control_line_model = TEEDdetector.from_pretrained(
                    "fal-ai/teed", filename="5_model.pth").to(device)
            img = image.copy()
            img = self.control_line_model(img, detect_resolution=1024)
            # apply threshold
            # img = img.filter(ImageFilter.GaussianBlur(radius=1))
            img = img.point(lambda p: p > 128 and 255)
            img = img.convert('RGB')
            img.save(save_path)
            return save_path
        elif control_type == 'inpaint' or control_type == 'mask':
            self.debug_print("Generating inpaint/mask control")
            img = image.copy()
            if self.control_bg_remover is None:
                from transformers import AutoModelForImageSegmentation
                self.control_bg_remover = AutoModelForImageSegmentation.from_pretrained(
                    'ZhengPeng7/BiRefNet_HR',
                    trust_remote_code=True,
                    revision="595e212b3eaa6a1beaad56cee49749b1e00b1596",
                    torch_dtype=torch.float16
                ).to(device)
                self.control_bg_remover.eval()

            image_size = (1024, 1024)
            transform_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ])

            input_images = transform_image(img).unsqueeze(
                0).to('cuda').to(torch.float16)

            # Prediction
            preds = self.control_bg_remover(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(img.size)
            if control_type == 'inpaint':
                # inpainting feature currently only supports "erased" section desired to inpaint
                mask = ImageOps.invert(mask)
                img.putalpha(mask)
                save_path = os.path.join(
                    coltrols_folder, f"{file_name_no_ext}.{control_type}.webp")
            else:
                img = mask
                img = img.convert('RGB')
            img.save(save_path)
            return save_path
        else:
            raise Exception(f"Error: unknown control type {control_type}")

    def cleanup(self):
        if self.control_depth_model is not None:
            self.control_depth_model = None
        if self.control_pose_model is not None:
            self.control_pose_model = None
        if self.control_line_model is not None:
            self.control_line_model = None
        if self.control_bg_remover is not None:
            self.control_bg_remover = None
        if self.sd is not None and self.has_unloaded:
            self.sd.restore_device_state()
        self.has_unloaded = False

        flush()


if __name__ == "__main__":
    import sys
    import argparse
    import time
    import transformers
    transformers.logging.set_verbosity_error()

    control_times = {
        'depth': 0,
        'pose': 0,
        'line': 0,
        'inpaint': 0,
        'mask': 0
    }

    controls = control_times.keys()

    parser = argparse.ArgumentParser(description="Generate control images")
    parser.add_argument("img_dir", type=str, help="Path to image directory")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode")
    parser.add_argument('--regen', action='store_true',
                        help="Regenerate all controls")

    args = parser.parse_args()
    img_dir = args.img_dir
    if not os.path.exists(img_dir):
        print(f"Error: {img_dir} does not exist")
        exit()
    if not os.path.isdir(img_dir):
        print(f"Error: {img_dir} is not a directory")
        exit()

    # find images
    img_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if "_controls" in root:
                continue
            if file.startswith('.'):
                continue
            if file.lower().endswith(tuple(img_ext_list)):
                img_list.append(os.path.join(root, file))
    if len(img_list) == 0:
        print(f"Error: no images found in {img_dir}")
        exit()

    # load model
    idx = 0
    for img_path in tqdm(img_list):
        for control in controls:
            start = time.time()
            control_gen = ControlGenerator(torch.device('cuda'))
            control_gen.debug = args.debug
            control_gen.regen = args.regen
            control_path = control_gen.get_control_path(img_path, control)
            end = time.time()
            # dont track for first 2 images
            if idx < 2:
                continue
            control_times[control] += end - start
        idx += 1

    # determine avgt time
    for control in controls:
        control_times[control] /= (idx - 2)
        print(
            f"Avg time for {control} control: {control_times[control]:.2f} seconds")

    print("Done")
