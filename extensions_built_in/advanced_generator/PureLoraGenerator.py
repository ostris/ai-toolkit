import os
from collections import OrderedDict

from toolkit.config_modules import ModelConfig, GenerateImageConfig, SampleConfig, LoRMConfig
from toolkit.lorm import ExtractMode, convert_diffusers_unet_to_lorm
from toolkit.sd_device_states_presets import get_train_sd_device_state_preset
from toolkit.stable_diffusion_model import StableDiffusion
import gc
import torch
from jobs.process import BaseExtensionProcess
from toolkit.train_tools import get_torch_dtype


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class PureLoraGenerator(BaseExtensionProcess):

    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.output_folder = self.get_conf('output_folder', required=True)
        self.device = self.get_conf('device', 'cuda')
        self.device_torch = torch.device(self.device)
        self.model_config = ModelConfig(**self.get_conf('model', required=True))
        self.generate_config = SampleConfig(**self.get_conf('sample', required=True))
        self.dtype = self.get_conf('dtype', 'float16')
        self.torch_dtype = get_torch_dtype(self.dtype)
        lorm_config = self.get_conf('lorm', None)
        self.lorm_config = LoRMConfig(**lorm_config) if lorm_config is not None else None

        self.device_state_preset = get_train_sd_device_state_preset(
            device=torch.device(self.device),
        )

        self.progress_bar = None
        self.sd = StableDiffusion(
            device=self.device,
            model_config=self.model_config,
            dtype=self.dtype,
        )

    def run(self):
        super().run()
        print("Loading model...")
        with torch.no_grad():
            self.sd.load_model()
            self.sd.unet.eval()
            self.sd.unet.to(self.device_torch)
            if isinstance(self.sd.text_encoder, list):
                for te in self.sd.text_encoder:
                    te.eval()
                    te.to(self.device_torch)
            else:
                self.sd.text_encoder.eval()
                self.sd.to(self.device_torch)

            print(f"Converting to LoRM UNet")
            # replace the unet with LoRMUnet
            convert_diffusers_unet_to_lorm(
                self.sd.unet,
                config=self.lorm_config,
            )

            sample_folder = os.path.join(self.output_folder)
            gen_img_config_list = []

            sample_config = self.generate_config
            start_seed = sample_config.seed
            current_seed = start_seed
            for i in range(len(sample_config.prompts)):
                if sample_config.walk_seed:
                    current_seed = start_seed + i

                filename = f"[time]_[count].{self.generate_config.ext}"
                output_path = os.path.join(sample_folder, filename)
                prompt = sample_config.prompts[i]
                extra_args = {}
                gen_img_config_list.append(GenerateImageConfig(
                    prompt=prompt,  # it will autoparse the prompt
                    width=sample_config.width,
                    height=sample_config.height,
                    negative_prompt=sample_config.neg,
                    seed=current_seed,
                    guidance_scale=sample_config.guidance_scale,
                    guidance_rescale=sample_config.guidance_rescale,
                    num_inference_steps=sample_config.sample_steps,
                    network_multiplier=sample_config.network_multiplier,
                    output_path=output_path,
                    output_ext=sample_config.ext,
                    adapter_conditioning_scale=sample_config.adapter_conditioning_scale,
                    **extra_args
                ))

            # send to be generated
            self.sd.generate_images(gen_img_config_list, sampler=sample_config.sampler)
            print("Done generating images")
            # cleanup
            del self.sd
            gc.collect()
            torch.cuda.empty_cache()
