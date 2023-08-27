import gc
import os
from collections import OrderedDict
from typing import ForwardRef

import torch
from safetensors.torch import save_file, load_file

from jobs.process.BaseProcess import BaseProcess
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_model_hash_to_meta, \
    add_base_model_info_to_meta
from toolkit.train_tools import get_torch_dtype


class ModRescaleLoraProcess(BaseProcess):
    process_id: int
    config: OrderedDict
    progress_bar: ForwardRef('tqdm') = None

    def __init__(
            self,
            process_id: int,
            job,
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        self.process_id: int
        self.config: OrderedDict
        self.progress_bar: ForwardRef('tqdm') = None
        self.input_path = self.get_conf('input_path', required=True)
        self.output_path = self.get_conf('output_path', required=True)
        self.replace_meta = self.get_conf('replace_meta', default=False)
        self.save_dtype = self.get_conf('save_dtype', default='fp16', as_type=get_torch_dtype)
        self.current_weight = self.get_conf('current_weight', required=True, as_type=float)
        self.target_weight = self.get_conf('target_weight', required=True, as_type=float)
        self.scale_target = self.get_conf('scale_target', default='up_down')  # alpha or up_down
        self.is_xl = self.get_conf('is_xl', default=False, as_type=bool)
        self.is_v2 = self.get_conf('is_v2', default=False, as_type=bool)

        self.progress_bar = None

    def run(self):
        super().run()
        source_state_dict = load_file(self.input_path)
        source_meta = load_metadata_from_safetensors(self.input_path)

        if self.replace_meta:
            self.meta.update(
                add_base_model_info_to_meta(
                    self.meta,
                    is_xl=self.is_xl,
                    is_v2=self.is_v2,
                )
            )
            save_meta = get_meta_for_safetensors(self.meta, self.job.name)
        else:
            save_meta = get_meta_for_safetensors(source_meta, self.job.name, add_software_info=False)

        # save
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        new_state_dict = OrderedDict()

        for key in list(source_state_dict.keys()):
            v = source_state_dict[key]
            v = v.detach().clone().to("cpu").to(get_torch_dtype('fp32'))

            # all loras have an alpha, up weight and down weight
            #  - "lora_te_text_model_encoder_layers_0_mlp_fc1.alpha",
            #  - "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
            #  - "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight",
            # we can rescale by adjusting the alpha or the up weights, or the up and down weights
            # I assume doing both up and down would be best all around, but I'm not sure
            # some locons also have mid weights, we will leave those alone for now, will work without them

            # when adjusting alpha, it is used to calculate the multiplier in a lora module
            #  - scale = alpha / lora_dim
            #  - output = layer_out + lora_up_out * multiplier * scale
            total_module_scale = torch.tensor(self.current_weight / self.target_weight) \
                .to("cpu", dtype=get_torch_dtype('fp32'))
            num_modules_layers = 2  # up and down
            up_down_scale = torch.pow(total_module_scale, 1.0 / num_modules_layers) \
                .to("cpu", dtype=get_torch_dtype('fp32'))
            # only update alpha
            if self.scale_target == 'alpha' and key.endswith('.alpha'):
                v = v * total_module_scale
            if self.scale_target == 'up_down' and key.endswith('.lora_up.weight') or key.endswith('.lora_down.weight'):
                # would it be better to adjust the up weights for fp16 precision? Doing both should reduce chance of NaN
                v = v * up_down_scale
            v = v.detach().clone().to("cpu").to(self.save_dtype)
            new_state_dict[key] = v

        save_meta = add_model_hash_to_meta(new_state_dict, save_meta)
        save_file(new_state_dict, self.output_path, save_meta)

        # cleanup incase there are other jobs
        del new_state_dict
        del source_state_dict
        del source_meta

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Saved to {self.output_path}")
