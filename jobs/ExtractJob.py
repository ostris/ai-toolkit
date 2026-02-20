from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
from collections import OrderedDict
from jobs import BaseJob
from toolkit.train_tools import get_torch_dtype

process_dict = {
    'locon': 'ExtractLoconProcess',
    'lora': 'ExtractLoraProcess',
}


class ExtractJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.base_model_path = self.get_conf('base_model', required=True)
        self.model_base = None
        self.model_base_text_encoder = None
        self.model_base_vae = None
        self.model_base_unet = None
        self.extract_model_path = self.get_conf('extract_model', required=True)
        self.model_extract = None
        self.model_extract_text_encoder = None
        self.model_extract_vae = None
        self.model_extract_unet = None
        self.extract_unet = self.get_conf('extract_unet', True)
        self.extract_text_encoder = self.get_conf('extract_text_encoder', True)
        self.dtype = self.get_conf('dtype', 'fp16')
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.output_folder = self.get_conf('output_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.device = self.get_conf('device', 'cpu')

        # loads the processes from the config
        self.load_processes(process_dict)

    def run(self):
        super().run()
        # load models
        print(f"Loading models for extraction")
        print(f" - Loading base model: {self.base_model_path}")
        # (text_model, vae, unet)
        self.model_base = load_models_from_stable_diffusion_checkpoint(self.is_v2, self.base_model_path)
        self.model_base_text_encoder = self.model_base[0]
        self.model_base_vae = self.model_base[1]
        self.model_base_unet = self.model_base[2]

        print(f" - Loading extract model: {self.extract_model_path}")
        self.model_extract = load_models_from_stable_diffusion_checkpoint(self.is_v2, self.extract_model_path)
        self.model_extract_text_encoder = self.model_extract[0]
        self.model_extract_vae = self.model_extract[1]
        self.model_extract_unet = self.model_extract[2]

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
