import torch
import gc
from collections import OrderedDict
from typing import TYPE_CHECKING
from jobs.process import BaseExtensionProcess
from toolkit.config_modules import ModelConfig
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
from tqdm import tqdm

# Type check imports. Prevents circular imports
if TYPE_CHECKING:
    from jobs import ExtensionJob


# extend standard config classes to add weight
class ModelInputConfig(ModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = kwargs.get('weight', 1.0)
        # overwrite default dtype unless user specifies otherwise
        # float 32 will give up better precision on the merging functions
        self.dtype: str = kwargs.get('dtype', 'float32')


def flush():
    torch.cuda.empty_cache()
    gc.collect()


# this is our main class process
class ExampleMergeModels(BaseExtensionProcess):
    def __init__(
            self,
            process_id: int,
            job: 'ExtensionJob',
            config: OrderedDict
    ):
        super().__init__(process_id, job, config)
        # this is the setup process, do not do process intensive stuff here, just variable setup and
        # checking requirements. This is called before the run() function
        # no loading models or anything like that, it is just for setting up the process
        # all of your process intensive stuff should be done in the run() function
        # config will have everything from the process item in the config file

        # convince methods exist on BaseProcess to get config values
        # if required is set to true and the value is not found it will throw an error
        # you can pass a default value to get_conf() as well if it was not in the config file
        # as well as a type to cast the value to
        self.save_path = self.get_conf('save_path', required=True)
        self.save_dtype = self.get_conf('save_dtype', default='float16', as_type=get_torch_dtype)
        self.device = self.get_conf('device', default='cpu', as_type=torch.device)

        # build models to merge list
        models_to_merge = self.get_conf('models_to_merge', required=True, as_type=list)
        # build list of ModelInputConfig objects. I find it is a good idea to make a class for each config
        # this way you can add methods to it and it is easier to read and code. There are a lot of
        # inbuilt config classes located in toolkit.config_modules as well
        self.models_to_merge = [ModelInputConfig(**model) for model in models_to_merge]
        # setup is complete. Don't load anything else here, just setup variables and stuff

    # this is the entire run process be sure to call super().run() first
    def run(self):
        # always call first
        super().run()
        print(f"Running process: {self.__class__.__name__}")

        # let's adjust our weights first to normalize them so the total is 1.0
        total_weight = sum([model.weight for model in self.models_to_merge])
        weight_adjust = 1.0 / total_weight
        for model in self.models_to_merge:
            model.weight *= weight_adjust

        output_model: StableDiffusion = None
        # let's do the merge, it is a good idea to use tqdm to show progress
        for model_config in tqdm(self.models_to_merge, desc="Merging models"):
            # setup model class with our helper class
            sd_model = StableDiffusion(
                device=self.device,
                model_config=model_config,
                dtype="float32"
            )
            # load the model
            sd_model.load_model()

            # adjust the weight of the text encoder
            if isinstance(sd_model.text_encoder, list):
                # sdxl model
                for text_encoder in sd_model.text_encoder:
                    for key, value in text_encoder.state_dict().items():
                        value *= model_config.weight
            else:
                # normal model
                for key, value in sd_model.text_encoder.state_dict().items():
                    value *= model_config.weight
            # adjust the weights of the unet
            for key, value in sd_model.unet.state_dict().items():
                value *= model_config.weight

            if output_model is None:
                # use this one as the base
                output_model = sd_model
            else:
                # merge the models
                # text encoder
                if isinstance(output_model.text_encoder, list):
                    # sdxl model
                    for i, text_encoder in enumerate(output_model.text_encoder):
                        for key, value in text_encoder.state_dict().items():
                            value += sd_model.text_encoder[i].state_dict()[key]
                else:
                    # normal model
                    for key, value in output_model.text_encoder.state_dict().items():
                        value += sd_model.text_encoder.state_dict()[key]
                # unet
                for key, value in output_model.unet.state_dict().items():
                    value += sd_model.unet.state_dict()[key]

                # remove the model to free memory
                del sd_model
                flush()

        # merge loop is done, let's save the model
        print(f"Saving merged model to {self.save_path}")
        output_model.save(self.save_path, meta=self.meta, save_dtype=self.save_dtype)
        print(f"Saved merged model to {self.save_path}")
        # do cleanup here
        del output_model
        flush()
