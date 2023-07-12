# from jobs import BaseJob
# from toolkit.kohya_model_util import load_models_from_stable_diffusion_checkpoint
# from collections import OrderedDict
# from typing import List
# from jobs.process import BaseExtractProcess, TrainFineTuneProcess
# import gc
# import time
# import argparse
# import itertools
# import math
# import os
# from multiprocessing import Value
#
# from tqdm import tqdm
# import torch
# from accelerate.utils import set_seed
# from accelerate import Accelerator
# import diffusers
# from diffusers import DDPMScheduler
#
# from toolkit.paths import SD_SCRIPTS_ROOT
#
# import sys
#
# sys.path.append(SD_SCRIPTS_ROOT)
#
# import library.train_util as train_util
# import library.config_util as config_util
# from library.config_util import (
#     ConfigSanitizer,
#     BlueprintGenerator,
# )
# import toolkit.train_tools as train_tools
# import library.custom_train_functions as custom_train_functions
# from library.custom_train_functions import (
#     apply_snr_weight,
#     get_weighted_text_embeddings,
#     prepare_scheduler_for_custom_training,
#     pyramid_noise_like,
#     apply_noise_offset,
#     scale_v_prediction_loss_like_noise_prediction,
# )
#
# process_dict = {
#     'fine_tine': 'TrainFineTuneProcess'
# }
#
#
# class TrainJob(BaseJob):
#     process: List[BaseExtractProcess]
#
#     def __init__(self, config: OrderedDict):
#         super().__init__(config)
#         self.base_model_path = self.get_conf('base_model', required=True)
#         self.base_model = None
#         self.training_folder = self.get_conf('training_folder', required=True)
#         self.is_v2 = self.get_conf('is_v2', False)
#         self.device = self.get_conf('device', 'cpu')
#         self.gradient_accumulation_steps = self.get_conf('gradient_accumulation_steps', 1)
#         self.mixed_precision = self.get_conf('mixed_precision', False)  # fp16
#         self.logging_dir = self.get_conf('logging_dir', None)
#
#         # loads the processes from the config
#         self.load_processes(process_dict)
#
#         # setup accelerator
#         self.accelerator = Accelerator(
#             gradient_accumulation_steps=self.gradient_accumulation_steps,
#             mixed_precision=self.mixed_precision,
#             log_with=None if self.logging_dir is None else 'tensorboard',
#             logging_dir=self.logging_dir,
#         )
#
#     def run(self):
#         super().run()
#         # load models
#         print(f"Loading base model for training")
#         print(f" - Loading base model: {self.base_model_path}")
#         self.base_model = load_models_from_stable_diffusion_checkpoint(self.is_v2, self.base_model_path)
#
#         print("")
#         print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")
#
#         for process in self.process:
#             process.run()
