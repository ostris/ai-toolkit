from collections import OrderedDict
from torch.utils.data import DataLoader
from toolkit.prompt_utils import concat_prompt_embeds, split_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
import gc
import torch
from jobs.process import BaseSDTrainProcess


def flush():
    torch.cuda.empty_cache()
    gc.collect()


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)

        # textual inversion
        # if self.embedding is not None:
            # set text encoder to train. Not sure if this is necessary but diffusers example did it
            # self.sd.text_encoder.train()

    def hook_train_loop(self, batch):
        dtype = get_torch_dtype(self.train_config.dtype)
        noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
        network_weight_list = batch.get_network_weight_list()

        self.optimizer.zero_grad()
        flush()

        # text encoding
        grad_on_text_encoder = False
        if self.train_config.train_text_encoder:
            grad_on_text_encoder = True

        if self.embedding:
            grad_on_text_encoder = True

        # have a blank network so we can wrap it in a context and set multipliers without checking every time
        if self.network is not None:
            network = self.network
        else:
            network = BlankNetwork()

        # set the weights
        network.multiplier = network_weight_list

        # activate network if it exits
        with network:
            with torch.set_grad_enabled(grad_on_text_encoder):
                conditional_embeds = self.sd.encode_prompt(conditioned_prompts).to(self.device_torch, dtype=dtype)
            if not grad_on_text_encoder:
                # detach the embeddings
                conditional_embeds = conditional_embeds.detach()
                self.optimizer.zero_grad()
                flush()

            noise_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                timestep=timesteps,
                guidance_scale=1.0,
            )
        # 9.18 gb
        noise = noise.to(self.device_torch, dtype=dtype)

        if self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
        else:
            target = noise

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean([1, 2, 3])

        if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
            # add min_snr_gamma
            loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        loss = loss.mean()

        # back propagate loss to free ram
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
        flush()

        # apply gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        if self.embedding is not None:
            # Let's make sure we don't update any embedding weights besides the newly added token
            self.embedding.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )
        # reset network multiplier
        network.multiplier = 1.0

        return loss_dict
