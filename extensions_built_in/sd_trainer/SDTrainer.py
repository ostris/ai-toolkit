import os.path
from collections import OrderedDict

from PIL import Image
from diffusers import T2IAdapter
from torch.utils.data import DataLoader

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.ip_adapter import IPAdapter
from toolkit.prompt_utils import concat_prompt_embeds, split_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms


def flush():
    torch.cuda.empty_cache()
    gc.collect()

adapter_transforms = transforms.Compose([
    # transforms.PILToTensor(),
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()

    def get_adapter_images(self, batch: 'DataLoaderBatchDTO'):
        if self.adapter_config.image_dir is None:
            # adapter needs 0 to 1 values, batch is -1 to 1
            adapter_batch = batch.tensor.clone().to(
                self.device_torch, dtype=get_torch_dtype(self.train_config.dtype)
            )
            adapter_batch = (adapter_batch + 1) / 2
            return adapter_batch
        img_ext_list = ['.jpg', '.jpeg', '.png', '.webp']
        adapter_folder_path = self.adapter_config.image_dir
        adapter_images = []
        # loop through images
        for file_item in batch.file_items:
            img_path = file_item.path
            file_name_no_ext = os.path.basename(img_path).split('.')[0]
            # find the image
            for ext in img_ext_list:
                if os.path.exists(os.path.join(adapter_folder_path, file_name_no_ext + ext)):
                    adapter_images.append(os.path.join(adapter_folder_path, file_name_no_ext + ext))
                    break
        width, height = batch.file_items[0].crop_width, batch.file_items[0].crop_height
        adapter_tensors = []
        # load images with torch transforms
        for idx, adapter_image in enumerate(adapter_images):
            img = Image.open(adapter_image)
            # resize to match batch shape
            img = img.resize((width, height))
            img = adapter_transforms(img)
            adapter_tensors.append(img)

        # stack them
        adapter_tensors = torch.stack(adapter_tensors).to(
            self.device_torch, dtype=get_torch_dtype(self.train_config.dtype)
        )
        return adapter_tensors

    def hook_train_loop(self, batch):

        dtype = get_torch_dtype(self.train_config.dtype)
        noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
        network_weight_list = batch.get_network_weight_list()

        adapter_images = None
        sigmas = None
        if self.adapter:
            # todo move this to data loader
            adapter_images = self.get_adapter_images(batch)
            # not 100% sure what this does. But they do it here
            # https://github.com/huggingface/diffusers/blob/38a664a3d61e27ab18cd698231422b3c38d6eebf/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1170
            # sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
            # noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

        # flush()
        self.optimizer.zero_grad()

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
            # flush()
            pred_kwargs = {}
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                down_block_additional_residuals = self.adapter(adapter_images)
                down_block_additional_residuals = [
                    sample.to(dtype=dtype) for sample in down_block_additional_residuals
                ]
                pred_kwargs['down_block_additional_residuals'] = down_block_additional_residuals

            if self.adapter and isinstance(self.adapter, IPAdapter):
                with torch.no_grad():
                    conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(adapter_images)
                conditional_embeds = self.adapter(conditional_embeds, conditional_clip_embeds)


            noise_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                timestep=timesteps,
                guidance_scale=1.0,
                **pred_kwargs
            )

            # if self.adapter:
            #     # todo, diffusers does this on t2i training, is it better approach?
            #     # Denoise the latents
            #     denoised_latents = noise_pred * (-sigmas) + noisy_latents
            #     weighing = sigmas ** -2.0
            #
            #     # Get the target for loss depending on the prediction type
            #     if self.sd.noise_scheduler.config.prediction_type == "epsilon":
            #         target = batch.latents  # we are computing loss against denoise latents
            #     elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
            #         target = self.sd.noise_scheduler.get_velocity(batch.latents, noise, timesteps)
            #     else:
            #         raise ValueError(f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}")
            #
            #     # MSE loss
            #     loss = torch.mean(
            #         (weighing.float() * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            #         dim=1,
            #     )
            # else:
            noise = noise.to(self.device_torch, dtype=dtype).detach()
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
            # check if nan
            if torch.isnan(loss):
                raise ValueError("loss is nan")

            # IMPORTANT if gradient checkpointing do not leave with network when doing backward
            # it will destroy the gradients. This is because the network is a context manager
            # and will change the multipliers back to 0.0 when exiting. They will be
            # 0.0 for the backward pass and the gradients will be 0.0
            # I spent weeks on fighting this. DON'T DO IT
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            # flush()

        # apply gradients
        self.optimizer.step()
        self.lr_scheduler.step()

        if self.embedding is not None:
            # Let's make sure we don't update any embedding weights besides the newly added token
            self.embedding.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        return loss_dict
