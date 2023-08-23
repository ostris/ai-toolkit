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
    sd: StableDiffusion
    data_loader: DataLoader = None

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        pass

    def before_model_load(self):
        pass

    def hook_before_train_loop(self):
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)

        # textual inversion
        if self.embedding is not None:
            # keep original embeddings as reference
            self.orig_embeds_params = self.sd.text_encoder.get_input_embeddings().weight.data.clone()
            # set text encoder to train. Not sure if this is necessary but diffusers example did it
            self.sd.text_encoder.train()

    def hook_train_loop(self, batch):
        with torch.no_grad():
            imgs, prompts, dataset_config = batch

            # convert the 0 or 1 for is reg to a bool list
            is_reg_list = dataset_config.get('is_reg', [0 for _ in range(imgs.shape[0])])
            if isinstance(is_reg_list, torch.Tensor):
                is_reg_list = is_reg_list.numpy().tolist()
            is_reg_list = [bool(x) for x in is_reg_list]

            conditioned_prompts = []

            for prompt, is_reg in zip(prompts, is_reg_list):

                # make sure the embedding is in the prompts
                if self.embedding is not None:
                    prompt = self.embedding.inject_embedding_to_prompt(
                        prompt,
                        expand_token=True,
                        add_if_not_present=True,
                    )

                # make sure trigger is in the prompts if not a regularization run
                if self.trigger_word is not None and not is_reg:
                    prompt = self.sd.inject_trigger_into_prompt(
                        prompt,
                        add_if_not_present=True,
                    )
                conditioned_prompts.append(prompt)

            batch_size = imgs.shape[0]

            dtype = get_torch_dtype(self.train_config.dtype)
            imgs = imgs.to(self.device_torch, dtype=dtype)
            latents = self.sd.encode_images(imgs)

            noise_scheduler = self.sd.noise_scheduler
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler

            self.sd.noise_scheduler.set_timesteps(
                self.train_config.max_denoising_steps, device=self.device_torch
            )

            timesteps = torch.randint(0, self.train_config.max_denoising_steps, (batch_size,), device=self.device_torch)
            timesteps = timesteps.long()

            # get noise
            noise = self.sd.get_latent_noise(
                pixel_height=imgs.shape[2],
                pixel_width=imgs.shape[3],
                batch_size=batch_size,
                noise_offset=self.train_config.noise_offset
            ).to(self.device_torch, dtype=dtype)

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # remove grads for these
            noisy_latents.requires_grad = False
            noise.requires_grad = False

        flush()

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

        # activate network if it exits
        with network:
            with torch.set_grad_enabled(grad_on_text_encoder):
                embedding_list = []
                # embed the prompts
                for prompt in conditioned_prompts:
                    embedding = self.sd.encode_prompt(prompt).to(self.device_torch, dtype=dtype)
                    embedding_list.append(embedding)
                conditional_embeds = concat_prompt_embeds(embedding_list)

            noise_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                timestep=timesteps,
                guidance_scale=1.0,
            )

        noise = noise.to(self.device_torch, dtype=dtype)

        if self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = noise_scheduler.get_velocity(noisy_latents, noise, timesteps)
        else:
            target = noise

        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
        loss = loss.mean([1, 2, 3])

        if self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
            # add min_snr_gamma
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, self.train_config.min_snr_gamma)

        loss = loss.mean()

        # back propagate loss to free ram
        loss.backward()
        flush()

        # apply gradients
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        if self.embedding is not None:
            # Let's make sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(self.sd.tokenizer),), dtype=torch.bool)
            index_no_updates[
            min(self.embedding.placeholder_token_ids): max(self.embedding.placeholder_token_ids) + 1] = False
            with torch.no_grad():
                self.sd.text_encoder.get_input_embeddings().weight[
                    index_no_updates
                ] = self.orig_embeds_params[index_no_updates]

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        return loss_dict
