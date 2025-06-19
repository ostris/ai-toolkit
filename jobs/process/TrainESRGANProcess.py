import copy
import glob
import os
import time
from collections import OrderedDict
from typing import List, Optional

from PIL import Image
import pillow_avif
from extensions_built_in.dataset_tools.tools.image_tools import load_image

from toolkit.basic import flush
from toolkit.models.RRDB import RRDBNet as ESRGAN, esrgan_safetensors_keys
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader, ConcatDataset
import torch
from torch import nn
from torchvision.transforms import transforms

from jobs.process import BaseTrainProcess
from toolkit.data_loader import AugmentedImageDataset
from toolkit.esrgan_utils import convert_state_dict_to_basicsr, convert_basicsr_state_dict_to_save_format
from toolkit.losses import ComparativeTotalVariation, get_gradient_penalty, PatternLoss
from toolkit.metadata import get_meta_for_safetensors
from toolkit.optimizer import get_optimizer
from toolkit.style import get_style_model_and_losses
from toolkit.train_tools import get_torch_dtype
from diffusers import AutoencoderKL
from tqdm import tqdm
import time
import numpy as np
from .models.vgg19_critic import Critic

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ]
)


class TrainESRGANProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.data_loader = None
        self.model: ESRGAN = None
        self.device = self.get_conf('device', self.job.device)
        self.pretrained_path = self.get_conf('pretrained_path', 'None')
        self.datasets_objects = self.get_conf('datasets', required=True)
        self.batch_size = self.get_conf('batch_size', 1, as_type=int)
        self.resolution = self.get_conf('resolution', 256, as_type=int)
        self.learning_rate = self.get_conf('learning_rate', 1e-6, as_type=float)
        self.sample_every = self.get_conf('sample_every', None)
        self.optimizer_type = self.get_conf('optimizer', 'adam')
        self.epochs = self.get_conf('epochs', None, as_type=int)
        self.max_steps = self.get_conf('max_steps', None, as_type=int)
        self.save_every = self.get_conf('save_every', None)
        self.upscale_sample = self.get_conf('upscale_sample', 4)
        self.dtype = self.get_conf('dtype', 'float32')
        self.sample_sources = self.get_conf('sample_sources', None)
        self.log_every = self.get_conf('log_every', 100, as_type=int)
        self.style_weight = self.get_conf('style_weight', 0, as_type=float)
        self.content_weight = self.get_conf('content_weight', 0, as_type=float)
        self.mse_weight = self.get_conf('mse_weight', 1e0, as_type=float)
        self.zoom = self.get_conf('zoom', 4, as_type=int)
        self.tv_weight = self.get_conf('tv_weight', 1e0, as_type=float)
        self.critic_weight = self.get_conf('critic_weight', 1, as_type=float)
        self.pattern_weight = self.get_conf('pattern_weight', 1, as_type=float)
        self.optimizer_params = self.get_conf('optimizer_params', {})
        self.augmentations = self.get_conf('augmentations', {})
        self.torch_dtype = get_torch_dtype(self.dtype)
        if self.torch_dtype == torch.bfloat16:
            self.esrgan_dtype = torch.float32
        else:
            self.esrgan_dtype = torch.float32

        self.vgg_19 = None
        self.style_weight_scalers = []
        self.content_weight_scalers = []

        # throw error if zoom if not divisible by 2
        if self.zoom % 2 != 0:
            raise ValueError('zoom must be divisible by 2')

        self.step_num = 0
        self.epoch_num = 0

        self.use_critic = self.get_conf('use_critic', False, as_type=bool)
        self.critic = None

        if self.use_critic:
            self.critic = Critic(
                device=self.device,
                dtype=self.dtype,
                process=self,
                **self.get_conf('critic', {})  # pass any other params
            )

        if self.sample_every is not None and self.sample_sources is None:
            raise ValueError('sample_every is specified but sample_sources is not')

        if self.epochs is None and self.max_steps is None:
            raise ValueError('epochs or max_steps must be specified')

        self.data_loaders = []
        # check datasets
        assert isinstance(self.datasets_objects, list)
        for dataset in self.datasets_objects:
            if 'path' not in dataset:
                raise ValueError('dataset must have a path')
            # check if is dir
            if not os.path.isdir(dataset['path']):
                raise ValueError(f"dataset path does is not a directory: {dataset['path']}")

        # make training folder
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        self._pattern_loss = None

        # build augmentation transforms
        aug_transforms = []

    def update_training_metadata(self):
        self.add_meta(OrderedDict({"training_info": self.get_training_info()}))

    def get_training_info(self):
        info = OrderedDict({
            'step': self.step_num,
            'epoch': self.epoch_num,
        })
        return info

    def load_datasets(self):
        if self.data_loader is None:
            print(f"Loading datasets")
            datasets = []
            for dataset in self.datasets_objects:
                print(f" - Dataset: {dataset['path']}")
                ds = copy.copy(dataset)
                ds['resolution'] = self.resolution

                if 'augmentations' not in ds:
                    ds['augmentations'] = self.augmentations

                # add the resize down augmentation
                ds['augmentations'] = [{
                    'method': 'Resize',
                    'params': {
                        'width': int(self.resolution // self.zoom),
                        'height': int(self.resolution // self.zoom),
                        # downscale interpolation, string will be evaluated
                        'interpolation': 'cv2.INTER_AREA'
                    }
                }] + ds['augmentations']

                image_dataset = AugmentedImageDataset(ds)
                datasets.append(image_dataset)

            concatenated_dataset = ConcatDataset(datasets)
            self.data_loader = DataLoader(
                concatenated_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=6
            )

    def setup_vgg19(self):
        if self.vgg_19 is None:
            self.vgg_19, self.style_losses, self.content_losses, self.vgg19_pool_4 = get_style_model_and_losses(
                single_target=True,
                device=self.device,
                output_layer_name='pool_4',
                dtype=self.torch_dtype
            )
            self.vgg_19.to(self.device, dtype=self.torch_dtype)
            self.vgg_19.requires_grad_(False)

            # we run random noise through first to get layer scalers to normalize the loss per layer
            # bs of 2 because we run pred and target through stacked
            noise = torch.randn((2, 3, self.resolution, self.resolution), device=self.device, dtype=self.torch_dtype)
            self.vgg_19(noise)
            for style_loss in self.style_losses:
                # get a scaler  to normalize to 1
                scaler = 1 / torch.mean(style_loss.loss).item()
                self.style_weight_scalers.append(scaler)
            for content_loss in self.content_losses:
                # get a scaler  to normalize to 1
                scaler = 1 / torch.mean(content_loss.loss).item()
                # if is nan, set to 1
                if scaler != scaler:
                    scaler = 1
                    print(f"Warning: content loss scaler is nan, setting to 1")
                self.content_weight_scalers.append(scaler)

            self.print(f"Style weight scalers: {self.style_weight_scalers}")
            self.print(f"Content weight scalers: {self.content_weight_scalers}")

    def get_style_loss(self):
        if self.style_weight > 0:
            # scale all losses with loss scalers
            loss = torch.sum(
                torch.stack([loss.loss * scaler for loss, scaler in zip(self.style_losses, self.style_weight_scalers)]))
            return loss
        else:
            return torch.tensor(0.0, device=self.device)

    def get_content_loss(self):
        if self.content_weight > 0:
            # scale all losses with loss scalers
            loss = torch.sum(torch.stack(
                [loss.loss * scaler for loss, scaler in zip(self.content_losses, self.content_weight_scalers)]))
            return loss
        else:
            return torch.tensor(0.0, device=self.device)

    def get_mse_loss(self, pred, target):
        if self.mse_weight > 0:
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred, target)
            return loss
        else:
            return torch.tensor(0.0, device=self.device)

    def get_tv_loss(self, pred, target):
        if self.tv_weight > 0:
            get_tv_loss = ComparativeTotalVariation()
            loss = get_tv_loss(pred, target)
            return loss
        else:
            return torch.tensor(0.0, device=self.device)

    def get_pattern_loss(self, pred, target):
        if self._pattern_loss is None:
            self._pattern_loss = PatternLoss(
                pattern_size=self.zoom,
                dtype=self.torch_dtype
            ).to(self.device, dtype=self.torch_dtype)
            self._pattern_loss = self._pattern_loss.to(self.device, dtype=self.torch_dtype)
        loss = torch.mean(self._pattern_loss(pred, target))
        return loss

    def save(self, step=None):
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        step_num = ''
        if step is not None:
            # zeropad 9 digits
            step_num = f"_{str(step).zfill(9)}"

        self.update_training_metadata()
        # filename = f'{self.job.name}{step_num}.safetensors'
        filename = f'{self.job.name}{step_num}.pth'
        # prepare meta
        save_meta = get_meta_for_safetensors(self.meta, self.job.name)

        # state_dict = self.model.state_dict()

        # state has the original state dict keys so we can save what we started from
        save_state_dict = self.model.state_dict()

        for key in list(save_state_dict.keys()):
            v = save_state_dict[key]
            v = v.detach().clone().to("cpu").to(torch.float32)
            save_state_dict[key] = v

        # most things wont use safetensors, save as torch
        # save_file(save_state_dict, os.path.join(self.save_root, filename), save_meta)
        torch.save(save_state_dict, os.path.join(self.save_root, filename))

        self.print(f"Saved to {os.path.join(self.save_root, filename)}")

        if self.use_critic:
            self.critic.save(step)

    def sample(self, step=None, batch: Optional[List[torch.Tensor]] = None):
        sample_folder = os.path.join(self.save_root, 'samples')
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder, exist_ok=True)
        batch_sample_folder = os.path.join(self.save_root, 'samples_batch')

        batch_targets = None
        batch_inputs = None
        if batch is not None and not os.path.exists(batch_sample_folder):
            os.makedirs(batch_sample_folder, exist_ok=True)

        self.model.eval()

        def process_and_save(img, target_img, save_path):
            img = img.to(self.device, dtype=self.esrgan_dtype)
            output = self.model(img)
            # output = (output / 2 + 0.5).clamp(0, 1)
            output = output.clamp(0, 1)
            img = img.clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            output = output.cpu().permute(0, 2, 3, 1).squeeze(0).float().numpy()
            img = img.cpu().permute(0, 2, 3, 1).squeeze(0).float().numpy()

            # convert to pillow image
            output = Image.fromarray((output * 255).astype(np.uint8))
            img = Image.fromarray((img * 255).astype(np.uint8))

            if isinstance(target_img, torch.Tensor):
                # convert to pil
                target_img = target_img.cpu().permute(0, 2, 3, 1).squeeze(0).float().numpy()
                target_img = Image.fromarray((target_img * 255).astype(np.uint8))

            # upscale to size * self.upscale_sample while maintaining pixels
            output = output.resize(
                (self.resolution * self.upscale_sample, self.resolution * self.upscale_sample),
                resample=Image.NEAREST
            )
            img = img.resize(
                (self.resolution * self.upscale_sample, self.resolution * self.upscale_sample),
                resample=Image.NEAREST
            )

            width, height = output.size

            # stack input image and decoded image
            target_image = target_img.resize((width, height))
            output = output.resize((width, height))
            img = img.resize((width, height))

            output_img = Image.new('RGB', (width * 3, height))

            output_img.paste(img, (0, 0))
            output_img.paste(output, (width, 0))
            output_img.paste(target_image, (width * 2, 0))

            output_img.save(save_path)

        with torch.no_grad():
            for i, img_url in enumerate(self.sample_sources):
                img = load_image(img_url, force_rgb=True)
                # crop if not square
                if img.width != img.height:
                    min_dim = min(img.width, img.height)
                    img = img.crop((0, 0, min_dim, min_dim))
                # resize
                img = img.resize((self.resolution * self.zoom, self.resolution * self.zoom), resample=Image.BICUBIC)

                target_image = img
                # downscale the image input
                img = img.resize((self.resolution, self.resolution), resample=Image.BICUBIC)

                # downscale the image input

                img = IMAGE_TRANSFORMS(img).unsqueeze(0).to(self.device, dtype=self.esrgan_dtype)
                img = img

                step_num = ''
                if step is not None:
                    # zero-pad 9 digits
                    step_num = f"_{str(step).zfill(9)}"
                seconds_since_epoch = int(time.time())
                # zero-pad 2 digits
                i_str = str(i).zfill(2)
                filename = f"{seconds_since_epoch}{step_num}_{i_str}.jpg"
                process_and_save(img, target_image, os.path.join(sample_folder, filename))

            if batch is not None:
                batch_targets = batch[0].detach()
                batch_inputs = batch[1].detach()
                batch_targets = torch.chunk(batch_targets, batch_targets.shape[0], dim=0)
                batch_inputs = torch.chunk(batch_inputs, batch_inputs.shape[0], dim=0)

                for i in range(len(batch_inputs)):
                    if step is not None:
                        # zero-pad 9 digits
                        step_num = f"_{str(step).zfill(9)}"
                    seconds_since_epoch = int(time.time())
                    # zero-pad 2 digits
                    i_str = str(i).zfill(2)
                    filename = f"{seconds_since_epoch}{step_num}_{i_str}.jpg"
                    process_and_save(batch_inputs[i], batch_targets[i], os.path.join(batch_sample_folder, filename))

        self.model.train()

    def load_model(self):
        state_dict = None
        path_to_load = self.pretrained_path
        # see if we have a checkpoint in out output to resume from
        self.print(f"Looking for latest checkpoint in {self.save_root}")
        files = glob.glob(os.path.join(self.save_root, f"{self.job.name}*.safetensors"))
        files += glob.glob(os.path.join(self.save_root, f"{self.job.name}*.pth"))
        if files and len(files) > 0:
            latest_file = max(files, key=os.path.getmtime)
            print(f" - Latest checkpoint is: {latest_file}")
            path_to_load = latest_file
            # todo update step and epoch count
        elif self.pretrained_path is None:
            self.print(f" - No checkpoint found, starting from scratch")
        else:
            self.print(f" - No checkpoint found, loading pretrained model")
            self.print(f" - path: {path_to_load}")

        if path_to_load is not None:
            self.print(f" - Loading pretrained checkpoint: {path_to_load}")
            # if ends with pth then assume pytorch checkpoint
            if path_to_load.endswith('.pth') or path_to_load.endswith('.pt'):
                state_dict = torch.load(path_to_load, map_location=self.device)
            elif path_to_load.endswith('.safetensors'):
                state_dict_raw = load_file(path_to_load)
                # make ordered dict as most things need it
                state_dict = OrderedDict()
                for key in esrgan_safetensors_keys:
                    state_dict[key] = state_dict_raw[key]
            else:
                raise Exception(f"Unknown file extension for checkpoint: {path_to_load}")

        # todo determine architecture from checkpoint
        self.model = ESRGAN(
            state_dict
        ).to(self.device, dtype=self.esrgan_dtype)

        # set the model to training mode
        self.model.train()
        self.model.requires_grad_(True)

    def run(self):
        super().run()
        self.load_datasets()
        steps_per_step = (self.critic.num_critic_per_gen + 1)

        max_step_epochs = self.max_steps // (len(self.data_loader) // steps_per_step)
        num_epochs = self.epochs
        if num_epochs is None or num_epochs > max_step_epochs:
            num_epochs = max_step_epochs

        max_epoch_steps = len(self.data_loader) * num_epochs * steps_per_step
        num_steps = self.max_steps
        if num_steps is None or num_steps > max_epoch_steps:
            num_steps = max_epoch_steps
        self.max_steps = num_steps
        self.epochs = num_epochs
        start_step = self.step_num
        self.first_step = start_step

        self.print(f"Training ESRGAN model:")
        self.print(f" - Training folder: {self.training_folder}")
        self.print(f" - Batch size: {self.batch_size}")
        self.print(f" - Learning rate: {self.learning_rate}")
        self.print(f" - Epochs: {num_epochs}")
        self.print(f" - Max steps: {self.max_steps}")

        # load model
        self.load_model()

        params = self.model.parameters()

        if self.style_weight > 0 or self.content_weight > 0 or self.use_critic:
            self.setup_vgg19()
            self.vgg_19.requires_grad_(False)
            self.vgg_19.eval()
            if self.use_critic:
                self.critic.setup()

        optimizer = get_optimizer(params, self.optimizer_type, self.learning_rate,
                                  optimizer_params=self.optimizer_params)

        # setup scheduler
        # todo allow other schedulers
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            total_iters=num_steps,
            factor=1,
            verbose=False
        )

        # setup tqdm progress bar
        self.progress_bar = tqdm(
            total=num_steps,
            desc='Training ESRGAN',
            leave=True
        )

        blank_losses = OrderedDict({
            "total": [],
            "style": [],
            "content": [],
            "mse": [],
            "kl": [],
            "tv": [],
            "ptn": [],
            "crD": [],
            "crG": [],
        })
        epoch_losses = copy.deepcopy(blank_losses)
        log_losses = copy.deepcopy(blank_losses)
        print("Generating baseline samples")
        self.sample(step=0)
        # range start at self.epoch_num go to self.epochs
        critic_losses = []
        for epoch in range(self.epoch_num, self.epochs, 1):
            if self.step_num >= self.max_steps:
                break
            flush()
            for targets, inputs in self.data_loader:
                if self.step_num >= self.max_steps:
                    break
                with torch.no_grad():
                    is_critic_only_step = False
                    if self.use_critic and 1 / (self.critic.num_critic_per_gen + 1) < np.random.uniform():
                        is_critic_only_step = True

                    targets = targets.to(self.device, dtype=self.esrgan_dtype).clamp(0, 1).detach()
                    inputs = inputs.to(self.device, dtype=self.esrgan_dtype).clamp(0, 1).detach()

                optimizer.zero_grad()
                # dont do grads here for critic step
                do_grad = not is_critic_only_step
                with torch.set_grad_enabled(do_grad):
                    pred = self.model(inputs)

                    pred = pred.to(self.device, dtype=self.torch_dtype).clamp(0, 1)
                    targets = targets.to(self.device, dtype=self.torch_dtype).clamp(0, 1)
                    if torch.isnan(pred).any():
                        raise ValueError('pred has nan values')
                    if torch.isnan(targets).any():
                        raise ValueError('targets has nan values')

                    # Run through VGG19
                    if self.style_weight > 0 or self.content_weight > 0 or self.use_critic:
                        stacked = torch.cat([pred, targets], dim=0)
                        # stacked = (stacked / 2 + 0.5).clamp(0, 1)
                        stacked = stacked.clamp(0, 1)
                        self.vgg_19(stacked)
                        # make sure we dont have nans
                        if torch.isnan(self.vgg19_pool_4.tensor).any():
                            raise ValueError('vgg19_pool_4 has nan values')

                if is_critic_only_step:
                    critic_d_loss = self.critic.step(self.vgg19_pool_4.tensor.detach())
                    critic_losses.append(critic_d_loss)
                    # don't do generator step
                    continue
                else:
                    # doing a regular step
                    if len(critic_losses) == 0:
                        critic_d_loss = 0
                    else:
                        critic_d_loss = sum(critic_losses) / len(critic_losses)

                style_loss = self.get_style_loss() * self.style_weight
                content_loss = self.get_content_loss() * self.content_weight

                mse_loss = self.get_mse_loss(pred, targets) * self.mse_weight
                tv_loss = self.get_tv_loss(pred, targets) * self.tv_weight
                pattern_loss = self.get_pattern_loss(pred, targets) * self.pattern_weight
                if self.use_critic:
                    critic_gen_loss = self.critic.get_critic_loss(self.vgg19_pool_4.tensor) * self.critic_weight
                else:
                    critic_gen_loss = torch.tensor(0.0, device=self.device, dtype=self.torch_dtype)

                loss = style_loss + content_loss + mse_loss + tv_loss + critic_gen_loss + pattern_loss
                # make sure non nan
                if torch.isnan(loss):
                    raise ValueError('loss is nan')

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # update progress bar
                loss_value = loss.item()
                # get exponent like 3.54e-4
                loss_string = f"loss: {loss_value:.2e}"
                if self.content_weight > 0:
                    loss_string += f" cnt: {content_loss.item():.2e}"
                if self.style_weight > 0:
                    loss_string += f" sty: {style_loss.item():.2e}"
                if self.mse_weight > 0:
                    loss_string += f" mse: {mse_loss.item():.2e}"
                if self.tv_weight > 0:
                    loss_string += f" tv: {tv_loss.item():.2e}"
                if self.pattern_weight > 0:
                    loss_string += f" ptn: {pattern_loss.item():.2e}"
                if self.use_critic and self.critic_weight > 0:
                    loss_string += f" crG: {critic_gen_loss.item():.2e}"
                if self.use_critic:
                    loss_string += f" crD: {critic_d_loss:.2e}"

                if self.optimizer_type.startswith('dadaptation') or self.optimizer_type.startswith('prodigy'):
                    learning_rate = (
                            optimizer.param_groups[0]["d"] *
                            optimizer.param_groups[0]["lr"]
                    )
                else:
                    learning_rate = optimizer.param_groups[0]['lr']

                lr_critic_string = ''
                if self.use_critic:
                    lr_critic = self.critic.get_lr()
                    lr_critic_string = f" lrC: {lr_critic:.1e}"

                self.progress_bar.set_postfix_str(f"lr: {learning_rate:.1e}{lr_critic_string} {loss_string}")
                self.progress_bar.set_description(f"E: {epoch}")
                self.progress_bar.update(1)

                epoch_losses["total"].append(loss_value)
                epoch_losses["style"].append(style_loss.item())
                epoch_losses["content"].append(content_loss.item())
                epoch_losses["mse"].append(mse_loss.item())
                epoch_losses["tv"].append(tv_loss.item())
                epoch_losses["ptn"].append(pattern_loss.item())
                epoch_losses["crG"].append(critic_gen_loss.item())
                epoch_losses["crD"].append(critic_d_loss)

                log_losses["total"].append(loss_value)
                log_losses["style"].append(style_loss.item())
                log_losses["content"].append(content_loss.item())
                log_losses["mse"].append(mse_loss.item())
                log_losses["tv"].append(tv_loss.item())
                log_losses["ptn"].append(pattern_loss.item())
                log_losses["crG"].append(critic_gen_loss.item())
                log_losses["crD"].append(critic_d_loss)

                # don't do on first step
                if self.step_num != start_step:
                    if self.sample_every and self.step_num % self.sample_every == 0:
                        # print above the progress bar
                        self.print(f"Sampling at step {self.step_num}")
                        self.sample(self.step_num, batch=[targets, inputs])

                    if self.save_every and self.step_num % self.save_every == 0:
                        # print above the progress bar
                        self.print(f"Saving at step {self.step_num}")
                        self.save(self.step_num)

                    if self.log_every and self.step_num % self.log_every == 0:
                        # log to tensorboard
                        if self.writer is not None:
                            # get avg loss
                            for key in log_losses:
                                log_losses[key] = sum(log_losses[key]) / (len(log_losses[key]) + 1e-6)
                                # if log_losses[key] > 0:
                                self.writer.add_scalar(f"loss/{key}", log_losses[key], self.step_num)
                        # reset log losses
                        log_losses = copy.deepcopy(blank_losses)

                self.step_num += 1
            # end epoch
            if self.writer is not None:
                eps = 1e-6
                # get avg loss
                for key in epoch_losses:
                    epoch_losses[key] = sum(log_losses[key]) / (len(log_losses[key]) + eps)
                    if epoch_losses[key] > 0:
                        self.writer.add_scalar(f"epoch loss/{key}", epoch_losses[key], epoch)
            # reset epoch losses
            epoch_losses = copy.deepcopy(blank_losses)

        self.save()
