import copy
import os
import time
from collections import OrderedDict

from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import save_file
from torch.utils.data import DataLoader, ConcatDataset
import torch
from torch import nn
from torchvision.transforms import transforms

from jobs.process import BaseTrainProcess
from toolkit.kohya_model_util import load_vae
from toolkit.data_loader import ImageDataset
from toolkit.metadata import get_meta_for_safetensors
from toolkit.train_tools import get_torch_dtype
from tqdm import tqdm
import time
import numpy as np

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

INVERSE_IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.Normalize(
            mean=[-0.5/0.5],
            std=[1/0.5]
        ),
        transforms.ToPILImage(),
    ]
)


class TrainVAEProcess(BaseTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.data_loader = None
        self.vae = None
        self.device = self.get_conf('device', self.job.device)
        self.vae_path = self.get_conf('vae_path', required=True)
        self.datasets_objects = self.get_conf('datasets', required=True)
        self.training_folder = self.get_conf('training_folder', self.job.training_folder)
        self.batch_size = self.get_conf('batch_size', 1)
        self.resolution = self.get_conf('resolution', 256)
        self.learning_rate = self.get_conf('learning_rate', 1e-4)
        self.sample_every = self.get_conf('sample_every', None)
        self.epochs = self.get_conf('epochs', None)
        self.max_steps = self.get_conf('max_steps', None)
        self.save_every = self.get_conf('save_every', None)
        self.dtype = self.get_conf('dtype', 'float32')
        self.sample_sources = self.get_conf('sample_sources', None)
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.save_root = os.path.join(self.training_folder, self.job.name)

        if self.sample_every is not None and self.sample_sources is None:
            raise ValueError('sample_every is specified but sample_sources is not')

        if self.epochs is None and self.max_steps is None:
            raise ValueError('epochs or max_steps must be specified')

        self.data_loaders = []
        datasets = []
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

    def load_datasets(self):
        if self.data_loader is None:
            print(f"Loading datasets")
            datasets = []
            for dataset in self.datasets_objects:
                print(f" - Dataset: {dataset['path']}")
                ds = copy.copy(dataset)
                ds['resolution'] = self.resolution
                image_dataset = ImageDataset(ds)
                datasets.append(image_dataset)

            concatenated_dataset = ConcatDataset(datasets)
            self.data_loader = DataLoader(
                concatenated_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

    def get_loss(self, pred, target):
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, target)
        return loss

    def get_elbo_loss(self, pred, target, mu, log_var):
        # ELBO (Evidence Lower BOund) loss, aka variational lower bound
        reconstruction_loss = nn.MSELoss(reduction='sum')
        BCE = reconstruction_loss(pred, target)  # reconstruction loss
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL divergence
        return BCE + KLD

    def save(self, step=None):
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        step_num = ''
        if step is not None:
            # zeropad 9 digits
            step_num = f"_{str(step).zfill(9)}"

        filename = f'{self.job.name}{step_num}.safetensors'
        save_path = os.path.join(self.save_root, filename)
        # prepare meta
        save_meta = get_meta_for_safetensors(self.meta, self.job.name)

        state_dict = self.vae.state_dict()

        for key in list(state_dict.keys()):
            v = state_dict[key]
            v = v.detach().clone().to("cpu").to(torch.float32)
            state_dict[key] = v

        # having issues with meta
        save_file(state_dict, os.path.join(self.save_root, filename), save_meta)

        print(f"Saved to {os.path.join(self.save_root, filename)}")

    def sample(self, step=None):
        sample_folder = os.path.join(self.save_root, 'samples')
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder, exist_ok=True)

        with torch.no_grad():
            self.vae.encoder.eval()
            self.vae.decoder.eval()

            for i, img_url in enumerate(self.sample_sources):
                img = exif_transpose(Image.open(img_url))
                img = img.convert('RGB')
                # crop if not square
                if img.width != img.height:
                    min_dim = min(img.width, img.height)
                    img = img.crop((0, 0, min_dim, min_dim))
                # resize
                img = img.resize((self.resolution, self.resolution))

                input_img = img
                img = IMAGE_TRANSFORMS(img).unsqueeze(0).to(self.device, dtype=self.torch_dtype)
                decoded = self.vae(img).sample.squeeze(0)
                decoded = INVERSE_IMAGE_TRANSFORMS(decoded)

                # stack input image and decoded image
                input_img = input_img.resize((self.resolution, self.resolution))
                decoded = decoded.resize((self.resolution, self.resolution))

                output_img = Image.new('RGB', (self.resolution * 2, self.resolution))
                output_img.paste(input_img, (0, 0))
                output_img.paste(decoded, (self.resolution, 0))

                step_num = ''
                if step is not None:
                    # zeropad 9 digits
                    step_num = f"_{str(step).zfill(9)}"
                seconds_since_epoch = int(time.time())
                # zeropad 2 digits
                i_str = str(i).zfill(2)
                filename = f"{seconds_since_epoch}{step_num}_{i_str}.png"
                output_img.save(os.path.join(sample_folder, filename))
        self.vae.decoder.train()

    def run(self):
        super().run()
        self.load_datasets()

        max_step_epochs = self.max_steps // len(self.data_loader)
        num_epochs = self.epochs
        if num_epochs is None or num_epochs > max_step_epochs:
            num_epochs = max_step_epochs

        max_epoch_steps = len(self.data_loader) * num_epochs
        num_steps = self.max_steps
        if num_steps is None or num_steps > max_epoch_steps:
            num_steps = max_epoch_steps

        print(f"Training VAE")
        print(f" - Training folder: {self.training_folder}")
        print(f" - Batch size: {self.batch_size}")
        print(f" - Learning rate: {self.learning_rate}")
        print(f" - Epochs: {num_epochs}")
        print(f" - Max steps: {self.max_steps}")

        # load vae
        print(f"Loading VAE")
        print(f" - Loading VAE: {self.vae_path}")
        if self.vae is None:
            self.vae = load_vae(self.vae_path, dtype=self.torch_dtype)

        # set decoder to train
        self.vae.to(self.device, dtype=self.torch_dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.vae.decoder.requires_grad_(True)
        self.vae.decoder.train()

        parameters = self.vae.decoder.parameters()

        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

        # setup scheduler
        # scheduler = lr_scheduler.ConstantLR
        # todo allow other schedulers
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            total_iters=num_steps,
            factor=1,
            verbose=False
        )

        # setup tqdm progress bar
        progress_bar = tqdm(
            total=num_steps,
            desc='Training VAE',
            leave=True
        )

        step = 0
        # sample first
        self.sample()
        for epoch in range(num_epochs):
            if step >= num_steps:
                break
            for batch in self.data_loader:
                if step >= num_steps:
                    break

                batch = batch.to(self.device, dtype=self.torch_dtype)

                # forward pass
                # with torch.no_grad():
                dgd = self.vae.encode(batch).latent_dist
                mu, logvar = dgd.mean, dgd.logvar
                latents = dgd.sample()
                latents.requires_grad_(True)

                pred = self.vae.decode(latents).sample

                loss = self.get_elbo_loss(pred, batch, mu, logvar)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update progress bar
                loss_value = loss.item()
                # get exponent like 3.54e-4
                loss_string = f"{loss_value:.2e}"
                learning_rate = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix_str(f"LR: {learning_rate:.2e} Loss: {loss_string}")
                progress_bar.set_description(f"E: {epoch} - S: {step} ")
                progress_bar.update(1)

                if step != 0:
                    if self.sample_every and step % self.sample_every == 0:
                        # print above the progress bar
                        print(f"Sampling at step {step}")
                        self.sample(step)

                    if self.save_every and  step % self.save_every == 0:
                        # print above the progress bar
                        print(f"Saving at step {step}")
                        self.save(step)

                step += 1

        self.save()

        pass
