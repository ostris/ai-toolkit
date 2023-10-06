from typing import Union

import torch
import copy

empty_preset = {
    'vae': {
        'training': False,
        'device': 'cpu',
    },
    'unet': {
        'training': False,
        'requires_grad': False,
        'device': 'cpu',
    },
    'text_encoder': {
        'training': False,
        'requires_grad': False,
        'device': 'cpu',
    },
    'adapter': {
        'training': False,
        'requires_grad': False,
        'device': 'cpu',
    },
}


def get_train_sd_device_state_preset(
        device: Union[str, torch.device],
        train_unet: bool = False,
        train_text_encoder: bool = False,
        cached_latents: bool = False,
        train_lora: bool = False,
        train_adapter: bool = False,
        train_embedding: bool = False,
):
    preset = copy.deepcopy(empty_preset)
    if not cached_latents:
        preset['vae']['device'] = device

    if train_unet:
        preset['unet']['training'] = True
        preset['unet']['requires_grad'] = True
        preset['unet']['device'] = device
    else:
        preset['unet']['device'] = device

    if train_text_encoder:
        preset['text_encoder']['training'] = True
        preset['text_encoder']['requires_grad'] = True
        preset['text_encoder']['device'] = device
    else:
        preset['text_encoder']['device'] = device

    if train_embedding:
        preset['text_encoder']['training'] = True
        preset['text_encoder']['requires_grad'] = True
        preset['text_encoder']['training'] = True
        preset['unet']['training'] = True

    if train_lora:
        # preset['text_encoder']['requires_grad'] = False
        preset['unet']['requires_grad'] = False

    if train_adapter:
        preset['adapter']['requires_grad'] = True
        preset['adapter']['training'] = True
        preset['adapter']['device'] = device
        preset['unet']['training'] = True

    return preset
