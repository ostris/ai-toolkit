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
    'refiner_unet': {
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
        train_refiner: bool = False,
        unload_text_encoder: bool = False,
        require_grads: bool = True,
):
    preset = copy.deepcopy(empty_preset)
    if not cached_latents:
        preset['vae']['device'] = device

    if train_unet:
        preset['unet']['training'] = True
        preset['unet']['requires_grad'] = require_grads
        preset['unet']['device'] = device
    else:
        preset['unet']['device'] = device

    if train_text_encoder:
        preset['text_encoder']['training'] = True
        preset['text_encoder']['requires_grad'] = require_grads
        preset['text_encoder']['device'] = device
    else:
        preset['text_encoder']['device'] = device

    if train_embedding:
        preset['text_encoder']['training'] = True
        preset['text_encoder']['requires_grad'] = require_grads
        preset['text_encoder']['training'] = True
        preset['unet']['training'] = True

    if train_refiner:
        preset['refiner_unet']['training'] = True
        preset['refiner_unet']['requires_grad'] = require_grads
        preset['refiner_unet']['device'] = device
        # if not training unet, move that to cpu
        if not train_unet:
            preset['unet']['device'] = 'cpu'

    if train_lora:
        # preset['text_encoder']['requires_grad'] = False
        preset['unet']['requires_grad'] = False
        if train_refiner:
            preset['refiner_unet']['requires_grad'] = False

    if train_adapter:
        preset['adapter']['requires_grad'] = require_grads
        preset['adapter']['training'] = True
        preset['adapter']['device'] = device
        preset['unet']['training'] = True
        preset['unet']['requires_grad'] = False
        preset['unet']['device'] = device
        preset['text_encoder']['device'] = device

    if unload_text_encoder:
        preset['text_encoder']['training'] = False
        preset['text_encoder']['requires_grad'] = False
        preset['text_encoder']['device'] = 'cpu'

    return preset
