import gc
import torch
from toolkit.basic import flush
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        # register a dummy parameter to avoid errors in some cases
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self._device = device
        self._dtype = dtype

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This is a fake text encoder and should not be used for inference."
        )
        return None

    @property
    def device(self):
        return self._device
    
    @property
    def dtype(self):
        return self._dtype
    
    def to(self, *args, **kwargs):
        return self


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    is_mps = isinstance(model.device_torch, torch.device) and model.device_torch.type == "mps"

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            text_encoder_list = []
            pipe = model.pipeline

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if hasattr(pipe, "text_encoder"):
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                # if we are on mps, we don't want to move to cpu because it's unified memory
                # and just freeing the reference is enough and faster
                if not is_mps:
                    pipe.text_encoder.to('cpu')
                else:
                    pipe.text_encoder.to('meta')
                pipe.text_encoder = te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                if is_mps:
                    getattr(pipe, f"text_encoder_{i}").to('meta')
                setattr(pipe, f"text_encoder_{i}", te)
                i += 1
            model.text_encoder = text_encoder_list
        else:
            # only has a single text encoder
            if is_mps:
                model.text_encoder.to('meta')
            model.text_encoder = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)

    if is_mps:
        gc.collect()
        torch.mps.empty_cache()

    flush()
