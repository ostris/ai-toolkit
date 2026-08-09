import gc
import torch
from toolkit.basic import flush
from toolkit.memory_management import MemoryManager
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


def _detach_and_free(te: torch.nn.Module):
    MemoryManager.detach(te)
    # bypass any nopped-out .to() override and free the weight storages by
    # moving them to the meta device. After the FakeTextEncoder swap every
    # public handle points at the fake, so keeping the real module resident
    # on CPU only costs host RAM (a quantized 32B text encoder is ~15.7 GB);
    # any stray reference that tried to run it after unload was already
    # unsupported.
    torch.nn.Module.to(te, 'meta')


def _trim_host_heap():
    # glibc keeps freed arenas mapped; without this the RSS of a large
    # unloaded text encoder never returns to the OS (measured ~3.4 GB
    # retained after a full gc on linux).
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            text_encoder_list = []
            pipe = model.pipeline

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if hasattr(pipe, "text_encoder"):
                _detach_and_free(pipe.text_encoder)
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                pipe.text_encoder = te

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                real_te = getattr(pipe, f"text_encoder_{i}")
                _detach_and_free(real_te)
                te = FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)
                text_encoder_list.append(te)
                setattr(pipe, f"text_encoder_{i}", te)
                i += 1
            model.text_encoder = text_encoder_list
        else:
            # only has a single text encoder
            _detach_and_free(model.text_encoder)
            model.text_encoder = FakeTextEncoder(
                device=model.device_torch,
                dtype=model.torch_dtype
            )

    torch.cuda.empty_cache()
    gc.collect()
    flush()
    _trim_host_heap()
