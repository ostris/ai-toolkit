import torch
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


def unload_text_encoder(model: "BaseModel"):
    # unload the text encoder in a way that will work with all models and will not throw errors
    # we need to make it appear as a text encoder module without actually having one so all
    # to functions and what not will work.

    if model.text_encoder is not None:
        if isinstance(model.text_encoder, list):
            pipe = model.pipeline
            freed = set()

            def _free(te):
                if te is None or isinstance(te, FakeTextEncoder) or id(te) in freed:
                    return
                MemoryManager.free(te)
                freed.add(id(te))

            def _new_fake():
                return FakeTextEncoder(device=model.device_torch, dtype=model.torch_dtype)

            # model.text_encoder is the source of truth; some pipelines (self-referencing
            # or built with text_encoder=None) never expose the TE as an attribute
            for real_te in model.text_encoder:
                _free(real_te)
            fakes = [_new_fake() for _ in model.text_encoder]
            model.text_encoder = fakes

            def _fake(idx):
                return fakes[idx] if idx < len(fakes) else _new_fake()

            # the pipeline stores text encoders like text_encoder, text_encoder_2, text_encoder_3, etc.
            if getattr(pipe, "text_encoder", None) is not None:
                _free(pipe.text_encoder)
                pipe.text_encoder = _fake(0)

            i = 2
            while hasattr(pipe, f"text_encoder_{i}"):
                real_te = getattr(pipe, f"text_encoder_{i}")
                if real_te is not None:
                    _free(real_te)
                    setattr(pipe, f"text_encoder_{i}", _fake(i - 1))
                i += 1
        else:
            # only has a single text encoder
            MemoryManager.free(model.text_encoder)
            model.text_encoder = FakeTextEncoder(
                device=model.device_torch,
                dtype=model.torch_dtype
            )

    MemoryManager.release_cached_memory()
