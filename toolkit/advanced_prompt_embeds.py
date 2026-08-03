import os
import torch
from safetensors.torch import load_file, save_file


class AdvancedPromptEmbeds:
    """
    Flexible container for prompt embedding tensors.

    Each value passed in must be a list of tensors, where each item in the
    list corresponds to a single item in the batch (list length == batch size).
    Do not store more than one tensor per batch item under the same key — if
    you need multiple tensors per batch item, give them different key names.

    Usage:
        pe = AdvancedPromptEmbeds(
            prompt_embeds=[t0, t1, t2],      # one tensor per batch item
            pooled_embeds=[p0, p1, p2],
        )

        pe.prompt_embeds        # -> [t0, t1, t2]
        pe['prompt_embeds']     # -> [t0, t1, t2]
        pe.keys()               # -> ['prompt_embeds', 'pooled_embeds']

        # add more after init
        pe.extra = [e0, e1, e2]
        pe['extra2'] = [e0, e1, e2]
        pe.set('extra3', [e0, e1, e2])
        pe.update(extra4=[e0, e1, e2])
    """

    def __init__(self, **kwargs):
        self._store = {}
        self._frozen_dtype_keys = []
        for key, value in kwargs.items():
            if not isinstance(value, list):
                value = [value]
            self._store[key] = value

    @property
    def frozen_dtype_keys(self):
        return self._frozen_dtype_keys

    @frozen_dtype_keys.setter
    def frozen_dtype_keys(self, keys):
        self._frozen_dtype_keys = list(keys) if keys else []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        store = self.__dict__.get("_store", {})
        if name in store:
            return store[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        cls_attr = getattr(type(self), name, None)
        if isinstance(cls_attr, property):
            super().__setattr__(name, value)
            return
        if not isinstance(value, list):
            value = [value]
        self._store[name] = value

    def set(self, key, value):
        if not isinstance(value, list):
            value = [value]
        self._store[key] = value

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not isinstance(value, list):
                value = [value]
            self._store[key] = value

    def keys(self):
        return list(self._store.keys())

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        if not isinstance(value, list):
            value = [value]
        self._store[key] = value

    def __contains__(self, key):
        return key in self._store

    def to(self, *args, **kwargs):
        frozen = set(self._frozen_dtype_keys)
        if frozen:
            no_dtype_args = [a for a in args if not isinstance(a, torch.dtype)]
            no_dtype_kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        new_pe = AdvancedPromptEmbeds()
        new_pe._frozen_dtype_keys = list(self._frozen_dtype_keys)
        for key, value in self._store.items():
            if key in frozen:
                new_pe._store[key] = [
                    v.to(*no_dtype_args, **no_dtype_kwargs) for v in value
                ]
            else:
                new_pe._store[key] = [v.to(*args, **kwargs) for v in value]
        return new_pe

    def detach(self):
        new_pe = AdvancedPromptEmbeds()
        new_pe._frozen_dtype_keys = list(self._frozen_dtype_keys)
        for key, value in self._store.items():
            new_pe._store[key] = [v.detach() for v in value]
        return new_pe

    def clone(self):
        new_pe = AdvancedPromptEmbeds()
        new_pe._frozen_dtype_keys = list(self._frozen_dtype_keys)
        for key, value in self._store.items():
            new_pe._store[key] = [v.clone() for v in value]
        return new_pe

    def expand_to_batch(self, batch_size):
        new_pe = AdvancedPromptEmbeds()
        new_pe._frozen_dtype_keys = list(self._frozen_dtype_keys)
        for key, value in self._store.items():
            if len(value) == 1:
                new_pe._store[key] = value * batch_size
            elif len(value) == batch_size:
                new_pe._store[key] = value
            else:
                raise ValueError(
                    f"Cannot expand key {key!r}: expected list of length 1 or {batch_size}, got {len(value)}"
                )
        return new_pe

    def save(self, path):
        data = {}
        metadata = {"class_name": self.__class__.__name__}
        for key, value in self._store.items():
            if len(value) != 1:
                raise ValueError(
                    f"Cannot save key {key!r}: expected list of length 1, got {len(value)}"
                )
            data[key] = value[0]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(data, path, metadata=metadata)

    @classmethod
    def load(cls, path=None):
        if path is not None:
            loaded = load_file(path)
        else:
            raise ValueError("Must provide a path")

        data = {}
        for key in loaded.keys():
            data[key] = loaded[key]

        return cls(**data)

    @classmethod
    def concat_prompt_embeds(
        cls, prompt_embeds: list["AdvancedPromptEmbeds"], padding_side: str = "right"
    ):
        embeds = {}
        frozen = []
        for pe in prompt_embeds:
            for key in pe.keys():
                if key not in embeds:
                    embeds[key] = []
                embeds[key].extend(pe[key])
            for k in pe.frozen_dtype_keys:
                if k not in frozen:
                    frozen.append(k)
        out = cls(**embeds)
        out.frozen_dtype_keys = frozen
        return out

    @classmethod
    def split_prompt_embeds(cls, concatenated: "AdvancedPromptEmbeds", num_parts=None):
        if num_parts is None:
            # use length of first item as num_parts
            num_parts = len(concatenated[concatenated.keys()[0]])
        split_embeds = [cls() for _ in range(num_parts)]
        for pe in split_embeds:
            pe.frozen_dtype_keys = list(concatenated.frozen_dtype_keys)
        for key in concatenated.keys():
            values = concatenated[key]
            if len(values) != num_parts:
                raise ValueError(
                    f"Cannot split key {key!r}: expected list of length {num_parts}, got {len(values)}"
                )
            for i in range(num_parts):
                split_embeds[i]._store[key] = [values[i]]
