import torch

# extra tensors ride in the latent cache safetensors under this key prefix
DISK_PREFIX = "dto."


def _unwrap(value):
    """Recursively convert any DTO in value back to a plain torch.Tensor.
    Internal: everywhere else, use `dto.tensor`."""
    if isinstance(value, DTO):
        return value.tensor
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(v) for v in value)
    return value


def _rebuild_dto(tensor, extras):
    return DTO(tensor, **extras)


class DTO(torch.Tensor):
    """A torch.Tensor that carries named side-channel data (audio rows, video
    tokens, extra targets, ...) through code that only knows about the main
    tensor.

    Backwards compatible by design: a DTO *is* the main tensor, so every
    existing shape check, math op, and indexing keeps working. Any torch op
    returns a plain tensor — extras never leak through math — and only the
    explicit carriers (``to``, ``clone``, ``detach``, ``cpu``, ``cuda``,
    ``pin_memory``, ``map``, ``cat``) keep the extras attached.

        latent = DTO(video_latent, audio=audio_rows, num_frames=77)
        latent.audio          # extra lookup, AttributeError if missing
        latent.get("audio")   # None if missing
        latent.tensor         # plain tensor view (shares storage)
        latent * 2            # plain tensor, extras dropped
        latent.to("cuda")     # DTO, tensor extras moved too
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, **extras):
        if isinstance(tensor, DTO):
            extras = {**tensor.extras, **extras}
            tensor = tensor.tensor
        obj = tensor.as_subclass(cls)
        obj._dto_extras = dict(extras)
        return obj

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # run every torch op as if on plain tensors so extras never
        # accidentally propagate through math with stale values
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunctionSubclass():
            return _unwrap(func(*args, **kwargs))

    @property
    def tensor(self) -> torch.Tensor:
        with torch._C.DisableTorchFunctionSubclass():
            return self.as_subclass(torch.Tensor)

    @property
    def extras(self) -> dict:
        return self._dto_extras

    def get(self, key, default=None):
        return self._dto_extras.get(key, default)

    def set(self, key, value):
        self._dto_extras[key] = value
        return self

    def __getattr__(self, name):
        if name == "_dto_extras":
            raise AttributeError(name)
        try:
            return self._dto_extras[name]
        except KeyError:
            raise AttributeError(
                f"DTO has no extra '{name}'; extras: {list(self._dto_extras.keys())}"
            )

    def __repr__(self):
        return f"DTO(extras={list(self._dto_extras.keys())}, tensor={self.tensor!r})"

    def __reduce_ex__(self, protocol):
        return (_rebuild_dto, (self.tensor, self._dto_extras))

    def map(self, fn):
        """Apply fn to the main tensor and every tensor extra, keep the rest."""
        return DTO(
            fn(self.tensor),
            **{
                k: fn(v) if torch.is_tensor(v) else v
                for k, v in self._dto_extras.items()
            },
        )

    def _carry(self, base, fn_tensor):
        return DTO(
            base,
            **{
                k: fn_tensor(v) if torch.is_tensor(v) else v
                for k, v in self._dto_extras.items()
            },
        )

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)

        def move(t):
            # dtype casts only follow onto floating extras; int extras
            # (frame counts, indices) keep their dtype on device moves
            d = dtype if dtype is not None and t.is_floating_point() else None
            return t.to(device=device, dtype=d, non_blocking=non_blocking)

        return self._carry(self.tensor.to(*args, **kwargs), move)

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device=None):
        return self.to(device if device is not None else "cuda")

    def clone(self):
        return self.map(lambda t: t.clone())

    def detach(self):
        return self.map(lambda t: t.detach())

    def pin_memory(self):
        return self.map(lambda t: t.pin_memory())

    @classmethod
    def cat(cls, items, dim=0):
        """Batch-collate: cat main tensors along dim, tensor extras shared by
        every item along dim 0. Non-tensor extras keep a single value when
        identical everywhere, else become a list."""
        base = torch.cat([_unwrap(x) for x in items], dim=dim)
        dtos = [x for x in items if isinstance(x, cls)]
        if len(dtos) != len(items):
            return base if not dtos else cls(base, **dtos[0].extras)
        keys = set(dtos[0].extras.keys())
        for d in dtos[1:]:
            keys &= set(d.extras.keys())
        extras = {}
        for k in keys:
            vals = [d.extras[k] for d in dtos]
            if all(torch.is_tensor(v) for v in vals):
                extras[k] = torch.cat(vals, dim=0)
            elif all(v == vals[0] for v in vals[1:]) if len(vals) > 1 else True:
                extras[k] = vals[0]
            else:
                extras[k] = vals
        return cls(base, **extras)

    @classmethod
    def stack(cls, items):
        """Collate per-item latents into a batch: unsqueeze(0) + cat. A tensor
        extra missing on some items is zero-filled there (a missing stream is
        silence). Returns a plain tensor when no item carries extras."""
        base = torch.cat([_unwrap(x).unsqueeze(0) for x in items], dim=0)
        keys = []
        for x in items:
            if isinstance(x, cls):
                keys.extend(k for k in x.extras if k not in keys)
        if not keys:
            return base
        extras = {}
        for k in keys:
            vals = [x.get(k) if isinstance(x, cls) else None for x in items]
            present = [v for v in vals if v is not None]
            if all(torch.is_tensor(v) for v in present):
                extras[k] = torch.cat(
                    [
                        (v if v is not None else torch.zeros_like(present[0])).unsqueeze(0)
                        for v in vals
                    ],
                    dim=0,
                )
            elif all(v == present[0] for v in present[1:]):
                extras[k] = present[0]
            else:
                extras[k] = vals
        return cls(base, **extras)

    def to_state_dict(self, key="latent") -> dict:
        """Flatten for safetensors: main tensor under ``key``, tensor extras
        under ``dto.<name>``. Non-tensor extras are not persisted."""
        state_dict = {key: self.tensor.contiguous()}
        for k, v in self._dto_extras.items():
            if torch.is_tensor(v):
                state_dict[f"{DISK_PREFIX}{k}"] = v.contiguous()
        return state_dict

    @staticmethod
    def from_state_dict(state_dict: dict, key="latent"):
        """Inverse of ``to_state_dict``. Returns a plain tensor when the file
        holds no dto extras, so legacy caches load unchanged."""
        extras = {
            k[len(DISK_PREFIX):]: v
            for k, v in state_dict.items()
            if k.startswith(DISK_PREFIX)
        }
        if not extras:
            return state_dict[key]
        return DTO(state_dict[key], **extras)
