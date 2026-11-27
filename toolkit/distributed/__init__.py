"""Minimal multi-GPU (torchrun) support helpers.

Everything here is a no-op when running single-process (WORLD_SIZE unset or 1),
so the existing single-GPU path is byte-for-byte unchanged. Launch multi-GPU
with torchrun; accelerate's Accelerator picks up the env automatically.
"""

import os

import torch
from torch.utils.data import Dataset


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get('WORLD_SIZE', '1'))


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get('RANK', '0'))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_process_group_if_needed(backend=None):
    """Initialize the default process group when launched under torchrun."""
    if not is_distributed():
        return
    if torch.distributed.is_initialized():
        return
    if backend is None:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    torch.distributed.init_process_group(backend=backend)


class RankShardDataset(Dataset):
    """Sharded view of a dataset: rank r sees indices [r::world_size].

    Applied at the outermost (already-batched) dataset level, so ai-toolkit's
    bucketed batching is preserved within each rank. Attribute access is
    forwarded so epoch-setup hooks and cache flags keep working.
    """

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __len__(self):
        n = len(self.dataset)
        return n // self.world_size + (1 if self.rank < n % self.world_size else 0)

    def __getitem__(self, idx):
        return self.dataset[idx * self.world_size + self.rank]

    def __getattr__(self, name):
        # only called when normal lookup fails; forward to wrapped dataset
        return getattr(self.dataset, name)
