import gc

import torch


def value_map(inputs, min_in, max_in, min_out, max_out):
    return (inputs - min_in) * (max_out - min_out) / (max_in - min_in) + min_out


def flush(garbage_collect=True):
    torch.cuda.empty_cache()
    if garbage_collect:
        gc.collect()
