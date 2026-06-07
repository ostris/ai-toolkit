import torch.nn.functional as F

def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y