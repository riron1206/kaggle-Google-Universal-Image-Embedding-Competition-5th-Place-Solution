import torch
import numpy as np

def x_mixup(x, y, a: float = 1.0, enable: bool = True):
    a = np.clip(a, 0.0, 1.0)
    if enable and np.random.rand() >= 0.5:
        j = torch.randperm(x.size(0))
        u = x[j]
        z = y[j]
        a = np.random.beta(a, a)
        w = a * x + (1.0 - a) * u
        return w, y, z, a, True
    return x, y, y, 1.0, False
