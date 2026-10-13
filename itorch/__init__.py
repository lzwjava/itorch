"""itorch: a minimal autograd / deep learning library, tinygrad-style.

Requires only numpy. Fully usable for training small models on CPU:

    import itorch as tt
    from itorch import nn, optim, Tensor, cross_entropy

    x = Tensor.randn(8, 3)
    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2))
    opt = optim.Adam(model.parameters(), lr=0.01)
    ...
"""
from .tensor import Tensor, dtype
from .tensor import (tensor, as_tensor, from_numpy, zeros, ones, full, arange, randn, rand)
from . import tensor as _tensor_mod
from . import functions, nn, optim
from .functions import (no_grad, manual_seed,
                        cat, stack, where,
                        exp, log, sqrt, relu, sigmoid, tanh, softmax, log_softmax,
                        max, min, sum, mean,
                        mse, bce, cross_entropy, accuracy,
                        clip_grad_norm_,
                        TensorDataset, DataLoader,
                        save, load)

__version__ = "0.1.0"

__all__ = [
    "Tensor", "tensor", "as_tensor", "from_numpy", "zeros", "ones", "full", "arange", "randn", "rand",
    "nn", "optim", "functions",
    "no_grad", "manual_seed",
    "cat", "stack", "where",
    "exp", "log", "sqrt", "relu", "sigmoid", "tanh", "softmax", "log_softmax",
    "max", "min", "sum", "mean",
    "mse", "bce", "cross_entropy", "accuracy",
    "clip_grad_norm_", "TensorDataset", "DataLoader", "save", "load",
]