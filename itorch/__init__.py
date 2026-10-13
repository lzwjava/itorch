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
from .tensor import (tensor, as_tensor, from_numpy, zeros, ones, full, arange, randn, rand,
                     randn_like, rand_like, zeros_like, ones_like)
from . import functions, nn, optim, schedulers
from .functions import (no_grad, manual_seed, grad, gradcheck, profile,
                        cat, stack, where,
                        exp, log, sqrt, relu, gelu, silu, softplus, sin, cos,
                        sigmoid, tanh, softmax, log_softmax,
                        max, min, sum, mean,
                        mse, l1, huber, bce, cross_entropy, nll_loss, accuracy,
                        clip_grad_norm_, count_parameters,
                        TensorDataset, DataLoader,
                        save, load)

__version__ = "0.4.0"

__all__ = [
    "Tensor", "tensor", "as_tensor", "from_numpy", "zeros", "ones", "full", "arange", "randn", "rand",
    "randn_like", "rand_like", "zeros_like", "ones_like",
    "nn", "optim", "functions", "schedulers",
    "no_grad", "manual_seed", "grad", "gradcheck", "profile",
    "cat", "stack", "where",
    "exp", "log", "sqrt", "relu", "gelu", "silu", "softplus", "sin", "cos",
    "sigmoid", "tanh", "softmax", "log_softmax",
    "max", "min", "sum", "mean",
    "mse", "l1", "huber", "bce", "cross_entropy", "nll_loss", "accuracy",
    "clip_grad_norm_", "count_parameters", "TensorDataset", "DataLoader", "save", "load",
]