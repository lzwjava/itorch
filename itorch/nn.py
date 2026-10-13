"""Neural network building blocks built on itorch.Tensor autograd."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor, dtype


# --------------------------------------------------------------------------
# base Module
# --------------------------------------------------------------------------

class Module:
    training = True

    def __call__(self, *x):
        return self.forward(x[0] if len(x) == 1 else x)

    def forward(self, *x):
        raise NotImplementedError

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def named_parameters(self, prefix=""):
        for name, v in self.__dict__.items():
            if isinstance(v, Module):
                yield from v.named_parameters(prefix + name + ".")
            elif isinstance(v, Tensor):
                if v.requires_grad:
                    yield prefix + name, v
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    path = f"{prefix}{name}.{i}"
                    if isinstance(item, Module):
                        yield from item.named_parameters(path + ".")
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield path, item
            elif isinstance(v, dict):
                for k, item in v.items():
                    path = f"{prefix}{name}.{k}"
                    if isinstance(item, Module):
                        yield from item.named_parameters(path + ".")
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield path, item

    def state_dict(self):
        return dict(self.named_parameters())

    def load_state_dict(self, sd):
        for name, p in self.named_parameters():
            if name in sd:
                v = sd[name]
                p.data = np.asarray(v.data if isinstance(v, Tensor) else v, dtype=dtype)

    def _set_training(self, mode):
        self.training = mode
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v._set_training(mode)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item._set_training(mode)
            elif isinstance(v, dict):
                for item in v.values():
                    if isinstance(item, Module):
                        item._set_training(mode)

    def train(self, mode=True):
        self._set_training(mode)
        return self

    def eval(self):
        return self.train(False)

    def __repr__(self):
        n = sum(p.data.size for p in self.parameters())
        return f"{self.__class__.__name__}({n:,} params)"

    def __str__(self):  # pretty tree
        lines = [f"{self.__class__.__name__}({sum(p.data.size for p in self.parameters()):,} params) -> {{"] if self.parameters() else [f"{self.__class__.__name__} -> {{"]

        def walk(m, indent):
            for name, v in m.__dict__.items():
                if isinstance(v, Module):
                    lines.append(f"  {' '*indent}{name}: {v.__class__.__name__},")
                    walk(v, indent + 2)
                elif isinstance(v, Tensor) and v.requires_grad:
                    lines.append(f"  {' '*indent}{name}: Tensor{tuple(v.shape)},")
                elif isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        if isinstance(item, Module):
                            lines.append(f"  {' '*indent}{name}.{i}: {item.__class__.__name__},")
                            walk(item, indent + 2)

        walk(self, 2)
        lines.append("}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

    def __repr__(self):
        parts = ",\n  ".join(f"{m.__class__.__name__}({sum(p.data.size for p in m.parameters()):,} params)"
                             for m in self.modules)
        return f"{self.__class__.__name__}(\n  {parts}\n)"


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming init
        self.weight = Tensor(np.random.randn(in_features, out_features).astype(dtype) * np.sqrt(2.0 / in_features),
                             requires_grad=True)
        self.bias = Tensor(np.zeros(out_features, dtype=dtype), requires_grad=True) if bias else None

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim).astype(dtype) * 0.1,
                             requires_grad=True)

    def forward(self, idx):
        # idx: integer tensor / array; duplicates handled by np.add.at in backward
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx, dtype=np.int64)
        return self.weight[idx]


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.weight = Tensor(np.random.randn(out_channels, in_channels, k, k).astype(dtype)
                             * np.sqrt(2.0 / (in_channels * k * k)),
                             requires_grad=True)
        self.bias = Tensor(np.zeros(out_channels, dtype=dtype), requires_grad=True) if bias else None

    def forward(self, x):
        N, C, H, W = x.shape
        k, p, s = self.kernel_size, self.padding, self.stride
        cols = x.im2col(k, p, s)                      # (N, C k^2, OH*OW)
        y = self.weight.reshape(self.out_channels, -1) @ cols  # (N, O, OH*OW)
        y = y.reshape(N, self.out_channels, cols._oh, cols._ow)
        if self.bias is not None:
            y = y + self.bias.reshape(1, self.out_channels, 1, 1)
        return y


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        cols = x.im2col(k, self.padding, self.stride)     # (N, Ck^2, OH*OW)
        kk = k * k
        out = cols.reshape(N, C, kk, cols._oh * cols._ow).max(2)
        return out.reshape(N, C, cols._oh, cols._ow)


# --------------------------------------------------------------------------
# batch normalization
# --------------------------------------------------------------------------

class _BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features, dtype=dtype)
        self.running_var = np.ones(num_features, dtype=dtype)
        self.gamma = Tensor(np.ones(num_features, dtype=dtype), requires_grad=True) if affine else None
        self.beta = Tensor(np.zeros(num_features, dtype=dtype), requires_grad=True) if affine else None

    def _axes(self, ndim):
        raise NotImplementedError

    def _view(self, shape):
        return (1,) + (self.num_features,) + (1,) * (len(shape) - 2)

    def forward(self, x):
        shape = x.shape
        axes = self._axes(len(shape))
        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            var = (x - mean).pow(2).mean(axis=axes, keepdims=True)
            mm = mean.data.reshape(-1)
            vv = var.data.reshape(-1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mm
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * vv
            xhat = (x - mean) / (var + self.eps).sqrt()
        else:
            mean = Tensor(np.broadcast_to(self.running_mean.reshape(self._view(shape)), shape).copy())
            var = Tensor(np.broadcast_to(self.running_var.reshape(self._view(shape)), shape).copy())
            xhat = (x - mean) / (var + self.eps).sqrt()
        if self.gamma is None:
            return xhat
        v = self._view(shape)
        return xhat * self.gamma.reshape(v) + self.beta.reshape(v)


class BatchNorm1d(_BatchNorm):
    """Normalize over the batch dimension: input (N, F)."""
    def _axes(self, ndim):
        return (0,)


class BatchNorm2d(_BatchNorm):
    """Normalize over (N, H, W): input (N, C, H, W)."""
    def _axes(self, ndim):
        return (0, 2, 3)


# --------------------------------------------------------------------------
# activations / regularization
# --------------------------------------------------------------------------

class ReLU(Module):
    def forward(self, x): return x.relu()


class LeakyReLU(Module):
    def __init__(self, alpha=0.01): self.alpha = alpha
    def forward(self, x): return x.leaky_relu(self.alpha)


class Sigmoid(Module):
    def forward(self, x): return x.sigmoid()


class Tanh(Module):
    def forward(self, x): return x.tanh()


class Softmax(Module):
    def __init__(self, axis=-1): self.axis = axis
    def forward(self, x): return x.softmax(self.axis)


class LogSoftmax(Module):
    def __init__(self, axis=-1): self.axis = axis
    def forward(self, x): return x.log_softmax(self.axis)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class Dropout(Module):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(dtype) / (1.0 - self.p)
        return x * Tensor(mask)