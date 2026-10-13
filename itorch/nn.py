"""Neural network building blocks built on itorch.Tensor autograd."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor, dtype


def _iter_seq(item, path):
    """Iterate parameters inside a list/tuple/dict entry (handles Modules too)."""
    if isinstance(item, Module):
        yield from item._iter_params(path + ".")
    elif isinstance(item, Tensor):
        yield path, item


# --------------------------------------------------------------------------
# base Module
# --------------------------------------------------------------------------

class Module:
    training = True

    def __call__(self, *x):
        if len(x) == 1:
            return self.forward(x[0])
        return self.forward(*x)

    def forward(self, *x):
        raise NotImplementedError

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def _iter_params(self, prefix=""):
        """All parameter Tensors, including ones with requires_grad=False."""
        for name, v in self.__dict__.items():
            if isinstance(v, Module):
                yield from v._iter_params(prefix + name + ".")
            elif isinstance(v, Tensor):
                yield prefix + name, v
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    yield from _iter_seq(item, f"{prefix}{name}.{i}")
            elif isinstance(v, dict):
                for k, item in v.items():
                    yield from _iter_seq(item, f"{prefix}{name}.{k}")

    def named_parameters(self, prefix=""):
        for name, p in self._iter_params(prefix):
            if p.requires_grad:
                yield name, p

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

    def freeze(self):
        """Disable gradient tracking on all parameters (weights are fixed)."""
        for _, p in self._iter_params():
            p.requires_grad = False
        return self

    def unfreeze(self):
        for _, p in self._iter_params():
            p.requires_grad = True
        return self

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


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        cols = x.im2col(k, self.padding, self.stride)     # (N, Ck^2, OH*OW)
        kk = k * k
        out = cols.reshape(N, C, kk, cols._oh * cols._ow).mean(2)
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


class LayerNorm(Module):
    """Normalize over the trailing dimension(s) `normalized_shape`; input (..., F)."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        n = int(np.prod(self.normalized_shape))
        self.gamma = Tensor(np.ones(n, dtype=dtype), requires_grad=True) if elementwise_affine else None
        self.beta = Tensor(np.zeros(n, dtype=dtype), requires_grad=True) if elementwise_affine else None

    def forward(self, x):
        ndim = x.ndim
        d = len(self.normalized_shape)
        axes = tuple(range(ndim - d, ndim))
        mean = x.mean(axis=axes, keepdims=True)
        var = (x - mean).pow(2).mean(axis=axes, keepdims=True)
        xhat = (x - mean) / (var + self.eps).sqrt()
        if self.gamma is None:
            return xhat
        view = (1,) * (ndim - d) + self.normalized_shape
        return xhat * self.gamma.reshape(view) + self.beta.reshape(view)


class GroupNorm(Module):
    """Split `num_channels` into `num_groups` and normalize each group over
    (channels-per-group x spatial); input (N, C, ...)."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        assert num_channels % num_groups == 0, "channels must be divisible by groups"
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = Tensor(np.ones(num_channels, dtype=dtype), requires_grad=True) if affine else None
        self.beta = Tensor(np.zeros(num_channels, dtype=dtype), requires_grad=True) if affine else None

    def forward(self, x):
        N, C = x.shape[0], x.shape[1]
        G = self.num_groups
        rest = int(np.prod(x.shape[2:])) if x.ndim > 2 else 1
        n = C // G
        g = x.reshape(N, G, n * rest)
        mean = g.mean(axis=2, keepdims=True)
        var = (g - mean).pow(2).mean(axis=2, keepdims=True)
        y = ((g - mean) / (var + self.eps).sqrt()).reshape(*x.shape)
        if self.gamma is None:
            return y
        view = (1, C) + (1,) * (x.ndim - 2)
        return y * self.gamma.reshape(view) + self.beta.reshape(view)


# --------------------------------------------------------------------------
# activations / regularization
# --------------------------------------------------------------------------

class ReLU(Module):
    def forward(self, x): return x.relu()


class GELU(Module):
    def forward(self, x): return x.gelu()


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

# --------------------------------------------------------------------------
# recurrent layers (batch_first=True)
# --------------------------------------------------------------------------

class LSTM(Module):
    """A single-layer LSTM. Returned: (outputs (B, T, hidden), (h, c))."""

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        scale = np.sqrt(1.0 / input_size)
        self.weight_ih = Tensor(np.random.randn(input_size, 4 * H).astype(dtype) * scale, requires_grad=True)
        self.weight_hh = Tensor(np.random.randn(H, 4 * H).astype(dtype) * np.sqrt(1.0 / H), requires_grad=True)
        self.bias = Tensor(np.zeros(4 * H, dtype=dtype), requires_grad=True) if bias else None

    def forward(self, x, state=None):
        B = x.shape[0]
        H = self.hidden_size
        h = Tensor(np.zeros((B, H), dtype=dtype)) if state is None else state[0]
        c = Tensor(np.zeros((B, H), dtype=dtype)) if state is None else state[1]
        W_ih, W_hh = self.weight_ih, self.weight_hh
        bias = self.bias
        outs = []
        for t in range(x.shape[1]):
            xt = x[:, t]                                  # (B, in)
            g = xt @ W_ih + h @ W_hh + bias               # (B, 4H)
            i = g[:, 0:H].sigmoid()
            f = g[:, H:2 * H].sigmoid()
            gg = g[:, 2 * H:3 * H].tanh()
            o = g[:, 3 * H:4 * H].sigmoid()
            c = f * c + i * gg
            h = o * c.tanh()
            outs.append(h)
        return stack_t(outs), (h, c)


class GRU(Module):
    """A single-layer GRU. Returned: (outputs (B, T, hidden), h)."""

    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        scale = np.sqrt(1.0 / input_size)
        self.weight_ih = Tensor(np.random.randn(input_size, 3 * H).astype(dtype) * scale, requires_grad=True)
        self.weight_hh = Tensor(np.random.randn(H, 3 * H).astype(dtype) * np.sqrt(1.0 / H), requires_grad=True)
        self.bias = Tensor(np.zeros(3 * H, dtype=dtype), requires_grad=True) if bias else None

    def forward(self, x, state=None):
        B = x.shape[0]
        H = self.hidden_size
        h = Tensor(np.zeros((B, H), dtype=dtype)) if state is None else state
        W_ih, W_hh, bias = self.weight_ih, self.weight_hh, self.bias
        outs = []
        for t in range(x.shape[1]):
            xt = x[:, t]
            g = xt @ W_ih + h @ W_hh + bias               # (B, 3H): r, z, n
            r = g[:, 0:H].sigmoid()
            z = g[:, H:2 * H].sigmoid()
            n = (xt @ W_ih[:, 2 * H:3 * H] + (r * h) @ W_hh[:, 2 * H:3 * H] + bias[2 * H:3 * H]).tanh()
            h = (1 - z) * n + z * h
            outs.append(h)
        return stack_t(outs), h


def stack_t(outs):
    """Stack a list of tensors along dim 1 (local import to avoid a cycle)."""
    from .functions import stack
    return stack(outs, 1)


# --------------------------------------------------------------------------
# attention / transformer
# --------------------------------------------------------------------------

def causal_mask(length):
    """Boolean (length, length) mask: True where timestep i may attend to j (i >= j)."""
    return ~np.triu(np.ones((length, length), dtype=bool), k=1)


class MultiHeadAttention(Module):
    """Self-attention with `num_heads` heads over the last axis. x: (B, T, D)."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.wq = Linear(embed_dim, embed_dim, bias=bias)
        self.wk = Linear(embed_dim, embed_dim, bias=bias)
        self.wv = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.drop = Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, hd = self.num_heads, self.head_dim
        q = self.wq(x).reshape(B, T, H, hd).permute(0, 2, 1, 3)   # (B, H, T, hd)
        k = self.wk(x).reshape(B, T, H, hd).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, H, hd).permute(0, 2, 1, 3)
        scores = q @ k.permute(0, 1, 3, 2) / np.sqrt(hd)         # (B, H, T, T)
        if mask is not None:
            m = Tensor(mask.astype(dtype))
            scores = scores * m - (1 - m) * 1e9
        attn = self.drop(scores.softmax(-1))
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(Module):
    """Pre-norm transformer block: x -> x + Attn(LN(x)) -> x + FF(LN(x))."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = Sequential(Linear(embed_dim, ff_dim), GELU(),
                             Dropout(dropout), Linear(ff_dim, embed_dim), Dropout(dropout))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(Module):
    """Stack of transformer blocks with a final LayerNorm (dryvinec PyTorch style)."""

    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.ln = LayerNorm(embed_dim)

    def forward(self, x):
        mask = causal_mask(x.shape[1])
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)
