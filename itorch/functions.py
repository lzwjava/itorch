"""Functional API, loss functions, data helpers and utilities."""
from __future__ import annotations

import numpy as np

import sys

from .tensor import Tensor, dtype, _as_tensor

# reference the REAL tensor module (the name `tensor` in the package namespace is
# shadowed by the factory function), so no_grad can toggle the shared flag.
_t = sys.modules[__name__.rsplit('.', 1)[0] + '.tensor']

# re-export constructors
from .tensor import tensor, as_tensor, from_numpy, zeros, ones, full, arange, randn, rand


# --------------------------------------------------------------------------
# no_grad context
# --------------------------------------------------------------------------

class no_grad:
    """Disable gradient tracking inside a with-block."""

    def __enter__(self):
        self._prev = _t._grad_enabled
        _t._grad_enabled = False
        return self

    def __exit__(self, *_exc):
        _t._grad_enabled = self._prev
        return False


def manual_seed(seed):
    np.random.seed(seed)


# --------------------------------------------------------------------------
# functional ops
# --------------------------------------------------------------------------

def cat(tensors, dim=0):
    tensors = [_as_tensor(t) for t in tensors]
    dim = dim % tensors[0].data.ndim
    data = np.concatenate([t.data for t in tensors], axis=dim)
    out = Tensor(data, requires_grad=any(t.requires_grad for t in tensors))
    if out.requires_grad:
        sizes = [t.data.shape[dim] for t in tensors]
        offsets = [0] + list(np.cumsum(sizes)[:-1])

        def backward(g):
            g = np.asarray(g, dtype=dtype)
            for t, (off, sz) in zip(tensors, zip(offsets, sizes)):
                sl = [slice(None)] * g.ndim
                sl[dim] = slice(off, off + sz)
                t._accumulate(g[tuple(sl)])
        out._set(tuple(tensors), backward)
    return out


def stack(tensors, dim=0):
    tensors = [_as_tensor(t) for t in tensors]
    d = dim if dim >= 0 else dim - 1
    return cat([t.unsqueeze(d) for t in tensors], dim=dim)


def where(cond, a, b):
    a = _as_tensor(a)
    b = _as_tensor(b)
    cond = cond.data if isinstance(cond, Tensor) else np.asarray(cond, dtype=bool)
    out = Tensor(np.where(cond, a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        def backward(g):
            g = np.asarray(g, dtype=dtype)
            if a.requires_grad:
                a._accumulate(np.where(cond, g, np.zeros_like(g)))
            if b.requires_grad:
                b._accumulate(np.where(cond, np.zeros_like(g), g))
        out._set((a, b), backward)
    return out


def exp(t):   return _as_tensor(t).exp()
def log(t):   return _as_tensor(t).log()
def sqrt(t):  return _as_tensor(t).sqrt()
def relu(t):  return _as_tensor(t).relu()
def sigmoid(t): return _as_tensor(t).sigmoid()
def tanh(t):  return _as_tensor(t).tanh()
def softmax(t, axis=-1):    return _as_tensor(t).softmax(axis=axis)
def log_softmax(t, axis=-1): return _as_tensor(t).log_softmax(axis=axis)
def max(t, axis=None, keepdims=False):    return _as_tensor(t).max(axis=axis, keepdims=keepdims)
def min(t, axis=None, keepdims=False):    return _as_tensor(t).min(axis=axis, keepdims=keepdims)
def sum(t, axis=None, keepdims=False):    return _as_tensor(t).sum(axis=axis, keepdims=keepdims)
def mean(t, axis=None, keepdims=False):   return _as_tensor(t).mean(axis=axis, keepdims=keepdims)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def mse(input, target):
    """Mean squared error between two tensors."""
    input, target = _as_tensor(input), _as_tensor(target)
    return (input - target).pow(2).mean()


def bce(input, target):
    """Binary cross entropy on raw logits."""
    input, target = _as_tensor(input), _as_tensor(target)
    s = input.sigmoid()
    return -(target * s.log() + (1 - target) * (1 - s).log()).mean()


def cross_entropy(logits, targets):
    """Cross entropy on raw logits. logits: (N, C), targets: (N,) ints."""
    logits = _as_tensor(logits)
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets, dtype=np.int64).reshape(-1, 1)
    ls = logits.log_softmax(axis=-1)
    return -ls.gather(-1, targets).mean()


def accuracy(logits, targets):
    """Fraction of correctly predicted classes (no gradients)."""
    logits = _as_tensor(logits)
    targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets)
    with no_grad():
        pred = logits.argmax(axis=-1)
    return float(np.mean(pred.reshape(targets.shape) == targets))


# --------------------------------------------------------------------------
# gradient clipping
# --------------------------------------------------------------------------

def clip_grad_norm_(params, max_norm):
    params = list(params)
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(np.sum(p.grad * p.grad))
    total = float(np.sqrt(sq))
    clip = max_norm / (total + 1e-6)
    if clip < 1:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * clip
    return total


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------

class TensorDataset:
    def __init__(self, *tensors):
        tensors = [np.asarray(t.data if isinstance(t, Tensor) else t) for t in tensors]
        assert len({len(t) for t in tensors}) == 1, "all tensors must have the same length"
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, i):
        return tuple(Tensor(t[i]) for t in self.tensors)


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(idx)
        dataset = self.dataset
        batch_size = self.batch_size
        for i in range(0, len(idx), batch_size):
            batch = idx[i:i + batch_size]
            items = [dataset[j] for j in batch]
            yield tuple(stack([it[k] for it in items], 0) for k in range(len(items[0])))


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------

def save(model, path):
    """Save a Module's parameters (or a single Tensor) to a .npz file."""
    if isinstance(model, Tensor):
        np.savez(path, **{"_data": model.data})
        return
    np.savez(path, **{k: v.data for k, v in model.state_dict().items()})


def load(model, path):
    """Load parameters into a Module (or return a Tensor for .npz with `_data`)."""
    data = dict(np.load(path))
    if isinstance(model, Tensor):
        if "_data" in data:
            model.data = np.asarray(data["_data"], dtype=dtype)
        return model
    sd = {k: np.asarray(v) for k, v in data.items()}
    model.load_state_dict(sd)
    return model