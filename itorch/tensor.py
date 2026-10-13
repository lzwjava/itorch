"""A minimal autograd engine, tinygrad-style.

A Tensor wraps a numpy array and tracks the graph of operations that produced it.
Each node stores `_parents` (the inputs of the op) and `_backward` (a closure that,
given the gradient of the output, propagates gradients to the parents).
`backward()` performs a topological traversal and calls those closures in order.
"""
from __future__ import annotations

import sys
import numpy as np

if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)

dtype = np.float32          # default dtype for all tensors
_grad_enabled = True        # toggled by itorch.no_grad()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _norm_axes(ndim, axis):
    """Normalize axis/axes to positive ints (axis=None -> None)."""
    if axis is None:
        return None
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(a % ndim for a in axis)


def _unbroadcast(g, shape):
    """Reduce grad array to `shape`, summing over broadcast dimensions."""
    if g.shape == tuple(shape):
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if g.shape[i] != s:
            g = g.sum(axis=i, keepdims=True)
    return g


def _matmul_grads(a, b, g):
    """Gradients of np.matmul(a, b) = g, returned as (grad_a, grad_b)."""
    if a.ndim == 1 and b.ndim == 1:
        return g * b, g * a
    if a.ndim == 1:                      # (m,) @ (m, n) -> (n,)
        return b @ g, np.outer(a, g)
    if b.ndim == 1:                      # (m, n) @ (n,) -> (m,)
        return np.outer(g, b), a.T @ g
    ga = np.matmul(g, np.swapaxes(b, -1, -2))
    gb = np.matmul(np.swapaxes(a, -1, -2), g)
    return ga, gb


def _erf(x):
    """Vectorized erf via Abramowitz & Stegun 7.1.26 (~1.5e-7 max error).
    Backward uses the derivative of THIS formula, so autograd stays exact."""
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    p = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    y = 1.0 - p * t * np.exp(-ax * ax)
    return np.where(x >= 0, y, -y)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _gelu_grad(x):
    z = x / np.sqrt(2.0)
    return 0.5 * (1.0 + _erf(z)) + x * np.exp(-z * z) / np.sqrt(2.0 * np.pi)


def _as_tensor(data, requires_grad=False):
    if isinstance(data, Tensor):
        return data
    return Tensor(data, requires_grad=requires_grad)


def _requires(*ts):
    """True iff an op must build a graph (gradient enabled + any input needs grad)."""
    if not _grad_enabled:
        return False
    return any(t is not None and t.requires_grad for t in ts)


# --------------------------------------------------------------------------
# Tensor
# --------------------------------------------------------------------------

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=dtype) if not isinstance(data, np.ndarray) else np.asarray(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        self.grad = None                  # accumulated gradient (numpy array)
        self._parents = ()                # input Tensors of the op that made this
        self._backward = lambda grad: None
        self._op = None                   # op name, for grad_fn debugging

    # -- bookkeeping -------------------------------------------------------
    def _set(self, parents, backward):
        # name the op from the calling method (e.g. '__add__', 'matmul', 'relu')
        try:
            self._op = sys._getframe(1).f_code.co_name
        except Exception:
            self._op = "?"
        self._parents = tuple(parents)
        self._backward = backward
        return self

    @property
    def grad_fn(self):
        """A human-readable description of the op that produced this tensor
        (None for user-created leaves and for freed graphs)."""
        if self._backward is None or self._op is None:
            return None
        return f"{self._op}({','.join(str(p.shape) for p in self._parents)}) -> {self.data.shape}"

    def _accumulate(self, g):
        if self.requires_grad and g is not None:
            self.grad = g if self.grad is None else self.grad + g

    # -- public api ---------------------------------------------------------
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    @property
    def size(self): return self.data.size
    @property
    def dtype(self): return self.data.dtype
    @property
    def device(self): return "cpu"

    def numpy(self):       return self.data
    def to_numpy(self):    return self.data
    def item(self):        return self.data.item()
    def tolist(self):      return self.data.tolist()
    def to(self, *_a, **_k): return self
    def cpu(self):         return self

    def __array__(self, dtype=None):
        # let numpy treat a Tensor as a read-only view of its data (np.asarray(t), etc.)
        return np.asarray(self.data, dtype=dtype)

    def detach(self):
        """Return a new Tensor sharing data but detached from the graph."""
        return Tensor(self.data)

    def requires_grad_(self, v=True):
        self.requires_grad = bool(v)
        return self

    # -- autograd -----------------------------------------------------------
    def backward(self, grad=None, retain_graph=False):
        """
        Compute gradients by reverse-mode autograd.

        The graph is freed after the pass unless `retain_graph=True` (call
        backward() twice on the same graph only with retain_graph=True).
        """
        if self._backward is None:
            raise RuntimeError("trying to backward() through a freed graph; "
                               "use retain_graph=True to support repeated backward calls")
        if grad is None:
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=dtype)

        def visit(t, topo, seen):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p, topo, seen)
            topo.append(t)

        topo = []
        visit(self, topo, set())
        for t in topo:
            t.grad = None
        self._accumulate(grad)
        for t in reversed(topo):
            if t.requires_grad:
                t._backward(t.grad)
        if not retain_graph:
            # free the graph (keep user-created leaf nodes usable for new graphs)
            for t in topo:
                if t._op is not None:
                    t._backward = None
                    t._parents = ()
        return self

    # -- arithmetic -----------------------------------------------------------
    def __add__(self, o):
        o = _as_tensor(o)
        out = Tensor(self.data + o.data, requires_grad=_requires(self, o))
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                if self.requires_grad:
                    self._accumulate(_unbroadcast(g, self.data.shape))
                if o.requires_grad:
                    o._accumulate(_unbroadcast(g, o.data.shape))
            out._set((self, o), backward)
        return out
    __radd__ = lambda self, o: self.__add__(o)

    def __sub__(self, o):
        o = _as_tensor(o)
        out = Tensor(self.data - o.data, requires_grad=_requires(self, o))
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                if self.requires_grad:
                    self._accumulate(_unbroadcast(g, self.data.shape))
                if o.requires_grad:
                    o._accumulate(_unbroadcast(-g, o.data.shape))
            out._set((self, o), backward)
        return out

    def __rsub__(self, o): return _as_tensor(o).__sub__(self)

    def __mul__(self, o):
        o = _as_tensor(o)
        out = Tensor(self.data * o.data, requires_grad=_requires(self, o))
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                if self.requires_grad:
                    self._accumulate(_unbroadcast(g * o.data, self.data.shape))
                if o.requires_grad:
                    o._accumulate(_unbroadcast(g * self.data, o.data.shape))
            out._set((self, o), backward)
        return out
    __rmul__ = lambda self, o: self.__mul__(o)

    def __truediv__(self, o):
        o = _as_tensor(o)
        out = Tensor(self.data / o.data, requires_grad=_requires(self, o))
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                if self.requires_grad:
                    self._accumulate(_unbroadcast(g / o.data, self.data.shape))
                if o.requires_grad:
                    o._accumulate(_unbroadcast(g * (-self.data / (o.data ** 2)), o.data.shape))
            out._set((self, o), backward)
        return out

    def __rtruediv__(self, o): return _as_tensor(o).__truediv__(self)

    def __pow__(self, exp):
        if not isinstance(exp, (int, float, np.number)):
            raise TypeError("only scalar exponents are supported")
        out = Tensor(self.data ** exp, requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                with np.errstate(divide="ignore", invalid="ignore"):
                    self._accumulate(g * (exp * self.data ** (exp - 1)))
            out._set((self,), backward)
        return out
    pow = __pow__

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            out._set((self,), lambda g: self._accumulate(-np.asarray(g, dtype=dtype)))
        return out

    def __abs__(self):
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(g * np.sign(self.data))
            out._set((self,), backward)
        return out

    def abs(self):
        return self.__abs__()

    def __matmul__(self, o):
        o = _as_tensor(o)
        out = Tensor(self.data @ o.data, requires_grad=_requires(self, o))
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                ga, gb = _matmul_grads(self.data, o.data, g)
                if self.requires_grad:
                    self._accumulate(_unbroadcast(ga, self.data.shape))
                if o.requires_grad:
                    o._accumulate(_unbroadcast(gb, o.data.shape))
            out._set((self, o), backward)
        return out

    # -- reductions -----------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        axes = _norm_axes(self.data.ndim, axis)
        out = Tensor(self.data.sum(axis=None if axes is None else tuple(axes), keepdims=keepdims),
                     requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                if axes is None or keepdims:
                    # gradient already has 1s on reduced axes (or is scalar)
                    self._accumulate(np.broadcast_to(g, self.data.shape).astype(dtype))
                else:
                    shape = list(self.data.shape)
                    for ax in axes:
                        shape[ax] = 1            # put back a size-1 at original position
                    g = np.reshape(g, shape)
                    self._accumulate(np.broadcast_to(g, self.data.shape).astype(dtype))
            out._set((self,), backward)
        return out

    def mean(self, axis=None, keepdims=False):
        axes = _norm_axes(self.data.ndim, axis)
        n = self.data.size if axes is None else int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / n

    def var(self, axis=None, keepdims=False):
        m = self.mean(axis=axis, keepdims=True)
        return (self - m).pow(2).mean(axis=axis, keepdims=keepdims)

    def std(self, axis=None, keepdims=False):
        return self.var(axis=axis, keepdims=keepdims).pow(0.5)

    def max(self, axis=None, keepdims=False):
        axes = _norm_axes(self.data.ndim, axis)
        out = Tensor(np.max(self.data, axis=None if axes is None else tuple(axes), keepdims=keepdims),
                     requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                if axes is None:
                    zero = np.zeros_like(self.data)
                    ix = np.unravel_index(np.argmax(self.data), self.data.shape)
                    zero[ix] = g.reshape(())
                    self._accumulate(zero)
                else:
                    amax = np.max(self.data, axis=tuple(axes), keepdims=True)
                    mask = (self.data == amax)
                    total = mask.sum(axis=tuple(axes), keepdims=True)
                    frac = np.divide(mask.astype(dtype), total.astype(dtype),
                                     out=np.zeros_like(self.data, dtype=dtype), where=total != 0)
                    if keepdims:
                        g = np.broadcast_to(g, self.data.shape).astype(dtype)
                    else:
                        shape = list(self.data.shape)
                        for ax in axes:
                            shape[ax] = 1
                        g = np.reshape(g, shape)
                        g = np.broadcast_to(g, self.data.shape).astype(dtype)
                    self._accumulate(frac * g)
            out._set((self,), backward)
        return out

    def min(self, axis=None, keepdims=False):
        return -(-self).max(axis=axis, keepdims=keepdims)

    def argmax(self, axis=None):
        return np.argmax(self.data, axis=axis)

    def argmin(self, axis=None):
        return np.argmin(self.data, axis=axis)

    # -- nonlinearities --------------------------------------------------------
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(g * out.data)
            out._set((self,), backward)
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                with np.errstate(divide="ignore", invalid="ignore"):
                    self._accumulate(g / self.data)
            out._set((self,), backward)
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                with np.errstate(divide="ignore", invalid="ignore"):
                    self._accumulate(g * 0.5 / out.data)
            out._set((self,), backward)
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(g * (self.data > 0))
            out._set((self,), backward)
        return out

    def leaky_relu(self, alpha=0.01):
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data),
                     requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(g * np.where(self.data > 0, 1.0, alpha))
            out._set((self,), backward)
        return out

    def gelu(self):
        out = Tensor(_gelu(self.data), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(g * _gelu_grad(self.data))
            out._set((self,), backward)
        return out

    def sigmoid(self):
        out = Tensor(1.0 / (1.0 + np.exp(-self.data)), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                s = out.data
                self._accumulate(g * s * (1.0 - s))
            out._set((self,), backward)
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                t = out.data
                self._accumulate(g * (1.0 - t * t))
            out._set((self,), backward)
        return out

    def softmax(self, axis=-1):
        m = self.max(axis=axis, keepdims=True)
        e = (self - m).exp()
        return e / e.sum(axis=axis, keepdims=True)

    def log_softmax(self, axis=-1):
        m = self.max(axis=axis, keepdims=True)
        s = (self - m).exp().sum(axis=axis, keepdims=True)
        return (self - m) - s.log()

    def clip(self, lo, hi):
        out = Tensor(np.clip(self.data, lo, hi), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(g * ((self.data >= lo) & (self.data <= hi)))
            out._set((self,), backward)
        return out

    # -- shape ops -------------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(shape)
        # resolve -1
        target = list(shape)
        if -1 in target:
            i = target.index(-1)
            n = 1
            for j, v in enumerate(target):
                if j != i:
                    n *= v
            target[i] = self.data.size // n
        target = tuple(int(v) for v in target)
        out = Tensor(self.data.reshape(target), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(g.reshape(self.data.shape))
            out._set((self,), backward)
        return out
    view = reshape

    def unsqueeze(self, dim):
        nd = self.data.ndim + 1
        dim = dim if dim >= 0 else dim + nd
        return self.reshape(self.data.shape[:dim] + (1,) + self.data.shape[dim:])

    def squeeze(self, dim=None):
        if dim is None:
            new_shape = tuple(s for s in self.data.shape if s != 1)
        else:
            if self.data.shape[dim] != 1:
                return self
            new_shape = self.data.shape[:dim] + self.data.shape[dim + 1:]
        return self.reshape(new_shape)

    def flatten(self, start_dim=0, end_dim=-1):
        shp = list(self.data.shape)
        if start_dim < 0:
            start_dim += len(shp)
        if end_dim < 0:
            end_dim += len(shp)
        n = int(np.prod(shp[start_dim:end_dim + 1]))
        new = shp[:start_dim] + [n] + shp[end_dim + 1:]
        return self.reshape(*new)

    def transpose(self, dim0, dim1):
        d0 = dim0 % self.data.ndim
        d1 = dim1 % self.data.ndim
        out = Tensor(np.swapaxes(self.data, d0, d1), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(np.swapaxes(g, d0, d1))
            out._set((self,), backward)
        return out

    def permute(self, *axes):
        axes = tuple(a % self.data.ndim for a in axes)
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            inv = tuple(np.argsort(axes))
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                self._accumulate(np.transpose(g, inv))
            out._set((self,), backward)
        return out

    @property
    def t(self):  # transpose last two dims
        if self.data.ndim < 2:
            return self
        return self.transpose(-2, -1)
    T = t

    # -- indexing / gathering ----------------------------------------------------
    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data
        out = Tensor(self.data[idx], requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                zero = np.zeros_like(self.data)
                if np.asarray(idx).dtype == bool and np.asarray(idx).shape == self.data.shape:
                    zero[idx] += g          # boolean mask: unique positions, safe
                else:
                    np.add.at(zero, idx, g)  # handles duplicates
                self._accumulate(zero)
            out._set((self,), backward)
        return out

    def gather(self, dim, index):
        """torch.gather: collect values along `dim` at `index` (same shape as index)."""
        dim = dim % self.data.ndim
        index = np.asarray(index.data if isinstance(index, Tensor) else index, dtype=np.int64)
        out = Tensor(np.take_along_axis(self.data, index, axis=dim),
                     requires_grad=self.requires_grad and _grad_enabled)
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                zero = np.zeros_like(self.data)
                inds = []
                for d, s in enumerate(self.data.shape):
                    if d == dim:
                        inds.append(index)
                    else:
                        shp = [1] * self.data.ndim
                        shp[d] = s
                        inds.append(np.arange(s).reshape(shp))
                ind = np.broadcast_arrays(*inds)
                np.add.at(zero, tuple(a if a.shape == index.shape else np.broadcast_to(a, index.shape)
                                      for a in ind), g)
                self._accumulate(zero)
            out._set((self,), backward)
        return out

    # -- convolution helper (unfold / fold) -----------------------------------------
    def im2col(self, kernel_size, padding=0, stride=1):
        """Unfold (N, C, H, W) into columns (N, C*K*K, OH*OW).

        Backward scatters gradients back into the input (col2im) with overlap-add.
        """
        k = _pair(kernel_size)
        p = _pair(padding)
        s = _pair(stride)
        kh, kw = k
        N, C, H, W = self.data.shape
        xp = np.pad(self.data, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])))
        Hp, Wp = xp.shape[-2], xp.shape[-1]
        oh = (Hp - kh) // s[0] + 1
        ow = (Wp - kw) // s[1] + 1
        win = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(-2, -1))
        win = win[:, :, ::s[0], ::s[1]]                     # (N, C, OH, OW, kh, kw)
        cols = win.transpose(0, 1, 4, 5, 2, 3).reshape(N, C * kh * kw, oh * ow)
        out = Tensor(np.array(cols, copy=True), requires_grad=self.requires_grad and _grad_enabled)
        out._oh, out._ow = oh, ow
        if out.requires_grad:
            def backward(g):
                g = np.asarray(g, dtype=dtype)
                gr = g.reshape(N, C, kh, kw, oh, ow).transpose(0, 1, 4, 5, 2, 3)  # (N,C,OH,OW,kh,kw)
                gp = np.zeros((N, C, Hp, Wp), dtype=dtype)
                an = np.arange(N)[:, None, None, None, None, None]
                ac = np.arange(C)[None, :, None, None, None, None]
                aa = np.arange(oh)[None, None, :, None, None, None]
                ab = np.arange(ow)[None, None, None, :, None, None]
                ai = np.arange(kh)[None, None, None, None, :, None]
                aj = np.arange(kw)[None, None, None, None, None, :]
                np.add.at(gp, (np.broadcast_to(an, gr.shape),
                               np.broadcast_to(ac, gr.shape),
                               np.broadcast_to(aa * s[0] + ai, gr.shape),
                               np.broadcast_to(ab * s[1] + aj, gr.shape)), gr)
                crop = gp[:, :, p[0]:p[0] + H, p[1]:p[1] + W]
                self._accumulate(crop)
            out._set((self,), backward)
        return out

    # -- dunders ---------------------------------------------------------------------
    def __eq__(self, o): return Tensor(self.data == _as_tensor(o).data)
    def __ne__(self, o): return Tensor(self.data != _as_tensor(o).data)
    def __lt__(self, o): return Tensor(self.data < _as_tensor(o).data)
    def __le__(self, o): return Tensor(self.data <= _as_tensor(o).data)
    def __gt__(self, o): return Tensor(self.data > _as_tensor(o).data)
    def __ge__(self, o): return Tensor(self.data >= _as_tensor(o).data)
    __hash__ = object.__hash__

    def __bool__(self):
        if self.data.size != 1:
            raise ValueError("bool value of a Tensor with more than one element is ambiguous")
        return bool(self.data)

    def __len__(self): return len(self.data)

    def __repr__(self):
        return (f"Tensor(shape={tuple(self.data.shape)}, dtype={self.data.dtype}, "
                f"requires_grad={self.requires_grad})\n{self.data}")


# --------------------------------------------------------------------------
# convenience constructors (also re-exported from itorch/__init__.py)
# --------------------------------------------------------------------------

def tensor(data, requires_grad=False): return Tensor(data, requires_grad=requires_grad)
def as_tensor(data): return _as_tensor(data)
def from_numpy(data): return Tensor(data)

def zeros(*shape, requires_grad=False): return Tensor(np.zeros(shape, dtype=dtype), requires_grad)
def ones(*shape, requires_grad=False): return Tensor(np.ones(shape, dtype=dtype), requires_grad)
def full(shape, fill_value, requires_grad=False): return Tensor(np.full(shape, fill_value, dtype=dtype), requires_grad)
def arange(start, stop=None, step=1, requires_grad=False):
    return Tensor(np.arange(start, stop, step, dtype=dtype), requires_grad)
def randn(*shape, requires_grad=False): return Tensor(np.random.randn(*shape).astype(dtype), requires_grad)
def rand(*shape, requires_grad=False): return Tensor(np.random.rand(*shape).astype(dtype), requires_grad)