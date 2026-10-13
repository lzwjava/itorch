"""Optimizers: SGD (momentum/nesterov), Adam, AdamW."""
from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.nesterov = nesterov
        self.buf = {}

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                b = self.buf.get(i)
                if b is None:
                    b = np.zeros_like(p.data)
                b = self.momentum * b + g
                self.buf[i] = b
                g = g + self.momentum * b if self.nesterov else b
            p.data -= self.lr * g


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            m = b1 * self.m.get(i, 0.0) + (1 - b1) * g
            v = b2 * self.v.get(i, 0.0) + (1 - b2) * g * g
            self.m[i], self.v[i] = m, v
            mhat = m / (1 - b1 ** self.t)
            vhat = v / (1 - b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay=0.0)
        self.decay = weight_decay

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            m = b1 * self.m.get(i, 0.0) + (1 - b1) * g
            v = b2 * self.v.get(i, 0.0) + (1 - b2) * g * g
            self.m[i], self.v[i] = m, v
            mhat = m / (1 - b1 ** self.t)
            vhat = v / (1 - b2 ** self.t)
            p.data -= self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.decay * p.data)


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.eps = eps
        self.state = {}

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            s = self.state.get(i, 0.0)
            s = s + g * g
            self.state[i] = s
            p.data -= self.lr * g / (np.sqrt(s) + self.eps)


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.state = {}
        self.buf = {}

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            s = self.state.get(i, 0.0)
            s = self.alpha * s + (1 - self.alpha) * g * g
            self.state[i] = s
            step = g / (np.sqrt(s) + self.eps)
            if self.momentum:
                b = self.buf.get(i)
                if b is None:
                    b = np.zeros_like(p.data)
                b = self.momentum * b + step
                self.buf[i] = b
                step = b
            p.data -= self.lr * step


class Adamax(Optimizer):
    """Adam with the max (infinity) norm instead of the L2 norm."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = {}
        self.u = {}

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            m = b1 * self.m.get(i, 0.0) + (1 - b1) * g
            u = np.maximum(self.u.get(i, 0.0) * b2, np.abs(g))
            self.m[i], self.u[i] = m, u
            mhat = m / (1 - b1 ** self.t)
            p.data -= self.lr * mhat / (u + self.eps)


class Lion(Optimizer):
    """Lion (Google, 2023): sign-based update, no adaptive second moment."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, lr, weight_decay=0.0)
        self.betas = betas
        self.decay = weight_decay
        self.m = {}

    def step(self):
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad + self.decay * p.data     # decoupled weight decay
            prev = self.m.get(i, 0.0)            # m_{t-1}
            c = b1 * prev + (1 - b1) * g
            self.m[i] = c
            p.data -= self.lr * np.sign(b2 * prev + (1 - b2) * g)