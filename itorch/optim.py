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