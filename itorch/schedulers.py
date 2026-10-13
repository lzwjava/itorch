"""Learning-rate schedulers. Each one holds a reference to an optimizer and
mutates `optimizer.lr` when `.step()` is called (call once per epoch)."""
from __future__ import annotations

import math


class _Scheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = float(self.get_lr())
        return self.optimizer.lr


class StepLR(_Scheduler):
    """Decay by `gamma` every `step_size` epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class MultiStepLR(_Scheduler):
    """Decay by `gamma` at each listed milestone epoch."""

    def __init__(self, optimizer, milestones, gamma=0.1):
        super().__init__(optimizer)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self):
        decays = sum(1 for m in self.milestones if m <= self.last_epoch)
        return self.base_lr * self.gamma ** decays


class CosineAnnealingLR(_Scheduler):
    """Cosine annealing from the base LR down to `eta_min` over `T_max` epochs."""

    def __init__(self, optimizer, T_max, eta_min=0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        frac = (1.0 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0
        return self.eta_min + (self.base_lr - self.eta_min) * frac


class ReduceLROnPlateau:
    """Reduce LR by `factor` after `patience` epochs without improvement.

    Usage: scheduler.step(validation_metric)
    """

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, min_lr=0.0):
        assert mode in ("min", "max")
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_lr = min_lr
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad = 0

    def step(self, metrics):
        improved = (metrics < self.best) if self.mode == "min" else (metrics > self.best)
        if improved:
            self.best = metrics
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.num_bad = 0
        return self.optimizer.lr