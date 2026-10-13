"""Fit a synthetic linear model y = W x + b with pure gradient descent.

Run: python3 examples/02_linear_regression.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import optim, Tensor, mse

np.random.seed(0)

# y = x · w_true + b_true
N, D = 128, 5
x = Tensor(np.random.randn(N, D))
w_true = np.array([1.5, -2.0, 3.0, -0.5, 2.5], dtype=np.float32)
y = Tensor(x.data @ w_true.reshape(-1, 1) + 4.0)  # (N, 1)

w = Tensor(np.zeros((D, 1)), requires_grad=True)
b = Tensor(np.zeros(1), requires_grad=True)
opt = optim.SGD([w, b], lr=0.05)

for step in range(250):
    opt.zero_grad()
    pred = x @ w + b
    loss = mse(pred, y)
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss={float(loss.data):.2e}")

print("learned w:", w.data.flatten().round(3))
print("true    w:", w_true)
print("learned b:", round(float(b.item()), 3), " true b: 4.0")
assert float(mse(x @ w + b, y).item()) < 1e-3
print("linear regression converged ✓")