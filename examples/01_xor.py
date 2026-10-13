"""The classic XOR problem — proves a nonlinear MLP can learn non-linear boundaries.

Run: python3 examples/01_xor.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, mse

np.random.seed(0)

X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
Y = Tensor(np.array([0, 1, 1, 0], dtype=np.float32))

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
opt = optim.Adam(model.parameters(), lr=0.01)

for step in range(1000):
    opt.zero_grad()
    loss = mse(model(X).flatten(), Y)
    loss.backward()
    opt.step()
    if step in (0, 99, 299, 999) or (step + 1) % 200 == 0:
        print(f"step {step:4d}  loss={float(loss.data):.4f}")

pred = np.round(model(X).flatten().data)
print("predictions:", pred.astype(int), " targets:", Y.data.astype(int))
assert np.array_equal(pred, Y.data), "XOR not learned correctly"
print("XOR solved ✓")