# itorch — a minimal autograd deep learning library

A tiny PyTorch-like library, blank-film-style, inspired by [tinygrad](https://github.com/tinygrad/tinygrad).
The entire autograd engine is one file (`itorch/tensor.py`); everything else is a thin layer on top.
Only dependency: **numpy**. Runs on CPU.

```python
import itorch as tt
from itorch import nn, optim, Tensor, cross_entropy

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt   = optim.AdamW(model.parameters(), lr=1e-3)

for x, y in dataloader:          # x: Tensor(64, 784)
    opt.zero_grad()
    loss = cross_entropy(model(x), y)
    loss.backward()
    opt.step()
```

## what's inside

| area | pieces |
|------|--------|
| autograd | `Tensor` with reverse-mode autograd: `+ - * / ** @`, `matmul`, `sum/mean/var/std/max/min`, `exp/log/sqrt/abs`, `relu/sigmoid/tanh/softmax/log_softmax`, `reshape/transpose/permute/unsqueeze/squeeze/flatten`, slicing `x[i]`, `gather`, `cat/stack/where`, `im2col` (used by conv) |
| nn | `Module`, `Sequential`, `Linear`, `Conv2d` (via im2col), `MaxPool2d`, `BatchNorm1d/2d`, `Dropout`, `Embedding`, `ReLU/Sigmoid/Tanh/Softmax/LogSoftmax/LeakyReLU`, `Flatten` |
| optim | `SGD` (momentum/nesterov), `Adam`, `AdamW` |
| losses | `mse`, `bce`, `cross_entropy`, `accuracy` |
| utils | `no_grad`, `manual_seed`, `clip_grad_norm_`, `TensorDataset`, `DataLoader`, `save`/`load` (npz) |

## run it

```bash
python3 tests/test_autograd.py   # numeric-gradient check of every op vs finite differences
python3 tests/test_train.py      # end-to-end training smoke tests
python3 examples/01_xor.py
python3 examples/02_linear_regression.py
python3 examples/03_mnist_cnn.py --epochs 1   # downloads MNIST (~11 MB) and trains a CNN
```

## how the autograd works

Each `Tensor` wraps a numpy array and records

* `_parents` — the input tensors of the op that produced it
* `_backward(grad)` — a closure that maps the gradient of the output to gradients of the parents

`backward()` walks the graph once in topological order; each closure accumulates
(`p.grad += partial`) into its parents. Everything else — broadcasting reduction,
`no_grad`, module parameter collection, optimizers — is sugar around this.

Example: `z = (x * 2 + 1).sum()` builds `z → sum ← add ← mul ← x`; calling `z.backward()`
runs `∂z/∂(add)` → `∂add/∂(mul)` → `∂mul/∂x = 2`, leaving `x.grad == [2,2,...]`.

## design notes

* **Float32 everywhere**, CPU only, `Tensor(x).item()/numpy()` for inspection.
* **im2col-style conv**: `Conv2d` unfolds the input with `sliding_window_view` and does a
  single matmul; gradscatter back is overlap-add via `np.add.at`.
* **Used for real training** — see `examples/03_mnist_cnn.py` for the standard
  conv → relu → maxpool → linear stack: **98.3% test accuracy after 3 epochs
  (~2 minutes on a laptop CPU)**.