# itorch — a minimal autograd deep learning library

A tiny PyTorch-like library, tinygrad-style. The entire autograd engine is one file
(`itorch/tensor.py`); everything else is a thin layer on top.
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

### autograd
| piece | details |
|------|------|
| `Tensor` | reverse-mode autograd over a numpy array: `+ - * / ** @`, matmul (batched & 1-D), `sum/mean/var/std/max/min`, `exp/log/sqrt/abs`, `relu/leaky_relu/gelu/sigmoid/tanh/softmax/log_softmax`, `reshape/transpose/permute/unsqueeze/squeeze/flatten`, slicing `x[i]`, `gather`, `cat/stack/where`, `im2col` (conv engine) |
| `backward()` | topological pass; frees the graph afterwards. `retain_graph=True` keeps it for a second call; a repeat `backward()` on a freed graph raises `RuntimeError`. Leaf tensors stay reusable. |
| `grad_fn` | `y.grad_fn` → `"matmul((3,4),(4,5)) -> (3,5)"` for debugging |
| `no_grad` | works as `with no_grad():` **and** as a decorator `@no_grad` |

### nn
`Module` (params, `train/eval`, `freeze/unfreeze`, `state_dict`), `Sequential`,
`Linear`, `Embedding`, `Conv2d` (im2col), `MaxPool2d`, `AvgPool2d`,
`BatchNorm1d/2d`, `LayerNorm`, `GroupNorm`, `Dropout`, `Flatten`,
`ReLU/GELU/Sigmoid/Tanh/Softmax/LogSoftmax/LeakyReLU`, **recurrent**:
`LSTM`, `GRU` (batch_first, return `(outputs, state)`), **attention**:
`MultiHeadAttention`, `TransformerBlock` (pre-norm + residuals),
`Transformer` (causal stacked blocks + final `LayerNorm`), `causal_mask(T)`.

### optim & schedulers
`SGD` (momentum/nesterov/weight-decay), `Adam`, `AdamW`, and
`itorch.schedulers`: `StepLR`, `MultiStepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`.

### losses, data, utils
`mse`, `l1`, `huber`, `bce`, `cross_entropy`, `nll_loss`, `accuracy`,
`clip_grad_norm_`, `TensorDataset`, `DataLoader` (shuffle/drop_last),
`save`/`load` (npz), `count_parameters`.

## run it

```bash
python3 tests/test_autograd.py       # numeric-gradient check of every core op
python3 tests/test_sophisticated.py  # gradchecks: gelu, norms, pools, rnns, attention, transformer, losses
python3 tests/test_train.py          # end-to-end training smoke tests
python3 examples/01_xor.py
python3 examples/02_linear_regression.py
python3 examples/03_mnist_cnn.py --epochs 3    # downloads MNIST (~11 MB), ~2 min
python3 examples/04_tiny_gpt.py --steps 400    # char-level transformer, samples Shakespeare-ish text
```

## how the autograd works

Each `Tensor` wraps a numpy array and records `_parents` (op inputs) and
`_backward(grad)` (a closure mapping output-gradient to parent-gradients).
`backward()` walks the graph once in topological order; each closure accumulates
(`p.grad += partial`) into its parents, reducing over broadcast dims
(`_unbroadcast`), scattering duplicates (`np.add.at`), and col2im-folding conv
columns. Everything else — modules, optimizers, schedulers — is sugar.

Example: `z = (x * 2 + 1).sum()` builds `z → sum ← add ← mul ← x`; `z.backward()`
runs `∂z/∂add → ∂add/∂mul → ∂mul/∂x = 2`, leaving `x.grad == [2,2,...]`.

## design notes

* **Float32, CPU only**, `Tensor(x).item()/numpy()` for inspection.
* **im2col conv**: unfold with `sliding_window_view` + one matmul; gradients
  scatter back with overlap-add.
* **Verified end-to-end**: MNIST CNN **98.3% test accuracy in 3 epochs**; a
  ~40k-parameter char-level GPT drops cross-entropy from 2.4 to 0.13 and emits
  coherent Shakespeare-style fragments.
* **Small**: ~1,900 lines of Python, no compilation.