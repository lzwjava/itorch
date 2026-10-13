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
| `Tensor` | reverse-mode autograd over a numpy array: `+ - * / ** @`, matmul (batched & 1-D), `sum/mean/var/std/max/min`, `exp/log/sqrt/abs`, `relu/leaky_relu/gelu/silu/softplus/sigmoid/tanh/softmax/log_softmax`, `sin/cos`, `reshape/transpose/permute/unsqueeze/squeeze/flatten`, slicing `x[i]` (basic-slice backward does a fast zero-pad), `gather`, `cat/stack/where`, `im2col`/`col2im` (conv engines), `zeros_like/ones_like/randn_like/rand_like` |
| `backward()` | topological pass; frees the graph afterwards. `retain_graph=True` keeps it for a second call; a repeat `backward()` on a freed graph raises `RuntimeError`. `tt.grad(outputs, inputs)` exposes grads directly. |
| `grad_fn` | `y.grad_fn` → `"matmul((3,4),(4,5)) -> (3,5)"` for debugging |
| `no_grad` | works as `with no_grad():` **and** as a decorator `@no_grad` |
| `tt.gradcheck` | validate custom layers against finite differences: `tt.gradcheck(fn, x, y)` |
| `tt.profile` | per-op backward timings: `with tt.profile() as p: loss.backward(); p.report()` |

### nn
`Module` (params, `train/eval`, `freeze/unfreeze`, `state_dict`), `Sequential`,
`Linear`, `Embedding`, `Conv2d`, `ConvTranspose2d` (im2col/col2im), `MaxPool2d`,
`AvgPool2d`, `BatchNorm1d/2d`, `LayerNorm`, `GroupNorm`, `Dropout`, `Flatten`,
`ReLU/GELU/SiLU/Sigmoid/Tanh/Softplus/Softmax/LogSoftmax/LeakyReLU`, **recurrent**:
`LSTM`, `GRU` (batch_first, return `(outputs, state)`), **attention**:
`MultiHeadAttention`, `TransformerBlock` (pre-norm + residuals),
`Transformer` (causal stacked blocks + final `LayerNorm`), `causal_mask(T)`.
Class-based losses (`nn.MSELoss`, `nn.L1Loss`, `nn.HuberLoss`, `nn.BCELoss`,
`nn.BCEWithLogitsLoss`, `nn.CrossEntropyLoss`, `nn.NLLLoss`) and `nn.init`
(`kaiming_normal_/uniform_`, `xavier_normal_/uniform_`, `normal_/uniform_`, `zeros_/ones_`).

### optim & schedulers
`SGD` (momentum/nesterov/weight-decay), `Adam`, `AdamW`, `Adagrad`, `RMSprop`,
`Adamax`, `Lion`, and `itorch.schedulers`: `StepLR`, `MultiStepLR`,
`CosineAnnealingLR`, `ReduceLROnPlateau`.

## run it

```bash
python3 tests/test_autograd.py       # numeric-gradient check of every core op
python3 tests/test_sophisticated.py  # gradchecks: gelu, norms, pools, rnns, attention, transformer, losses
python3 tests/test_round3.py         # gradchecks: convT/col2im, new activations, optimizers, losses, VAE
python3 tests/test_round4.py         # gradcheck/profile utils, conv identities, slicing, DCGAN smoke
python3 tests/test_train.py          # end-to-end training smoke tests
python3 examples/01_xor.py
python3 examples/02_linear_regression.py
python3 examples/03_mnist_cnn.py --epochs 3    # downloads MNIST (~11 MB), ~2 min
python3 examples/04_tiny_gpt.py --steps 400    # char-level transformer, samples Shakespeare-ish text
python3 examples/05_vae_mnist.py --epochs 2    # convolutional VAE, ~1.5 min, writes grid_vae.png
python3 examples/06_dcgan_mnist.py             # DCGAN: adversarial min-max, ~2 min, writes grid_dcgan.png
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
  scatter back with **per-tap strided adds (17× faster than the original
  `np.add.at` version)**.
* **Verified end-to-end**: MNIST CNN **98.3% test accuracy in 3 epochs**; a
  ~40k-parameter char-level GPT drops cross-entropy from 2.4 to 0.13 and emits
  coherent Shakespeare-style fragments; a convolutional **VAE** (free-bits KL)
  reconstructs digits; a **DCGAN** trains with healthy adversarial dynamics and
  generates digits (**256/256 healthy samples** after 1000 steps).
* **Small**: ~3,600 lines of Python, no compilation.