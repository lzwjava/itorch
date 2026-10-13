"""Tests for round 3: col2im/ConvTranspose2d, new activations/optimizers/losses,
init helpers, autograd.grad, and a VAE smoke test (run: python3 tests/test_round3.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, bce

np.random.seed(3)


def numeric_grad(fn, x, eps=5e-3):
    g = np.zeros_like(x.data)
    for i in np.ndindex(x.data.shape):
        o = x.data[i]
        x.data[i] = o + eps
        fp = float(fn(x).sum().data)
        x.data[i] = o - eps
        fm = float(fn(x).sum().data)
        x.data[i] = o
        g[i] = (fp - fm) / (2 * eps)
    return g


def gradcheck(fn, shape, rtol=3e-2, atol=2e-2, trials=2):
    for _ in range(trials):
        x = Tensor(np.random.randn(*shape) * 1.5, requires_grad=True)
        fn(x).sum().backward()
        ga = x.grad.copy()
        gn = numeric_grad(fn, x)
        assert np.allclose(ga, gn, rtol=rtol, atol=atol), \
            f"FAIL max err {np.max(np.abs(ga - gn))}"
    print(f"  ok  {getattr(fn, '__name__', fn)}  {shape}")


def test_gradients():
    print("== gradient checks ==")
    gradcheck(lambda x: x.sin().sum(), (3, 4))
    gradcheck(lambda x: x.cos().sum(), (3, 4))
    gradcheck(lambda x: x.silu().sum(), (3, 4))
    gradcheck(lambda x: x.softplus().sum(), (3, 4))
    gradcheck(lambda x: x.softplus(2.0).sum(), (3, 4))

    # col2im folded without padding is im2col's left inverse
    x = Tensor(np.random.randn(2, 3, 8, 8) * 1.5, requires_grad=True)
    cols = x.im2col(3, padding=1, stride=2)                 # (2, 27, 16)
    back = cols.col2im(x.shape, 3, stride=2, padding=1)
    assert back.shape == x.shape
    back.sum().backward()
    gn = numeric_grad(lambda t: t.im2col(3, padding=1, stride=2).col2im(t.shape, 3, stride=2, padding=1), x)
    assert np.allclose(x.grad, gn, rtol=3e-2, atol=3e-2)
    print("  ok  im2col/col2im round-trip gradients")

    cT = nn.ConvTranspose2d(3, 2, 3, stride=2, padding=1)
    w = Tensor(np.random.randn(2, 2, 7, 7))
    gradcheck(lambda t: (cT(t) * w).sum(), (2, 3, 4, 4), trials=1)
    cT2 = nn.ConvTranspose2d(2, 1, 4, stride=2, padding=1)
    w2 = Tensor(np.random.randn(2, 1, 10, 10))
    gradcheck(lambda t: (cT2(t) * w2).sum(), (2, 2, 5, 5), trials=1)


def test_col2im_matches_numpy():
    print("== conv vs conv_transpose equivalence ==")
    # For stride 1, conv_transpose is the gradient-transpose of conv:
    #  sum(y * conv(x, w)) == sum(convT(y, w) * x)  for any x, y, w
    x = np.random.randn(2, 2, 5, 5).astype(np.float32)
    y = np.random.randn(2, 3, 5, 5).astype(np.float32)
    w = np.random.randn(3, 2, 3, 3).astype(np.float32) * 0.5
    conv = nn.Conv2d(2, 3, 3, padding=1)
    conv.weight.data = w.copy()
    convt = nn.ConvTranspose2d(3, 2, 3, stride=1, padding=1)
    convt.weight.data = w.copy()
    a = float((Tensor(y) * conv(Tensor(x))).sum().data)
    b = float((convt(Tensor(y)) * Tensor(x)).sum().data)
    assert abs(a - b) < 1e-4, f"adjoint mismatch: {a} vs {b}"
    print(f"  ok  adjoint identity  ({a:.4f} == {b:.4f})")


def test_optimizers():
    print("== optimizers ==")
    # 1-D quadratic (w - 3)^2: every optimizer must converge reliably
    for name, cls, lr in [("SGD", optim.SGD, 0.1), ("Adam", optim.Adam, 0.1),
                          ("AdamW", optim.AdamW, 0.1), ("Adagrad", optim.Adagrad, 0.5),
                          ("RMSprop", optim.RMSprop, 0.05), ("Adamax", optim.Adamax, 0.05),
                          ("Lion", optim.Lion, 0.05)]:
        w = Tensor(np.array([0.0]), requires_grad=True)
        opt = cls([w], lr=lr)
        for _ in range(200):
            opt.zero_grad()
            (w - 3).pow(2).backward()
            opt.step()
        assert abs(float(w.item()) - 3.0) < 0.2, f"{name} w={w.item():.3f}"
        print(f"  ok  {name:<9s} w={float(w.item()):.3f}")

    # regression still decreases for every optimizer
    x = Tensor(np.random.randn(64, 4))
    y = Tensor(x.data @ np.array([3., -2., 1., -0.5], dtype=np.float32) - 1.0).reshape(-1, 1)
    for name, cls, lr in [("Adagrad", optim.Adagrad, 0.3), ("RMSprop", optim.RMSprop, 0.01),
                          ("Adamax", optim.Adamax, 0.002), ("Lion", optim.Lion, 1e-4)]:
        w = Tensor(np.zeros((4, 1)), requires_grad=True)
        b = Tensor(np.zeros(1), requires_grad=True)
        opt = cls([w, b], lr=lr)
        first = None
        for _ in range(50):
            opt.zero_grad()
            loss = tt.mse(x @ w + b, y)
            loss.backward()
            opt.step()
            first = float(loss.data) if first is None else first
        assert float(loss.data) < first, f"{name}: {first:.4f} -> {float(loss.data):.4f}"
    print("  ok  all optimizers decrease a regression loss")


def test_losses_and_init():
    print("== class-based losses match functional ==")
    i, t = Tensor(np.random.randn(6, 3) * 2), Tensor(np.random.randn(6, 3))
    targets = np.array([0, 2, 1, 2, 0, 1])
    assert abs(float(nn.MSELoss()(i, t).item()) - float(tt.mse(i, t).item())) < 1e-6
    assert abs(float(nn.L1Loss()(i, t).item()) - float(tt.l1(i, t).item())) < 1e-6
    assert abs(float(nn.HuberLoss()(i, t).item()) - float(tt.huber(i, t).item())) < 1e-6
    assert abs(float(nn.CrossEntropyLoss()(i, targets).item()) - float(tt.cross_entropy(i, targets).item())) < 1e-5
    lp = i.softmax(1).log()
    assert abs(float(nn.NLLLoss()(lp, targets).item()) - float(tt.nll_loss(lp, targets).item())) < 1e-6
    tgt_bin = (i > 0).numpy().astype(np.float32)
    assert abs(float(nn.BCEWithLogitsLoss()(i, Tensor(tgt_bin)).item())
               - float(tt.bce(i, Tensor(tgt_bin)).item())) < 1e-6
    print("  ok  loss classes")

    print("== nn.init ==")
    lin = nn.Linear(20, 10)
    nn.init.zeros_(lin.weight)
    assert np.allclose(lin.weight.data, 0)
    nn.init.kaiming_normal_(lin.weight, a=0.0)
    target_std = np.sqrt(2.0 / 20)
    assert abs(float(lin.weight.data.std()) - target_std) < 0.05, \
        f"kaiming std {float(lin.weight.data.std()):.3f} vs {target_std:.3f}"
    nn.init.xavier_uniform_(lin.weight)
    bound = np.sqrt(6.0 / (10 + 20)) * 0.997  # uniform bound * (n-1)/(n+1) correction ~1
    assert np.max(np.abs(lin.weight.data)) <= bound * 1.05
    nn.init.normal_(lin.bias, mean=1.0, std=0.1)
    assert abs(float(lin.bias.data.mean()) - 1.0) < 0.1
    print("  ok  init helpers")


def test_autograd_grad():
    print("== autograd.grad ==")
    x = Tensor(np.random.randn(5), requires_grad=True)
    y = Tensor(np.random.randn(5), requires_grad=True)
    out = (x * y).sum() + (x + y).sum()
    gx, gy = tt.grad(out, [x, y])
    assert np.allclose(gx, y.data + 1.0) and np.allclose(gy, x.data + 1.0)
    gx2 = tt.grad((x * x).sum(), x)
    assert np.allclose(gx2, 2 * x.data)
    print("  ok  autograd.grad")


def test_vae_smoke():
    print("== VAE smoke (ConvTranspose2d reparameterization path) ==")
    tt.manual_seed(0)

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.SiLU(),   # 14
                                     nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU())  # 7
            self.mu = nn.Linear(16 * 7 * 7, 4)
            self.logvar = nn.Linear(16 * 7 * 7, 4)

        def forward(self, x):
            h = self.net(x)
            return self.mu(h.flatten(1)), self.logvar(h.flatten(1))

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 16 * 7 * 7), nn.SiLU())
            self.up = nn.Sequential(nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.SiLU(),  # 14
                                    nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1))             # 28

        def forward(self, z):
            return self.up(self.net(z).reshape(z.shape[0], 16, 7, 7))

    enc, dec = Enc(), Dec()
    opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    x = Tensor(np.random.rand(32, 1, 28, 28).astype(np.float32))
    losses = []
    for _ in range(20):
        opt.zero_grad()
        mu, lv = enc(x)
        z = mu + lv.mul(0.5).exp() * tt.randn_like(mu)
        recon = dec(z)
        kl_bits = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
        loss = bce(recon, x) + tt.relu(kl_bits - 0.5).sum(1).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0], f"VAE loss not decreasing: {losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"  ok  VAE smoke (loss {losses[0]:.3f} -> {losses[-1]:.3f})")


if __name__ == "__main__":
    test_gradients()
    test_col2im_matches_numpy()
    test_optimizers()
    test_losses_and_init()
    test_autograd_grad()
    test_vae_smoke()
    print("\nALL ROUND-3 CHECKS PASSED ✓")