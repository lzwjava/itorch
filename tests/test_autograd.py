"""Numeric gradient checks for every autograd op (run: python3 tests/test_autograd.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import Tensor

np.random.seed(0)


def num_grad(fn, x, eps=5e-3):
    """Central finite-difference gradient of scalar fn w.r.t. input tensor x."""
    g = np.zeros_like(x.data)
    for idx in np.ndindex(x.data.shape):
        orig = x.data[idx]
        x.data[idx] = orig + eps
        fp = float(fn(x).sum().data)
        x.data[idx] = orig - eps
        fm = float(fn(x).sum().data)
        x.data[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
    return g


def check(fn, shape, rtol=2e-2, atol=1e-2, trials=2):
    for _ in range(trials):
        x = Tensor(np.random.randn(*shape) * 2, requires_grad=True)
        fn(x).sum().backward()
        a = x.grad.copy()
        n = num_grad(fn, x)
        ok = np.allclose(a, n, rtol=rtol, atol=atol)
        if not ok:
            raise AssertionError(f"FAIL {getattr(fn, '__name__', fn)}\n analytic={a}\n numeric={n}\n diff={a-n}")
    print(f"  ok  {getattr(fn, '__name__', fn)}  shape={shape}")


def run_op_checks():
    print("== elementwise ops ==")
    check(lambda x: x * 3 - 1, (4,))
    check(lambda x: (x + 2) / 3, (3, 4))
    check(lambda x: (x * x).sum(), (3, 4))
    check(lambda x: x ** 3, (2, 3))
    check(lambda x: x.exp().sum(), (3,))
    check(lambda x: x.abs().sum(), (3, 4))
    check(lambda x: x.clip(-1, 1).sum(), (3, 4))
    check(lambda x: (x.relu() * 2).sum(), (3, 4))
    check(lambda x: x.leaky_relu(0.1).sum(), (3, 4))
    check(lambda x: x.sigmoid().sum(), (3, 4))
    check(lambda x: x.tanh().sum(), (3, 4))
    check(lambda x: (x * x + 0.5).log().sum(), (3, 4))
    check(lambda x: (x * x + 0.5).sqrt().sum(), (3, 4))

    print("== broadcasting ==")
    check(lambda x: (x + x.max(1, keepdims=True)).sum(), (3, 4))
    check(lambda x: (x * Tensor(np.array([1., 2., 3., 4.]))).sum(), (3, 4))

    print("== matmul ==")
    W1 = Tensor(np.random.randn(4, 5) * 2)
    check(lambda x: (x @ W1).sum(), (3, 4))
    v = Tensor(np.random.randn(7) * 2)
    check(lambda x: (x @ v).sum(), (5, 7))
    A = Tensor(np.random.randn(3, 6) * 2)
    check(lambda x: (A @ x).sum(), (6, 4))
    B = Tensor(np.random.randn(3, 5, 2) * 2)
    check(lambda x: (x @ B).sum(), (3, 4, 5))

    print("== reductions ==")
    check(lambda x: x.sum(), (3, 4))
    check(lambda x: x.sum(1), (3, 4))
    check(lambda x: x.sum(0, keepdims=True), (3, 4))
    check(lambda x: x.mean(), (3, 4))
    check(lambda x: x.mean(1), (3, 4))
    check(lambda x: x.max(), (3, 4))
    check(lambda x: x.max(1), (3, 4))
    check(lambda x: x.max(0, keepdims=True), (3, 4))
    check(lambda x: x.min(), (3, 4))
    check(lambda x: x.var(), (3, 4))
    check(lambda x: x.std(), (3, 4))
    check(lambda x: x.softmax(1).sum(1).sum(), (3, 4))

    print("== shape ops ==")
    check(lambda x: x.reshape(2, 6).sum(), (3, 4))
    check(lambda x: x.reshape(12, -1).sum(), (3, 4))
    check(lambda x: x.transpose(0, 1).sum(), (3, 4))
    check(lambda x: x.permute(1, 0).sum(), (3, 4))
    check(lambda x: x.unsqueeze(0).unsqueeze(-1).sum(), (3, 4))
    check(lambda x: x.flatten().sum(), (3, 4))

    print("== indexing / gather ==")
    check(lambda x: x[1].sum(), (3, 4))
    check(lambda x: x[:, 2].sum(), (3, 4))
    check(lambda x: x[[0, 2, 0]].sum(), (3, 4))
    check(lambda x: (x * (x > 0)).sum(), (3, 4))
    check(lambda x: x.gather(0, np.array([[0, 2, 1, 0], [1, 0, 2, 1], [2, 1, 0, 2]])).sum(), (3, 4))
    check(lambda x: x.gather(1, np.array([[0, 1, 2], [2, 0, 1]])).sum(), (2, 3))

    print("== cat / stack / where ==")
    check(lambda x: tt.cat([x, x * 2], 0).sum(), (3, 4))
    check(lambda x: tt.cat([x, x + 1], 1).sum(), (3, 4))
    check(lambda x: tt.stack([x, x * 3], 1).sum(), (3, 4))
    check(lambda x: tt.where(x > 0, x, x * 2).sum(), (3, 4))

    print("== nn modules ==")
    lin = tt.nn.Linear(4, 3)
    check(lambda x: (lin(x) ** 2).sum(), (5, 4))
    emb = tt.nn.Embedding(7, 4)
    check(lambda x: (emb(tt.Tensor(np.array([0, 2, 6]))) * x).sum(), (3, 4))
    conv = tt.nn.Conv2d(2, 3, 3, padding=1)
    check(lambda x: (conv(x) ** 2).sum(), (2, 2, 5, 5))
    conv2 = tt.nn.Conv2d(2, 1, 3, padding=1)
    pool = tt.nn.MaxPool2d(2, 2)
    check(lambda x: (pool(tt.nn.ReLU()(conv2(x)))).sum(), (2, 2, 6, 6))

    print("== batchnorm (training mode) ==")
    # NOTE: sum-loss of BN is degenerate (gradient 0 by shift-invariance),
    # so use a fixed-mask objective to verify gradients nontrivially.
    bn1 = tt.nn.BatchNorm1d(4)
    w1 = Tensor(np.random.randn(8, 4))
    check(lambda x: (bn1(x) * w1).sum(), (8, 4))
    bn2 = tt.nn.BatchNorm2d(3)
    w2 = Tensor(np.random.randn(4, 3, 7, 7))
    check(lambda x: (bn2(x) * w2).sum(), (4, 3, 7, 7), rtol=6e-2, atol=5e-2, trials=1)

    print("== losses ==")
    tgt = Tensor(np.random.randn(3, 4))
    check(lambda x: tt.mse(x, tgt), (3, 4))
    check(lambda x: tt.cross_entropy(x, np.array([0, 2, 1])), (3, 4))


def run_module_checks():
    print("== semantics ==")
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * 2).sum()
    y.backward()
    assert np.allclose(x.grad, [2, 2, 2]), "chain rule add/mul"
    z = (x * x).sum()
    x.grad = None
    z.backward()
    assert np.allclose(x.grad, [2, 4, 6]), "power rule"

    # detach / requires_grad_
    a = Tensor([1.0], requires_grad=True)
    b = a.detach()
    assert not b.requires_grad

    # no_grad
    with tt.no_grad():
        c = a * 2
    assert not c.requires_grad

    # broadcast of value/param grads
    w = Tensor(np.ones((3, 1)), requires_grad=True)
    out = (x.reshape(3, 1) * w).sum()
    out.backward()
    assert np.allclose(w.grad.flatten(), [1, 2, 3]), f"broadcast backward got {w.grad}"

    # params / state_dict / save-load
    model = tt.nn.Sequential(tt.nn.Linear(4, 8), tt.nn.ReLU(), tt.nn.Linear(8, 2))
    assert len(model.parameters()) == 4  # 2 weights + 2 biases
    assert sum(p.data.size for p in model.parameters()) == 4 * 8 + 8 + 8 * 2 + 2
    sd = model.state_dict()
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        tt.save(model, os.path.join(d, "m.npz"))
        m2 = tt.nn.Sequential(tt.nn.Linear(4, 8), tt.nn.ReLU(), tt.nn.Linear(8, 2))
        tt.load(m2, os.path.join(d, "m.npz"))
        assert all(np.allclose(p.data, q.data) for p, q in zip(model.parameters(), m2.parameters()))

    # dropout in eval mode returns identity
    dp = tt.nn.Dropout(0.5).eval()
    inp = Tensor(np.ones((10, 10)))
    assert np.allclose(dp(inp).data, np.ones((10, 10)))

    print("  ok  semantics")


if __name__ == "__main__":
    run_op_checks()
    run_module_checks()
    print("\nALL AUTODIFF CHECKS PASSED ✓")