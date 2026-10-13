"""End-to-end training smoke tests (run: python3 tests/test_train.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, mse, cross_entropy, accuracy

np.random.seed(42)


def test_linear_regression():
    # y = 3x + 2  (single parameter vector) -> gradient descent on MSE
    x = Tensor(np.random.randn(64, 5))
    w_true = Tensor(np.array([1.5, -2.0, 3.0, -0.5, 2.5]))
    y = (x * w_true.reshape(1, 5)).sum(1, keepdims=True) + 4.0

    w = Tensor(np.zeros((5, 1)), requires_grad=True)
    b = Tensor(np.zeros(1), requires_grad=True)
    opt = optim.SGD([w, b], lr=0.1)

    for _ in range(400):
        opt.zero_grad()
        pred = x @ w + b
        loss = mse(pred, y)
        loss.backward()
        opt.step()

    final = float(mse(x @ w + b, y).item())
    assert final < 1e-3, f"linear regression loss too high: {final}"
    assert np.allclose(w.data.flatten(), w_true.data, atol=1e-2)
    assert abs(float(b.item()) - 4.0) < 1e-2
    print(f"  ok  linear regression  (loss={final:.2e})")


def test_xor_mlp():
    # classic non-linear XOR with a hidden layer, Adam
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    Y = Tensor(np.array([0, 1, 1, 0], dtype=np.float32))

    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(1200):
        opt.zero_grad()
        loss = mse(model(X).flatten(), Y)
        loss.backward()
        opt.step()

    pred = np.round(model(X).flatten().data)
    acc = np.mean(pred == Y.data)
    assert acc == 1.0, f"xor accuracy {acc}"
    print(f"  ok  xor MLP (acc={acc:.0%})")


def test_cnn_classifier_synthetic():
    # tiny CNN on synthetic 8x8 images, BatchNorm + Dropout, Adam
    N = 32
    X = np.random.rand(N, 1, 8, 8).astype(np.float32)
    Y = (X.mean(axis=(1, 2, 3)) > 0.5).astype(np.int64)

    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # 8x8 -> 4x4
        nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # -> 2x2
        nn.Flatten(), nn.Linear(8 * 2 * 2, 2),
    )
    opt = optim.AdamW(model.parameters(), lr=1e-2)

    hist = []
    for _ in range(100):
        opt.zero_grad()
        loss = cross_entropy(model(Tensor(X)), Y)
        loss.backward()
        opt.step()
        hist.append(float(loss.data))

    acc = accuracy(model(Tensor(X)), Y)
    assert acc > 0.9, f"synthetic CNN acc {acc:.3f}"
    print(f"  ok  cnn classifier (loss {hist[0]:.3f} -> {hist[-1]:.3f}, acc={acc:.0%})")


def test_eval_mode_batchnorm():
    model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4))
    bn = model.modules[1]
    assert set(bn.__dict__) >= {"running_mean", "running_var"}, "batch norm buffers present"
    x = Tensor(np.random.randn(8, 8))
    model(x)                       # training mode: updates running stats
    rm = bn.running_mean.copy()
    assert not np.allclose(rm, 0), "running stats got updated in training"
    model.eval()
    model(x)
    assert np.allclose(bn.running_mean, rm), "running stats must not change in eval"
    # eval forward must be differentiable too
    x2 = Tensor(np.random.randn(8, 8), requires_grad=True)
    model(x2).sum().backward()
    assert x2.grad is not None and np.max(np.abs(x2.grad)) > 0
    print("  ok  batchnorm train/eval")


def test_dataloader_and_clip():
    # TensorDataset + DataLoader produce correctly stacked batches
    full = tt.TensorDataset(np.random.randn(10, 4), np.random.randint(0, 3, size=10))
    dl = tt.DataLoader(full, batch_size=3, shuffle=False)
    batches = list(dl)
    assert len(batches) == 4
    x0, y0 = batches[0]
    assert x0.shape == (3, 4) and y0.shape == (3,)
    assert np.allclose(np.concatenate([b[0].data for b in batches]), full.tensors[0])

    # clip_grad_norm_ actually scales huge grads
    p = Tensor(np.zeros(2), requires_grad=True)
    p.grad = np.array([100.0, -50.0])
    total = tt.clip_grad_norm_([p], 1.0)
    assert abs(total - np.sqrt(100 ** 2 + 50 ** 2)) < 1e-3
    assert np.allclose(p.grad, [100.0 / total, -50.0 / total])
    print("  ok  dataloader + clip_grad_norm")


if __name__ == "__main__":
    test_linear_regression()
    test_xor_mlp()
    test_cnn_classifier_synthetic()
    test_eval_mode_batchnorm()
    test_dataloader_and_clip()
    print("\nALL TRAINING CHECKS PASSED ✓")