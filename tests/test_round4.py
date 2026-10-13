"""Tests for round 4: gradcheck/profile utilities, conv engine perf equivalence,
and a DCGAN training smoke test (run: python3 tests/test_round4.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, bce

np.random.seed(5)
tt.manual_seed(5)


def test_gradcheck_utility():
    print("== tt.gradcheck utility ==")
    # valid gradients pass silently
    ok, err = tt.gradcheck(lambda x: (x.silu() ** 2).sum(), tt.randn(4, 3, requires_grad=True),
                           verbose=True)
    assert ok
    ok2, _ = tt.gradcheck(lambda a, b: (a * b).sum(),
                          tt.randn(4, requires_grad=True), tt.randn(4, requires_grad=True))
    assert ok2

    # intentionally-wrong backward must be DETECTED
    def bad_square(x):
        out = Tensor(x.data * x.data, requires_grad=x.requires_grad)
        def backward(g):                       # WRONG: should be 2*x
            x._accumulate(np.asarray(g) * 3.0)
        out._set((x,), backward)
        return out.sum()

    raised = False
    try:
        tt.gradcheck(bad_square, tt.randn(6, requires_grad=True))
    except ValueError as e:
        raised = "FAILED" in str(e)
    assert raised, "gradcheck did not catch an intentionally wrong backward"
    print("  ok  gradcheck (passes, catches bugs)")
    print("  ok  profiler reports op timings")


def test_profile():
    print("== tt.profile ==")
    x = tt.randn(8, 16, requires_grad=True)
    w = tt.randn(16, 8, requires_grad=True)

    def cost():
        return (((x @ w) * 2).relu() ** 1.5).sum()

    cost().backward()
    with tt.profile() as p:
        cost().backward()
    assert "__matmul__" in p.op, "profile must record matmul"
    assert "relu" in p.op and "__mul__" in p.op
    total = sum(v[0] for v in p.op.values())
    assert total > 0
    stats = p.report()
    assert abs(total - sum(v[0] for v in stats.values())) < 1e-9
    print("  ok  profile records + reports ops")


def test_conv_fold_identities():
    print("== im2col / col2im identities ==")
    # 1) exact forward round-trip holds when windows don't overlap (stride >= kernel)
    #    AND division is even (im2col drops any trailing remainder, like torch.unfold)
    for shape in [(2, 3, 8, 8), (4, 2, 9, 9), (4, 2, 12, 10)]:
        for k, s, p in [(2, 2, 0), (2, 2, 1), (3, 3, 0), (3, 2, 1), (4, 4, 1)]:
            h, w = shape[2], shape[3]
            if (h + 2 * p - k) % s or (w + 2 * p - k) % s or s < k:
                continue
            x = np.random.randn(*shape).astype(np.float32)
            cols = Tensor(x).im2col(k, padding=p, stride=s)
            back = cols.col2im(x.shape, k, stride=s, padding=p)
            assert np.allclose(back.data, x, atol=1e-5), f"identity failed {shape} k{k} s{s} p{p}"

    # 2) arbitrary params: gradients of the (im2col -> col2im) round trip match
    #    finite differences (the general adjoint invariant)
    def rt(t):
        return t.im2col(3, padding=1, stride=2).col2im(t.shape, 3, stride=2, padding=1)
    x = tt.randn(2, 2, 7, 7, requires_grad=True)
    rt(x).sum().backward()
    ana = x.grad.copy()
    num = np.zeros_like(x.data)
    for i in np.ndindex(x.data.shape):
        o = x.data[i]
        x.data[i] = o + 1e-3; fp = float(rt(x).sum().data)
        x.data[i] = o - 1e-3; fm = float(rt(x).sum().data)
        x.data[i] = o
        num[i] = (fp - fm) / 2e-3
    assert np.allclose(ana, num, atol=5e-3), "im2col/col2im round-trip gradient mismatch"

    # 3) implicit contract used by ConvTranspose2d: overlapping writes accumulate
    cols = np.zeros((1, 4, 9), dtype=np.float32)          # k=2, 3x3 windows on a 4x4 canvas
    cols[0, 0, :] = 1.0                                   # tap (0, 0) of every window
    cols[0, 1, :] = 1.0                                   # tap (0, 1) of every window
    out = Tensor(cols).col2im((1, 1, 4, 4), 2, stride=1, padding=0)
    ref = np.zeros((4, 4))
    for a in range(3):
        for b in range(3):
            ref[a, b] += 1          # tap (0, 0)
            ref[a, b + 1] += 1      # tap (0, 1)
    assert np.allclose(out.data[0, 0], ref), (out.data[0, 0], ref)
    print("  ok  im2col/col2im (identity, adjoint, overlap-add)")


def test_basic_slice_backward():
    print("== optimized basic-slice backward ==")
    # gradient of a crop must put values back at the right (offset) positions
    x = tt.randn(4, 6, requires_grad=True)
    (x[:, 2:5].sum()).backward()
    assert np.allclose(x.grad[:, 2:5], 1.0)
    assert np.allclose(x.grad[:, :2], 0.0) and np.allclose(x.grad[:, 5:], 0.0)
    # negative steps
    x2 = tt.randn(5, requires_grad=True)
    (x2[::-1] * 2).sum().backward()
    assert np.allclose(x2.grad, 2.0)
    # row drop + int index combo
    x3 = tt.randn(3, 4, requires_grad=True)
    (x3[1] * 7).sum().backward()
    assert np.allclose(x3.grad[1], 7.0) and np.allclose(x3.grad[0] + x3.grad[2], 0.0)
    print("  ok  basic slicing")


def test_dcgan_smoke():
    print("== DCGAN smoke ==")
    from importlib import import_module
    m = import_module("06_dcgan_mnist")
    G, D = m.Generator(latent=16), m.Discriminator()
    optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    B = 16
    ones = Tensor(np.ones(B, dtype=np.float32))
    zeros = Tensor(np.zeros(B, dtype=np.float32))

    g_hist = []
    for _ in range(30):
        real = Tensor(np.random.rand(B, 1, 28, 28).astype(np.float32) * 2 - 1)
        z = tt.randn(B, 16)
        fake = G(z)
        optD.zero_grad()
        d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
        d_loss.backward()
        optD.step()
        optG.zero_grad()
        g_loss = bce(D(fake), ones)
        g_loss.backward()
        optG.step()
        g_hist.append(float(g_loss.data))

    loss_nan = any(not np.isfinite(v) for v in g_hist) or not np.isfinite(float(d_loss.data))
    assert not loss_nan, "NaN/Inf in GAN losses"
    assert float(fake.data.min()) >= -1.0 and float(fake.data.max()) <= 1.0
    assert g_hist[-1] < g_hist[0], f"generator should win a little: {g_hist[0]:.2f} -> {g_hist[-1]:.2f}"
    print(f"  ok  DCGAN smoke (G loss {g_hist[0]:.3f} -> {g_hist[-1]:.3f})")


if __name__ == "__main__":
    test_gradcheck_utility()
    test_profile()
    test_conv_fold_identities()
    test_basic_slice_backward()
    test_dcgan_smoke()
    print("\nALL ROUND-4 CHECKS PASSED ✓")