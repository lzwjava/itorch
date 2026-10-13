"""Tests for the 'sophisticated' itorch features (run: python3 tests/test_sophisticated.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, cross_entropy, nll_loss

np.random.seed(7)


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


def gradcheck(fn, shape, rtol=2e-2, atol=1e-2, trials=2):
    for _ in range(trials):
        x = Tensor(np.random.randn(*shape) * 1.5, requires_grad=True)
        fn(x).sum().backward()
        ga = x.grad.copy()
        gn = numeric_grad(fn, x)
        assert np.allclose(ga, gn, rtol=rtol, atol=atol), \
            f"FAIL {getattr(fn, '__name__', fn)}: max err {np.max(np.abs(ga - gn))}"
    print(f"  ok  {getattr(fn, '__name__', fn)}  {shape}")


def test_gradients():
    print("== gradient checks ==")
    gradcheck(lambda x: x.gelu().sum(), (3, 4))
    ln_w = Tensor(np.random.randn(6, 4))
    gradcheck(lambda x: (nn.LayerNorm(4)(x) * ln_w).sum(), (6, 4))
    gn_w = Tensor(np.random.randn(2, 4, 5, 5))
    gradcheck(lambda x: (nn.GroupNorm(2, 4)(x) * gn_w).sum(), (2, 4, 5, 5))
    ap_w = Tensor(np.random.randn(2, 3, 4, 4))
    gradcheck(lambda x: (nn.AvgPool2d(2)(x) * ap_w).sum(), (2, 3, 8, 8))

    gru_w = Tensor(np.random.randn(3, 6, 4))
    gru = nn.GRU(5, 4)
    gradcheck(lambda x: (gru(x)[0] * gru_w).sum(), (3, 6, 5), trials=1)
    lstm_w = Tensor(np.random.randn(3, 6, 4))
    lstm = nn.LSTM(5, 4)
    gradcheck(lambda x: (lstm(x)[0] * lstm_w).sum(), (3, 6, 5), trials=1)

    w = Tensor(np.random.randn(3, 5, 8))
    mha = nn.MultiHeadAttention(8, 4)
    gradcheck(lambda x: (mha(x) * w).sum(), (3, 5, 8), trials=1)
    mha_c = nn.MultiHeadAttention(8, 4)
    gradcheck(lambda x: (mha_c(x, nn.causal_mask(5)) * w).sum(), (3, 5, 8), trials=1)

    blk = nn.TransformerBlock(8, 4, 32, dropout=0.0)
    gradcheck(lambda x: (blk(x) * w).sum(), (3, 5, 8), trials=1)

    print("== loss gradients ==")
    tgt = Tensor(np.random.randn(3, 4))
    gradcheck(lambda x: tt.l1(x, tgt), (3, 4))
    gradcheck(lambda x: tt.huber(x, tgt, delta=1.0), (3, 4))
    gradcheck(lambda x: tt.nll_loss(x.log_softmax(1), np.array([0, 2, 1])), (3, 4))


def test_engine():
    print("== engine semantics ==")
    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    y = (x * 2).sum()
    assert y.grad_fn.startswith("sum("), y.grad_fn
    y.backward()
    assert x.grad_fn is None, "leaf has no op"
    assert np.allclose(x.grad, 2)
    try:
        y.backward()
        raise AssertionError("expected RuntimeError for freed graph")
    except RuntimeError:
        pass
    # retained graph: second backward OK
    y2 = (x * 3).sum()
    y2.backward(retain_graph=True)
    y2.backward(retain_graph=True)
    assert np.allclose(x.grad, 3)

    # leaf reuse after a graph was freed
    y3 = (x * 4).sum()
    y3.backward()
    assert np.allclose(x.grad, 4)

    # no_grad as decorator + context
    @tt.no_grad
    def f(t):
        return t * 2
    assert not f(x).requires_grad
    with tt.no_grad():
        assert not (x * 2).requires_grad

    # freeze / unfreeze
    m = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    m.freeze()
    assert all(not p.requires_grad for p in m.parameters())
    m.unfreeze()
    assert all(p.requires_grad for p in m.parameters())
    assert tt.count_parameters(m) == 4 * 6 + 6 + 6 * 2 + 2
    print("  ok  engine semantics")


def test_schedulers():
    print("== schedulers ==")
    m = nn.Linear(3, 3)
    opt = optim.SGD(m.parameters(), lr=1.0)
    s1 = tt.schedulers.StepLR(opt, step_size=3, gamma=0.1)
    lrs = [s1.step() for _ in range(6)]
    assert np.allclose(lrs, [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]), lrs

    opt2 = optim.Adam(m.parameters(), lr=1.0)
    s2 = tt.schedulers.MultiStepLR(opt2, milestones=[2, 4], gamma=0.5)
    lrs = [s2.step() for _ in range(6)]
    assert np.allclose(lrs, [1, 1, 0.5, 0.5, 0.25, 0.25]), lrs

    opt3 = optim.SGD(m.parameters(), lr=1.0)
    s3 = tt.schedulers.CosineAnnealingLR(opt3, T_max=4, eta_min=0.0)
    lrs = [round(s3.step(), 4) for _ in range(5)]
    assert lrs[0] == 1.0 and abs(lrs[1] - 0.8536) < 1e-3 and abs(lrs[4]) < 1e-3, lrs

    opt4 = optim.SGD(m.parameters(), lr=1.0)
    s4 = tt.schedulers.ReduceLROnPlateau(opt4, patience=2, factor=0.5)
    for i in range(5):
        s4.step(10.0 - i * 2 if i < 3 else 10.0)   # improves, improves, then 2 bad epochs
    assert abs(opt4.lr - 0.5) < 1e-6, opt4.lr
    print("  ok  schedulers")


def test_tiny_transformer_training():
    print("== tiny transformer training ==")
    # char-level sequence modeling on a tiny corpus (loss must decrease)
    text = "the quick brown fox jumps over the lazy dog and the dog runs fast"
    vocab = sorted(set(text))
    ch2i = {c: i for i, c in enumerate(vocab)}
    ids = np.array([ch2i[c] for c in text], dtype=np.int64)
    B, T = 8, 16
    n = len(ids)

    model = nn.Transformer(embed_dim=32, num_heads=4, num_layers=2, ff_dim=64, dropout=0.0)
    tok_emb = nn.Embedding(len(vocab), 32)
    pos_emb = nn.Embedding(64, 32)
    head = nn.Linear(32, len(vocab))
    params = list(tok_emb.parameters()) + list(pos_emb.parameters()) + \
             list(model.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=1e-3)

    losses = []
    for step in range(60):
        i = np.random.randint(0, max(1, n - T - 1), size=B)
        x = padd(ids, i, T, len(vocab) - 1)
        y = padd(ids, i + 1, T, 0)
        opt.zero_grad()
        h = tok_emb(x) + pos_emb(np.broadcast_to(np.arange(T), (B, T)))
        logits = head(model(h))
        loss = cross_entropy(logits.reshape(B * T, -1), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    assert losses[-1] < losses[0] * 0.9, f"loss not decreasing: {losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"  ok  tiny transformer (loss {losses[0]:.3f} -> {losses[-1]:.3f})")


def padd(ids, start, T, pad):
    out = np.full((len(start), T), pad, dtype=np.int64)
    for b, s in enumerate(start):
        seg = ids[s:s + T]
        out[b, :len(seg)] = seg
    return out


if __name__ == "__main__":
    test_engine()
    test_gradients()
    test_schedulers()
    test_tiny_transformer_training()
    print("\nALL SOPHISTICATED CHECKS PASSED ✓")