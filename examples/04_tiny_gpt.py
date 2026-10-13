"""Train a tiny GPT-style character-level language model from scratch and sample text.

A multi-head self-attention transformer built entirely from itorch's autograd:
  token embeddings + positional embeddings -> nn.Transformer (causal stacked blocks)
  -> LM head -> cross entropy.

Run:  python3 examples/04_tiny_gpt.py --steps 200 --seed "to be "
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, cross_entropy

# small self-contained training corpus
CORPUS = (
    "to be or not to be, that is the question. "
    "there are more things in heaven and earth, horatio, than are dreamt of "
    "in your philosophy. the better part of valour is discretion. "
    "brevity is the soul of wit. lord, what fools these mortals be! "
    "all that glitters is not gold. this above all: to thine own self be true. "
    "the course of true love never did run smooth. "
)


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=48, num_heads=4, num_layers=2, ff_dim=96, block_len=96):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_len = block_len
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_len, embed_dim)
        self.tf = nn.Transformer(embed_dim, num_heads, num_layers, ff_dim, dropout=0.0)
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, pos):                       # idx (B, T), pos (B, T)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        return self.head(self.ln(self.tf(h)))          # (B, T, V) logits


def train(args):
    tt.manual_seed(0)
    vocab = sorted(set(CORPUS))
    ch2i, i2ch = {c: i for i, c in enumerate(vocab)}, {i: c for i, c in enumerate(vocab)}
    ids = np.array([ch2i[c] for c in CORPUS], dtype=np.int64)
    print(f"corpus: {len(CORPUS)} chars, vocab={len(vocab)}")

    B, T = args.batch, args.block
    model = TinyGPT(len(vocab), embed_dim=args.embed, num_heads=args.heads,
                    num_layers=args.layers, ff_dim=args.ff, block_len=T)
    opt = optim.AdamW(model.parameters(), lr=3e-3)
    sched = tt.schedulers.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"params: {tt.count_parameters(model):,}  structure: {model}")

    n = len(ids)
    for step in range(1, args.steps + 1):
        # random contiguous chunks with wrap-around for the tail
        start = np.random.randint(0, high=n, size=B)
        x = np.stack([roll(ids, s, T) for s in start])
        y = np.stack([roll(ids, s + 1, T) for s in start])
        pos = np.broadcast_to(np.arange(T), (B, T)).copy()

        opt.zero_grad()
        logits = model(Tensor(x), Tensor(pos))
        loss = cross_entropy(logits.reshape(B * T, -1), y.reshape(-1))
        loss.backward()
        tt.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 20 == 0:
            print(f"step {step:4d}  loss={float(loss.data):.4f}  "
                  f"lr={sched.optimizer.lr:.2e}")

    print("\n--- sampling after training ---")
    print(sample(model, i2ch, args.seed, args.length, args.temperature))


def roll(ids, start, T):
    """Take length-T window of ids starting at `start`, wrapping around the end."""
    out = np.empty(T, dtype=np.int64)
    for i in range(T):
        out[i] = ids[(start + i) % len(ids)]
    return out


@tt.no_grad
def sample(model, i2ch, seed, length, temperature=0.8):
    vocab = {c: i for i, c in i2ch.items()}
    idx = [vocab[c] for c in seed]
    T = model.block_len
    out = list(seed)
    for _ in range(length):
        window = np.array(idx[-T:], dtype=np.int64)
        pos = np.arange(len(window)).reshape(1, -1)
        logits = model(Tensor(window.reshape(1, -1)), Tensor(pos)).data[0, -1]
        probs = np.exp((logits - logits.max()) / temperature)
        probs /= probs.sum()
        nxt = int(np.random.choice(len(vocab), p=probs))
        out.append(i2ch[nxt])
        idx.append(nxt)
    return "".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--embed", type=int, default=48)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=96)
    ap.add_argument("--seed", type=str, default="to be or not to be, ")
    ap.add_argument("--length", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()
    train(args)