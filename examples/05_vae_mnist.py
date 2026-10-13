"""Train a small convolutional Variational Auto-Encoder on MNIST.

Showcases: shared MNIST loader, Conv2d/ConvTranspose2d (im2col/col2im autograd),
LayerNorm, the reparameterization trick (Tensor.randn_like), KL-divergence loss,
CosineAnnealingLR, and generating a 10x10 latent grid of new digits.

Run:  python3 examples/05_vae_mnist.py --epochs 1      # ~2-3 min on a CPU
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor
from itorch import bce
from mnist_util import load_mnist


class Encoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.SiLU(),      # 14x14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),     # 7x7
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.SiLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.SiLU(),
            nn.Linear(64, 32 * 7 * 7), nn.SiLU(),
            nn.Flatten(),  # -> (B, 1568)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.SiLU(),  # 7 -> 14
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),              # 14 -> 28
        )

    def forward(self, z):
        h = self.net(z).reshape(z.shape[0], 32, 7, 7)
        return self.up(h)          # (B, 1, 28, 28) logits


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.enc(x)
        # reparameterization trick: z = mu + sigma * eps
        eps = tt.randn_like(mu)
        z = mu + logvar.mul(0.5).exp() * eps
        return self.dec(z), mu, logvar

    def sample_grid(self, n=10, seed=0):
        rng = np.random.RandomState(seed)
        z = Tensor(rng.randn(n * n, self.latent_dim).astype(np.float32))
        logits = self.dec(z)
        return (1.0 / (1.0 + np.exp(-logits.data))).reshape(n, n, 28, 28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    tt.manual_seed(0)
    data = load_mnist()
    X = data["train_x"][:, None]            # (60000, 1, 28, 28), pixels in [0,1]
    N = len(X)
    B = args.batch

    model = VAE(latent_dim=args.latent)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sched = tt.schedulers.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"VAE params: {tt.count_parameters(model):,}  (latent dim {args.latent})")

    for epoch in range(args.epochs):
        perm = np.random.permutation(N)
        total = 0.0
        for step in range(N // B):
            idx = perm[step * B:(step + 1) * B]
            x = Tensor(X[idx])
            opt.zero_grad()
            recon, mu, logvar = model(x)
            rec_loss = bce(recon, x)
            # KL with free bits: max(0, kl - 0.5) per latent dim, so the encoder
            # is forced to actually use its capacity (prevents posterior collapse)
            kl_bits = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl = tt.relu(kl_bits - 0.5).sum(1).mean()
            loss = rec_loss + kl
            loss.backward()
            opt.step()
            total += float(loss.data)
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / (N // B):.4f}  "
              f"(recon={float(rec_loss.data):.4f}, kl={float(kl.data):.4f})  lr={opt.lr:.2e}")

    # generate a grid of fresh digits from the latent prior
    grid = model.sample_grid(n=10, seed=1)
    print("sampled a 10x10 latent grid -> examples/grid_vae.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(10, 10, figsize=(8, 8))
        for i in range(10):
            for j in range(10):
                axes[i, j].imshow(grid[i, j], cmap="gray")
                axes[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_vae.png"), dpi=80)
    except ImportError:
        print("(matplotlib not installed; skipping the grid plot)")


if __name__ == "__main__":
    main()