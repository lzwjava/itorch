"""Train a Deep Convolutional GAN (DCGAN) on MNIST — adversarial (min-max) training.

Uses Conv2d / ConvTranspose2d / BatchNorm / LeakyReLU, two Adam optimizers
(lr=2e-4, betas=(0.5, 0.999)), and BCE-on-logits losses. Run:

    python3 examples/06_dcgan_mnist.py --steps 600     # ~1-2 min on a CPU
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, bce
from mnist_util import load_mnist


class Generator(nn.Module):
    def __init__(self, latent=64, ngf=32):
        super().__init__()
        self.latent = latent
        self.fc = nn.Sequential(nn.Linear(latent, ngf * 8 * 7 * 7), nn.BatchNorm1d(ngf * 8 * 7 * 7), nn.ReLU())
        # 7x7 -> 14x14 -> 28x28
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 4, 1, 4, stride=2, padding=1),
            nn.Tanh(),                                    # pixels in [-1, 1]
        )

    def forward(self, z):
        h = self.fc(z).reshape(z.shape[0], -1, 7, 7)
        return self.deconv(h)


class Discriminator(nn.Module):
    def __init__(self, ndf=32):
        super().__init__()
        # 28x28 -> 14x14 -> 7x7 -> logistic (no batch norm, as in the DCGAN paper)
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(ndf * 2 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)                     # (B,) logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--latent", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grid", type=int, default=5)         # grid size when sampling
    args = ap.parse_args()

    tt.manual_seed(0)
    X = (load_mnist()["train_x"][:, None] * 2.0 - 1.0)     # [0,1] -> [-1,1]
    N, B = len(X), args.batch

    G = Generator(args.latent)
    D = Discriminator()
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"G params: {tt.count_parameters(G):,}   D params: {tt.count_parameters(D):,}")

    real = np.ones(B, dtype=np.float32)                   # discriminator targets

    for step in range(1, args.steps + 1):
        # ---- train discriminator on real + fake ----
        idx = np.random.randint(0, N, B)
        real_x = Tensor(X[idx])
        z = tt.randn(B, args.latent)
        fake = G(z)

        optD.zero_grad()
        d_real = bce(D(real_x), Tensor(real))
        d_fake = bce(D(fake.detach()), Tensor(np.zeros(B, dtype=np.float32)))
        d_loss = d_real + d_fake
        d_loss.backward()
        optD.step()

        # ---- train generator (separate graph so no graph-reuse conflicts) ----
        optG.zero_grad()
        g_loss = bce(D(fake), Tensor(real))
        g_loss.backward()
        optG.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}  D={float(d_loss.data):.3f} "
                  f"(real {float(d_real.data):.3f}, fake {float(d_fake.data):.3f})  "
                  f"G={float(g_loss.data):.3f}")

    # sample a grid of fresh digits
    z = tt.randn(args.grid * args.grid, args.latent)
    grid = (G(z).data + 1.0) / 2.0                          # [-1,1] -> [0,1]
    grid = grid.reshape(args.grid, args.grid, 28, 28)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_dcgan.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(args.grid, args.grid, figsize=(args.grid * 0.9, args.grid * 0.9))
        for i in range(args.grid):
            for j in range(args.grid):
                axes[i, j].imshow(grid[i, j], cmap="gray")
                axes[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(out, dpi=80)
    except ImportError:
        print("(matplotlib not installed; skipping the grid plot)")
    print(f"sampled a {args.grid}x{args.grid} grid of generated digits -> {out}")


if __name__ == "__main__":
    main()