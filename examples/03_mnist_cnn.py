"""Train a tiny CNN on real MNIST (downloaded automatically, ~11 MB).

Run:  python3 examples/03_mnist_cnn.py          # quick: 1 epoch, ~2-3 min on CPU
      python3 examples/03_mnist_cnn.py --epochs 3
"""
import sys, os, gzip, urllib.request, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import itorch as tt
from itorch import nn, optim, Tensor, cross_entropy, accuracy

BASE = "http://yann.lecun.com/exdb/mnist/"
MIRRORS = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train-images-idx3-ubyte.gz": "train_x",
    "train-labels-idx1-ubyte.gz": "train_y",
    "t10k-images-idx3-ubyte.gz": "test_x",
    "t10k-labels-idx1-ubyte.gz": "test_y",
}
CACHE = os.path.join(os.path.dirname(__file__), "mnist")


def fetch(name):
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        for base in MIRRORS:
            try:
                print(f"downloading {name} from {base} ...")
                urllib.request.urlretrieve(base + name, path)
                break
            except Exception as e:
                print(f"  failed ({e}); trying next mirror")
        else:
            raise RuntimeError(f"could not download {name} from any mirror")
    return path


def load_idx(gz_name):
    with gzip.open(fetch(gz_name), "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    magic = int(np.frombuffer(raw[:4].tobytes(), dtype=">i4")[0])
    ndim = magic % 256
    dims = [int(v) for v in np.frombuffer(raw[4:4 + 4 * ndim].tobytes(), dtype=">i4")]
    data = raw[4 + 4 * ndim:]
    if ndim == 3:  # images: N x rows x cols
        return data.reshape(*dims).astype(np.float32) / 255.0
    return data.reshape(*dims).astype(np.int64)


def load_mnist():
    return {key: load_idx(name) for name, key in FILES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    tt.manual_seed(0)
    data = load_mnist()
    train_x, train_y = data["train_x"][:, None], data["train_y"]  # (N,1,28,28), (N,)
    test_x, test_y = data["test_x"][:, None], data["test_y"]

    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 10),
    )
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    print(model)

    steps_per_epoch = len(train_x) // args.batch
    for epoch in range(args.epochs):
        # shuffle
        perm = np.random.permutation(len(train_x))
        total, correct = 0.0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch:(step + 1) * args.batch]
            opt.zero_grad()
            loss = cross_entropy(model(Tensor(train_x[idx])), train_y[idx])
            loss.backward()
            opt.step()
            total += float(loss.data)
            correct += (model(Tensor(train_x[idx])).argmax(-1) == train_y[idx]).sum()
        acc = correct / (steps_per_epoch * args.batch)
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total / steps_per_epoch:.4f}  train acc={acc:.4f}")

    # eval on test set
    model.eval()
    preds = np.concatenate([model(Tensor(test_x[i:i + 512])).argmax(-1) for i in range(0, len(test_x), 512)])
    print(f"test accuracy: {np.mean(preds == test_y):.4f}")


if __name__ == "__main__":
    main()