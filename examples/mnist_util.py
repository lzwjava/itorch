"""Shared MNIST loader for itorch examples (auto-download + idx parsing)."""
import os, gzip, urllib.request

import numpy as np

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist")
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


def _fetch(name):
    path = os.path.join(_CACHE, name)
    if not os.path.exists(path):
        os.makedirs(_CACHE, exist_ok=True)
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


def _load_idx(gz_name):
    with gzip.open(_fetch(gz_name), "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    magic = int(np.frombuffer(raw[:4].tobytes(), dtype=">i4")[0])
    ndim = magic % 256
    dims = [int(v) for v in np.frombuffer(raw[4:4 + 4 * ndim].tobytes(), dtype=">i4")]
    data = raw[4 + 4 * ndim:]
    if ndim == 3:  # images: N x rows x cols
        return data.reshape(*dims).astype(np.float32) / 255.0
    return data.reshape(*dims).astype(np.int64)


def load_mnist():
    """Returns {'train_x': (60000,28,28), 'train_y': (60000,), test_x, test_y}."""
    return {key: _load_idx(name) for name, key in FILES.items()}