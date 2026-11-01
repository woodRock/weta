"""The mnist_cnn example in PyTorch, as a speed baseline on the same GPU.
Same model, batch size, optimiser and data pipeline (uint8 on the host,
normalised per batch, then copied to the device).

    python3 scripts/torch_mnist_cnn.py data 5
"""
import gzip
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

root, epochs = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")


def idx(name, off):
    return np.frombuffer(gzip.open(f"{root}/MNIST/raw/{name}").read(), dtype=np.uint8, offset=off)


X = torch.from_numpy(idx("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).copy())
Y = torch.from_numpy(idx("train-labels-idx1-ubyte.gz", 8).astype(np.int64))
Xt = torch.from_numpy(idx("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28).copy())
Yt = torch.from_numpy(idx("t10k-labels-idx1-ubyte.gz", 8).astype(np.int64))
norm = lambda x: (x.float() / 255 - 0.1307) / 0.3081

torch.manual_seed(42)
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Dropout(0.25), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, 10),
).to(dev)
opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, 1e-5)
for epoch in range(epochs):
    start = time.perf_counter()
    model.train()
    perm = torch.randperm(len(Y))
    for i in range(0, len(Y), 128):
        b = perm[i:i + 128]
        x, y = norm(X[b]).to(dev), Y[b].to(dev)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss.item()  # weta's example syncs here too
    sched.step()
    model.eval()
    with torch.no_grad():
        correct = sum((model(norm(Xt[i:i + 1000]).to(dev)).argmax(1).cpu() == Yt[i:i + 1000]).sum().item() for i in range(0, 10000, 1000))
    print(f"epoch {epoch}: test accuracy {correct / 100:.2f}% ({time.perf_counter() - start:.1f}s)")
