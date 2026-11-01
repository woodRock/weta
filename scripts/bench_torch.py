"""The same benchmarks as crates/weta/benches/ops.rs, in PyTorch on the CPU."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def bench(name, f, iters=50):
    for _ in range(3):
        f()
    t = time.perf_counter()
    for _ in range(iters):
        f()
    print(f"{name}: {(time.perf_counter() - t) / iters * 1e3:.3f} ms")


for n in (128, 512):
    a, b = torch.randn(n, n), torch.randn(n, n)
    bench(f"matmul {n}x{n}", lambda: a @ b)
x, w = torch.randn(64, 32, 14, 14), torch.randn(64, 32, 3, 3)
bench("conv2d [64,32,14,14] * [64,32,3,3]", lambda: F.conv2d(x, w, padding=1))
model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
opt = torch.optim.Adam(model.parameters(), 1e-3)
x, y = torch.randn(128, 784), torch.randint(0, 10, (128,))


def step():
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()


bench("MLP 784-512-10 train step, batch 128", step)
