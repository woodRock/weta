"""Generates crates/weta/tests/fixtures/reference.safetensors: fixed inputs,
and PyTorch's outputs and gradients for them, so the Rust reference tests
can check weta against PyTorch without needing Python in CI.

    python3 scripts/make_fixtures.py   # needs torch and safetensors
"""
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
out = {}


def leaf(name, *shape):
    t = torch.randn(*shape, requires_grad=True)
    out[name] = t.detach().clone()
    return t


# MLP: linear -> relu -> linear -> cross_entropy (label smoothing 0.1)
x, w1, b1, w2, b2 = leaf("mlp.x", 4, 6), leaf("mlp.w1", 5, 6), leaf("mlp.b1", 5), leaf("mlp.w2", 3, 5), leaf("mlp.b2", 3)
y = torch.tensor([0, 2, 1, 2])
out["mlp.y"] = y
logits = F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)
loss = F.cross_entropy(logits, y, label_smoothing=0.1)
loss.backward()
out["mlp.logits"], out["mlp.loss"] = logits.detach(), loss.detach()
for n, t in [("x", x), ("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)]:
    out[f"mlp.grad.{n}"] = t.grad

# CNN: conv2d(padding 1, stride 1) -> batch_norm (train) -> relu -> max_pool2d -> mse
x, w, b = leaf("cnn.x", 2, 2, 6, 6), leaf("cnn.w", 3, 2, 3, 3), leaf("cnn.b", 3)
gamma, beta = leaf("cnn.gamma", 3), leaf("cnn.beta", 3)
rm, rv = torch.zeros(3), torch.ones(3)
h = F.conv2d(x, w, b, stride=1, padding=1)
h = F.batch_norm(h, rm, rv, gamma, beta, training=True, momentum=0.1, eps=1e-5)
h = F.max_pool2d(F.relu(h), 2)
target = torch.randn(h.shape)
out["cnn.target"] = target
loss = F.mse_loss(h, target)
loss.backward()
out["cnn.out"], out["cnn.loss"] = h.detach(), loss.detach()
out["cnn.running_mean"], out["cnn.running_var"] = rm, rv
for n, t in [("x", x), ("w", w), ("b", b), ("gamma", gamma), ("beta", beta)]:
    out[f"cnn.grad.{n}"] = t.grad

# Strided conv without bias, and avg pooling with padding.
x2, w2c = leaf("conv2.x", 1, 2, 7, 7), leaf("conv2.w", 4, 2, 3, 3)
h = F.avg_pool2d(F.conv2d(x2, w2c, None, stride=2, padding=0), 2, 2, padding=1)
g = torch.randn(h.shape)
out["conv2.g"], out["conv2.out"] = g, h.detach()
h.backward(g)
out["conv2.grad.x"], out["conv2.grad.w"] = x2.grad, w2c.grad

# Element-wise and normalisation functions, each with a random upstream gradient.
z = leaf("fn.z", 3, 5)
for name, f in [
    ("softmax", lambda t: F.softmax(t, 1)),
    ("log_softmax", lambda t: F.log_softmax(t, 1)),
    ("gelu", F.gelu),
    ("gelu_tanh", lambda t: F.gelu(t, approximate="tanh")),
    ("silu", F.silu),
    ("layer_norm", lambda t: F.layer_norm(t, (5,))),
    ("logsumexp", lambda t: torch.logsumexp(t, 1)),
]:
    z.grad = None
    r = f(z)
    g = torch.randn(r.shape)
    r.backward(g)
    out[f"fn.{name}.g"], out[f"fn.{name}.out"], out[f"fn.{name}.grad"] = g, r.detach(), z.grad.clone()

# Optimisers: three steps with fixed gradients.
grads = [torch.randn(4) for _ in range(3)]
for i, g in enumerate(grads):
    out[f"opt.grad{i}"] = g
p0 = torch.randn(4)
out["opt.p0"] = p0
for name, make in [
    ("sgd", lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, nesterov=True, weight_decay=0.01)),
    ("adam", lambda ps: torch.optim.Adam(ps, lr=0.01, betas=(0.8, 0.99), weight_decay=0.1)),
    ("adamw", lambda ps: torch.optim.AdamW(ps, lr=0.01, weight_decay=0.1)),
]:
    p = p0.clone().requires_grad_(True)
    opt = make([p])
    for g in grads:
        p.grad = g.clone()
        opt.step()
    out[f"opt.{name}"] = p.detach().clone()

save_file({k: v.contiguous() for k, v in out.items()}, "crates/weta/tests/fixtures/reference.safetensors")
print(f"wrote {len(out)} tensors")
