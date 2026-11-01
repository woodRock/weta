# weta

A small, readable deep learning framework in Rust: N-dimensional tensors,
reverse-mode autodiff, the standard layers and optimisers, a CPU backend that
runs anywhere, and CUDA behind a Cargo feature. The API follows PyTorch as
closely as Rust allows, so PyTorch code translates almost line for line.

The wētā is a New Zealand insect: small, ancient and nearly indestructible,
like a lightweight framework should be.

```rust
use weta::nn::functional as F;
use weta::prelude::*;

#[derive(Module)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
    drop: nn::Dropout,
}

impl Net {
    fn new() -> Self {
        Net { fc1: nn::Linear::new(784, 128), fc2: nn::Linear::new(128, 10), drop: nn::Dropout::new(0.1) }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = F::relu(&self.fc1.forward(&x.flatten(1, -1)));
        self.fc2.forward(&self.drop.forward(&x)) // logits
    }
}

fn main() -> weta::Result<()> {
    let device = Device::cuda_if_available();
    let mut model = Net::new().to(device);
    let mut opt = optim::Adam::new(model.parameters(), 1e-3);
    let train = data::DataLoader::new(data::datasets::Mnist::train("data")?, 64).shuffle(true);

    for epoch in 0..5 {
        model.train();
        let mut last = 0.0;
        for (x, y) in &train {
            let (x, y) = (x.to(device), y.to(device));
            let loss = F::cross_entropy(&model.forward(&x), &y);
            opt.zero_grad();
            loss.backward();
            opt.step();
            last = loss.item();
        }
        println!("epoch {epoch}: last loss {last:.4}");
    }
    model.state_dict().save("mnist.safetensors")?;
    Ok(())
}
```

## Results

MNIST test accuracy, from the examples as committed (seed 42):

| Model | Example | Accuracy | Epoch time, laptop CPU (M-series) | Epoch time, RTX A5000 |
| --- | --- | --- | --- | --- |
| MLP 784-512-256-10, dropout | `mnist_mlp`, 10 epochs | **98.64%** (98.68% on GPU) | 4 s | 0.8 s |
| CNN, 2 conv + BatchNorm, 422k params | `mnist_cnn`, 5 epochs | **99.29%** (99.30% on GPU) | ~85 s | 5.2 s |
| MLP smoke test, 5,000 images | `cargo test` | 92.8% | 3 s for the whole test | |

PyTorch gets 91.3–91.4% on the smoke-test setup (same model, data and
optimiser), so the numbers line up.

Speed against PyTorch 2.7 on the same laptop CPU (`cargo bench -p weta`,
`python3 scripts/bench_torch.py`). The goal is to track the gap, not to win:

| Benchmark | weta | PyTorch | Gap |
| --- | --- | --- | --- |
| matmul 128×128 | 74 µs | 5 µs | 15× |
| matmul 512×512 | 0.82 ms | 0.20 ms | 4× |
| conv2d [64,32,14,14] * [64,32,3,3] | 2.25 ms | 1.65 ms | 1.4× |
| MLP 784-512-10 train step, batch 128 | 4.4 ms | 1.0 ms | 4.5× |

PyTorch on macOS uses Apple's Accelerate (AMX units) for matmul; weta uses
the portable `gemm` crate.

```bash
cargo run --release --example mnist_mlp -- --data data --epochs 10
cargo run --release --example mnist_cnn -- --data data --epochs 5
cargo run --release --example mnist_cnn --features cuda -- --device cuda   # on a GPU
```

MNIST is downloaded and cached under `--data` on first use.

## PyTorch → weta

| PyTorch | weta | Why it differs |
| --- | --- | --- |
| `torch.tensor([[1., 2.], [3., 4.]])` | `tensor!([[1., 2.], [3., 4.]])` | No nested-list literals in Rust |
| `torch.randn(64, 784)` | `Tensor::randn(&[64, 784])` | Shapes are slices |
| `x.to("cuda")` | `x.to(Device::Cuda(0))` | An enum, so a typo is a compile error |
| `x @ w.T` | `x.matmul(&w.t())` | No `@`; `+ - * /` are overloaded |
| `x[:, 0]` | `x.i((.., 0))` | `Index` can't return a new tensor |
| `x.requires_grad_()` | `x.requires_grad_(true)` | |
| `x.grad` | `x.grad()` → `Option<Tensor>` | No gradient is `None`, not zeros |
| `with torch.no_grad():` | `let _guard = no_grad();` | RAII guard |
| `class Net(nn.Module)` | `#[derive(Module)] struct Net` | Parameters found from the fields |
| `optim.Adam(p, lr=1e-3, betas=(0.9, 0.99))` | `optim::Adam::new(p, 1e-3).betas(0.9, 0.99)` | Builders instead of keyword arguments |
| `torch.save(model.state_dict(), p)` | `model.state_dict().save(p)?` | safetensors, readable from PyTorch |
| `RuntimeError: mat1 and mat2 shapes...` | panic, or `try_matmul` → `Result` | e.g. `matmul: [64, 784] × [128, 784]: inner dims 784 ≠ 128` |

## Architecture

A Cargo workspace of small crates with a strict dependency direction. The
facade crate `weta` re-exports them, so users add one dependency.

```text
weta-core ─┬─ weta-cpu ──┐
           └─ weta-cuda ─┴─ weta-tensor ─┬─ weta-autograd ─┬─ weta-nn (+ weta-derive)
                                         │                 ├─ weta-optim
                                         │                 └─ weta-data
                                         └─ weta-io ────── weta-infer
```

| Crate | Lines | What it holds |
| --- | --- | --- |
| `weta-core` | ~870 | `DType`, `Device`, `Layout` (shape, strides, offset), op enums, the `Backend` trait |
| `weta-cpu` | ~610 | The reference backend: strided kernels, rayon, `gemm` |
| `weta-cuda` | ~790 | NVRTC kernels and cuBLAS through `cudarc` (feature `cuda`) |
| `weta-tensor` | ~1330 | Strided N-d tensors with no autograd (the NumPy layer) |
| `weta-autograd` | ~1930 | The autograd `Tensor`, graph, `backward`, `gradcheck`, `tensor!`, indexing |
| `weta-nn` | ~1320 | `Module`, layers, `functional`, `init`, losses |
| `weta-optim` | ~380 | SGD (momentum, Nesterov), Adam, AdamW, LR schedulers |
| `weta-data` | ~480 | `Dataset`, `DataLoader`, MNIST, CIFAR-10 |
| `weta-io` | ~210 | safetensors state dicts |
| `weta-infer` | ~300 | Inference without autograd |

### Design notes

- **Tensors** are a cheap handle to shared storage plus a layout. Transpose,
  narrow, reshape of contiguous data and broadcasting are new layouts over
  the same storage. Device and dtype are runtime values, as in PyTorch.
- **Backends implement about a dozen primitives** (`unary`, `binary`,
  `reduce`, `matmul`, `im2col`, `col2im`, `copy_strided`, `cast`, ...), each
  taking strided views. Everything else is composite: convolution is
  `im2col` plus one matmul, pooling is `im2col` plus a reduction, and their
  backward passes come from `col2im`. A new backend is a few hundred lines.
- **Autograd** records one node per op. Each op implements the `Op` trait
  with its backward pass next to its forward, written in ordinary tensor ops,
  so it runs on every backend and is itself differentiable (second-order
  gradients work). `backward` sorts the graph once and visits nodes in
  reverse, summing a node's gradient contributions before visiting it, so a
  tensor used twice is handled correctly. There's no recursion anywhere
  (backward or dropping a graph), so a 100,000-op chain is fine.
- **Saved tensors are versioned**: modifying one in place before backward
  fails loudly instead of giving a silently wrong gradient.
- **Fused ops where the maths demands it**: `cross_entropy` takes logits and
  gives the stable `softmax(z) − y` gradient; standalone `softmax` uses the
  full Jacobian-vector product `s ⊙ (g − Σ s ⊙ g)`.

Two places where the implementation departs from the original spec:

1. **Graph storage.** The spec proposed a thread-local arena tape indexed by
   `NodeId`. weta instead links nodes with `Arc`s, edges pointing only
   backwards (so no cycles, and no `RefCell`). With an arena, nodes from a
   forward pass that never calls `backward` (an evaluation loop that forgets
   `no_grad`) stay on the tape forever; with `Arc`s, a graph is freed as soon
   as its last tensor is dropped, and tensors can cross threads. Backward is
   still "sort once, visit in reverse".
2. **Convolution in the `Backend` trait** is `im2col`/`col2im` rather than a
   `conv2d` kernel, which also gives pooling and both convolution gradients
   from the same two kernels.

## Testing

`cargo test --workspace` takes about 20 s and needs no GPU or Python.

| Suite | What it checks |
| --- | --- |
| Unit tests | Layouts, strides, broadcasting, kernels, every op's forward pass |
| `weta-autograd/tests/gradcheck.rs`, `weta/tests/nn.rs` | Every differentiable op against central finite differences in f64 (relative error < 1e-6), including broadcast and non-contiguous inputs |
| `weta/tests/properties.rs` | proptest: reshape round-trips, `transpose(transpose(x)) = x`, broadcasting laws, `(AB)ᵀ = BᵀAᵀ` |
| `weta/tests/reference.rs` | Outputs and gradients match PyTorch to 1e-9 on fixed inputs (fixtures from `scripts/make_fixtures.py`), including SGD/Adam/AdamW steps |
| `weta/tests/smoke_mnist.rs` | An MLP reaches ≥ 90% on a 5,000-image MNIST subset in under 60 s |
| `weta/tests/cuda_parity.rs` | Each CUDA kernel matches the CPU backend; a full training step matches |

```bash
cargo test --workspace
cargo test --release --features cuda -p weta --test cuda_parity   # on a GPU machine
cargo bench -p weta && python3 scripts/bench_torch.py             # compare with PyTorch
```

## Deployment

`weta-infer` depends only on `weta-tensor` and `weta-io`: no autograd,
optimisers or data loaders. `nn::utils::export(&model, path)` writes one
safetensors file holding the weights and, in its metadata, the architecture,
so `weta_infer::Model::from_bytes(include_bytes!("model.safetensors"))` is a
complete deployable model.

```bash
cargo run --release --no-default-features --features infer --example infer_mnist -- --model mnist_cnn.safetensors
```

- The inference example binary is about 500 KB with `--profile release-small`
  (size-optimised) or 970 KB with `--release` (about 2.7× faster), before
  weights. The exported MNIST CNN above is 1.7 MB of weights and scores the
  same 99.29% through `weta-infer` as in training. Add `--features rayon` for
  multi-threaded inference.
- `weta-infer` builds for `wasm32-unknown-unknown`; CI also checks
  `aarch64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`.
- State dicts use PyTorch's names and layouts, so weights move both ways via
  `safetensors.torch`.

## Adding an op

Implement `Op` (the backward pass) and record the forward result with
`Tensor::from_op`; nothing in the core changes. See
[`examples/custom_op.rs`](crates/weta/examples/custom_op.rs), which adds
softplus and gradchecks it.

## Licence

MIT. See [LICENSE](LICENSE).
