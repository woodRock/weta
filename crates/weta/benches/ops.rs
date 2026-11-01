//! Fixed-size benchmarks: matmul, conv2d and one full training step.
//! Compare with PyTorch on the same machine via `scripts/bench_torch.py`.
//!
//! ```text
//! cargo bench -p weta
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use weta::nn::functional as F;
use weta::prelude::*;

fn matmul(c: &mut Criterion) {
    for n in [128, 512] {
        let (a, b) = (Tensor::randn(&[n, n]), Tensor::randn(&[n, n]));
        c.bench_function(&format!("matmul {n}x{n}"), |bch| bch.iter(|| a.matmul(&b)));
    }
}

fn conv(c: &mut Criterion) {
    let x = Tensor::randn(&[64, 32, 14, 14]);
    let w = Tensor::randn(&[64, 32, 3, 3]);
    c.bench_function("conv2d [64,32,14,14] * [64,32,3,3]", |b| b.iter(|| F::conv2d(&x, &w, None, 1, 1)));
}

fn train_step(c: &mut Criterion) {
    let model = nn::Sequential::new().add(nn::Linear::new(784, 512)).add(nn::ReLU).add(nn::Linear::new(512, 10));
    let mut opt = optim::Adam::new(model.parameters(), 1e-3);
    let x = Tensor::randn(&[128, 784]);
    let y = Tensor::randint(0, 10, &[128]);
    c.bench_function("MLP 784-512-10 train step, batch 128", |b| {
        b.iter(|| {
            let loss = F::cross_entropy(&model.forward(&x), &y);
            opt.zero_grad();
            loss.backward();
            opt.step();
        })
    });
}

criterion_group!(benches, matmul, conv, train_step);
criterion_main!(benches);
