//! Backend parity: every CUDA kernel against the CPU reference. Runs only
//! with `--features cuda` on a machine with a GPU; elsewhere it's empty.
//!
//! ```text
//! cargo test --release --features cuda --test cuda_parity
//! ```

#![cfg(feature = "cuda")]

use weta::nn::functional as F;
use weta::prelude::*;
use weta::raw::{allclose, BinaryOp, Im2ColParams, Tensor as Raw, UnaryOp};

const GPU: Device = Device::Cuda(0);

fn gpu() -> bool {
    let ok = weta::cuda_is_available();
    if !ok {
        eprintln!("skipping: no CUDA device");
    }
    ok
}

#[track_caller]
fn same(name: &str, cpu: &Raw, gpu: &Raw, tol: f64) {
    assert_eq!(gpu.device(), GPU, "{name}: result not on the GPU");
    let g = gpu.to(Device::Cpu);
    assert_eq!(cpu.shape(), g.shape(), "{name}: shape");
    assert_eq!(cpu.dtype(), g.dtype(), "{name}: dtype");
    assert!(allclose(&g, cpu, tol, tol), "{name}:\ncpu {:?}\ngpu {:?}", cpu.to_vec::<f64>(), g.to_vec::<f64>());
}

fn rand(shape: &[usize], dt: DType) -> Raw {
    Raw::randn(shape, dt, Device::Cpu)
}

#[test]
fn unary_kernels() {
    if !gpu() {
        return;
    }
    for dt in [DType::F32, DType::F64] {
        let tol = if dt == DType::F32 { 1e-5 } else { 1e-12 };
        let x = rand(&[3, 5, 4], dt).transpose(0, 2); // strided input
        let pos = x.abs().add_scalar(0.1);
        for op in [
            UnaryOp::Neg,
            UnaryOp::Exp,
            UnaryOp::Abs,
            UnaryOp::Sign,
            UnaryOp::Relu,
            UnaryOp::Step,
            UnaryOp::Tanh,
            UnaryOp::Sigmoid,
            UnaryOp::Erf,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Affine { mul: 2.5, add: -1.0 },
            UnaryOp::GtScalar(0.3),
            UnaryOp::MaxScalar(0.1),
            UnaryOp::MinScalar(-0.2),
        ] {
            same(op.name(), &x.unary(op), &x.to(GPU).unary(op), tol);
        }
        for op in [UnaryOp::Log, UnaryOp::Sqrt, UnaryOp::Recip, UnaryOp::Powf(1.7)] {
            same(op.name(), &pos.unary(op), &pos.to(GPU).unary(op), tol);
        }
    }
    let i = Raw::from_vec(vec![-3i64, 0, 4, 7], &[4]);
    for op in [UnaryOp::Neg, UnaryOp::Abs, UnaryOp::Relu, UnaryOp::Affine { mul: 2.0, add: 1.0 }] {
        same(op.name(), &i.unary(op), &i.to(GPU).unary(op), 0.0);
    }
}

#[test]
fn binary_kernels_with_broadcasting() {
    if !gpu() {
        return;
    }
    for dt in [DType::F32, DType::F64] {
        let tol = if dt == DType::F32 { 1e-5 } else { 1e-12 };
        let a = rand(&[4, 3, 5], dt);
        let b = rand(&[5, 3], dt).t(); // [3, 5], non-contiguous, broadcast over dim 0
        let c = rand(&[4, 1, 1], dt);
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Maximum,
            BinaryOp::Minimum,
            BinaryOp::Eq,
            BinaryOp::Ne,
            BinaryOp::Lt,
            BinaryOp::Le,
            BinaryOp::Gt,
            BinaryOp::Ge,
        ] {
            same(op.name(), &a.binary(op, &b), &a.to(GPU).binary(op, &b.to(GPU)), tol);
            same(op.name(), &a.binary(op, &c), &a.to(GPU).binary(op, &c.to(GPU)), tol);
        }
        let (p, q) = (a.abs().add_scalar(0.1), b.abs());
        same("pow", &p.pow(&q), &p.to(GPU).pow(&q.to(GPU)), tol * 10.0);
    }
    let (x, y) = (Raw::from_vec(vec![7i64, -7, 3], &[3]), Raw::from_vec(vec![2i64, 2, 3], &[3]));
    for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Eq, BinaryOp::Maximum] {
        same(op.name(), &x.binary(op, &y), &x.to(GPU).binary(op, &y.to(GPU)), 0.0);
    }
}

#[test]
fn reductions() {
    if !gpu() {
        return;
    }
    let x = rand(&[6, 7, 8], DType::F64).permute(&[2, 0, 1]);
    let g = x.to(GPU);
    for d in 0..3isize {
        same("sum", &x.sum_dims(&[d], false), &g.sum_dims(&[d], false), 1e-12);
        same("max", &x.max_dims(&[d], true), &g.max_dims(&[d], true), 0.0);
        same("min", &x.min_dims(&[d], false), &g.min_dims(&[d], false), 0.0);
        same("argmax", &x.argmax(d, false), &g.argmax(d, false), 0.0);
        same("argmin", &x.argmin(d, false), &g.argmin(d, false), 0.0);
    }
    same("sum all", &x.sum(), &g.sum(), 1e-12);
    same("softmax", &x.softmax(1), &g.softmax(1), 1e-12);
    let i = Raw::from_vec(vec![3i64, 1, 4, 1, 5, 9], &[2, 3]);
    same("i64 sum", &i.sum_dims(&[1], false), &i.to(GPU).sum_dims(&[1], false), 0.0);
}

#[test]
fn matmul_layouts() {
    if !gpu() {
        return;
    }
    for dt in [DType::F32, DType::F64] {
        let tol = if dt == DType::F32 { 1e-4 } else { 1e-11 };
        let cases: Vec<(Raw, Raw)> = vec![
            (rand(&[5, 7], dt), rand(&[7, 3], dt)),
            (rand(&[7, 5], dt).t(), rand(&[7, 3], dt)),
            (rand(&[5, 7], dt), rand(&[3, 7], dt).t()),
            (rand(&[7, 5], dt).t(), rand(&[3, 7], dt).t()),
            (rand(&[2, 4, 5, 7], dt), rand(&[7, 3], dt)),          // broadcast batch
            (rand(&[4, 1, 5, 7], dt), rand(&[2, 7, 3], dt)),       // both broadcast
            (rand(&[1, 7], dt), rand(&[7, 1], dt)),                // degenerate dims
            (rand(&[5, 14], dt).step_by(1, 2), rand(&[7, 3], dt)), // a strided view cuBLAS can't read directly
            (rand(&[7], dt), rand(&[7, 3], dt)),
        ];
        for (i, (a, b)) in cases.iter().enumerate() {
            same(&format!("matmul case {i}"), &a.matmul(b), &a.to(GPU).matmul(&b.to(GPU)), tol);
        }
    }
}

#[test]
fn im2col_col2im_cast_copy() {
    if !gpu() {
        return;
    }
    let x = rand(&[2, 3, 7, 6], DType::F32);
    let p = Im2ColParams { kernel: (3, 2), stride: (2, 1), padding: (1, 1), dilation: (1, 2) };
    let cols = x.im2col(p);
    same("im2col", &cols, &x.to(GPU).im2col(p), 0.0);
    same("im2col strided", &x.transpose(2, 3).im2col(p), &x.to(GPU).transpose(2, 3).im2col(p), 0.0);
    same("col2im", &cols.col2im(p, (3, 7, 6)), &cols.to(GPU).col2im(p, (3, 7, 6)), 1e-5);
    for dt in [DType::F32, DType::F64, DType::I64] {
        same("cast", &x.to_dtype(dt), &x.to(GPU).to_dtype(dt), 1e-6);
    }
    let a = rand(&[2, 3], DType::F32);
    let b = rand(&[4, 3], DType::F32);
    same("cat", &Raw::cat(&[&a, &b], 0), &Raw::cat(&[&a.to(GPU), &b.to(GPU)], 0), 0.0);
    let g = a.to(GPU);
    g.narrow(1, 1, 1).fill_(5.0);
    a.narrow(1, 1, 1).fill_(5.0);
    same("in-place slice", &a, &g, 0.0);
}

#[test]
fn gradcheck_on_the_gpu() {
    if !gpu() {
        return;
    }
    use weta::autograd::{gradcheck, GradcheckOptions};
    let leaf = |s: &[usize]| Tensor::from_raw(Raw::randn(s, DType::F64, GPU)).requires_grad_(true);
    let (x, w, b) = (leaf(&[2, 2, 5, 5]), leaf(&[3, 2, 3, 3]), leaf(&[3]));
    let r = gradcheck(
        |t| F::max_pool2d(&F::relu(&F::conv2d(&t[0], &t[1], Some(&t[2]), 1, 1)), 2, 2).softmax(1),
        &[x, w, b],
        GradcheckOptions::default(),
    );
    r.unwrap();
}

#[test]
fn training_step_matches_cpu() {
    if !gpu() {
        return;
    }
    manual_seed(3);
    let make = || {
        nn::Sequential::new()
            .add(nn::Conv2d::new(1, 4, 3).padding(1))
            .add(nn::BatchNorm2d::new(4))
            .add(nn::ReLU)
            .add(nn::MaxPool2d::new(2))
            .add(nn::Flatten::default())
            .add(nn::Linear::new(4 * 4 * 4, 10))
    };
    let cpu = make();
    let mut gpu_model = make();
    gpu_model.load_state_dict(&cpu.state_dict(), true).unwrap();
    let gpu_model = gpu_model.to(GPU);
    let x = Tensor::randn(&[8, 1, 8, 8]);
    let y = Tensor::randint(0, 10, &[8]);
    let mut o1 = optim::Adam::new(cpu.parameters(), 1e-2);
    let mut o2 = optim::Adam::new(gpu_model.parameters(), 1e-2);
    for _ in 0..3 {
        let l1 = F::cross_entropy(&cpu.forward(&x), &y);
        o1.zero_grad();
        l1.backward();
        o1.step();
        let l2 = F::cross_entropy(&gpu_model.forward(&x.to(GPU)), &y.to(GPU));
        o2.zero_grad();
        l2.backward();
        o2.step();
        assert!((l1.item() - l2.item()).abs() < 1e-4, "loss {} vs {}", l1.item(), l2.item());
    }
    for ((n, a), (_, b)) in cpu.state_dict().iter().zip(gpu_model.state_dict().iter()) {
        same(n, a, b, 1e-3);
    }
}
