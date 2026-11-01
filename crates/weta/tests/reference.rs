//! weta against PyTorch: fixed inputs, and PyTorch's outputs and gradients,
//! from `scripts/make_fixtures.py` (all f64). Loading the fixture also
//! checks that safetensors written by PyTorch read correctly.

use weta::autograd::Reduction;
use weta::io::StateDict;
use weta::nn::functional as F;
use weta::optim::{Adam, AdamW, Optimizer, SGD};
use weta::prelude::*;

fn fixtures() -> StateDict {
    StateDict::load(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/reference.safetensors")).unwrap()
}

fn get(sd: &StateDict, name: &str) -> Tensor {
    Tensor::from_raw(sd.get(name).unwrap_or_else(|| panic!("missing fixture {name}")).clone())
}

fn leaf(sd: &StateDict, name: &str) -> Tensor {
    Tensor::from_raw(sd.get(name).unwrap().copy()).requires_grad_(true)
}

#[track_caller]
fn close(name: &str, got: &Tensor, want: &Tensor) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    for (i, (a, b)) in got.to_vec::<f64>().iter().zip(want.to_vec::<f64>()).enumerate() {
        assert!((a - b).abs() <= 1e-9 + 1e-9 * b.abs(), "{name}[{i}]: weta {a} vs PyTorch {b}");
    }
}

#[test]
fn mlp_with_label_smoothing() {
    let sd = fixtures();
    let (x, w1, b1, w2, b2) = (leaf(&sd, "mlp.x"), leaf(&sd, "mlp.w1"), leaf(&sd, "mlp.b1"), leaf(&sd, "mlp.w2"), leaf(&sd, "mlp.b2"));
    let logits = F::linear(&F::relu(&F::linear(&x, &w1, Some(&b1))), &w2, Some(&b2));
    let loss = F::cross_entropy_with(&logits, &get(&sd, "mlp.y"), Reduction::Mean, 0.1);
    loss.backward();
    close("logits", &logits, &get(&sd, "mlp.logits"));
    close("loss", &loss, &get(&sd, "mlp.loss"));
    for (n, t) in [("x", &x), ("w1", &w1), ("b1", &b1), ("w2", &w2), ("b2", &b2)] {
        close(n, &t.grad().unwrap(), &get(&sd, &format!("mlp.grad.{n}")));
    }
}

#[test]
fn cnn_with_batch_norm() {
    let sd = fixtures();
    let (x, w, b, gamma, beta) =
        (leaf(&sd, "cnn.x"), leaf(&sd, "cnn.w"), leaf(&sd, "cnn.b"), leaf(&sd, "cnn.gamma"), leaf(&sd, "cnn.beta"));
    let (rm, rv) = (Tensor::zeros(&[3]).to_dtype(DType::F64), Tensor::ones(&[3]).to_dtype(DType::F64));
    let h = F::conv2d(&x, &w, Some(&b), 1, 1);
    let h = F::batch_norm(&h, Some(&rm), Some(&rv), Some(&gamma), Some(&beta), true, 0.1, 1e-5);
    let h = F::max_pool2d(&F::relu(&h), 2, 2);
    let loss = F::mse_loss(&h, &get(&sd, "cnn.target"));
    loss.backward();
    close("out", &h, &get(&sd, "cnn.out"));
    close("loss", &loss, &get(&sd, "cnn.loss"));
    close("running_mean", &rm, &get(&sd, "cnn.running_mean"));
    close("running_var", &rv, &get(&sd, "cnn.running_var"));
    for (n, t) in [("x", &x), ("w", &w), ("b", &b), ("gamma", &gamma), ("beta", &beta)] {
        close(n, &t.grad().unwrap(), &get(&sd, &format!("cnn.grad.{n}")));
    }
}

#[test]
fn strided_conv_and_padded_avg_pool() {
    let sd = fixtures();
    let (x, w) = (leaf(&sd, "conv2.x"), leaf(&sd, "conv2.w"));
    let h = F::avg_pool2d(&F::conv2d(&x, &w, None, 2, 0), 2, 2, 1);
    close("out", &h, &get(&sd, "conv2.out"));
    (&h * &get(&sd, "conv2.g")).sum().backward();
    close("grad x", &x.grad().unwrap(), &get(&sd, "conv2.grad.x"));
    close("grad w", &w.grad().unwrap(), &get(&sd, "conv2.grad.w"));
}

#[test]
fn functions_forward_and_backward() {
    let sd = fixtures();
    type Fn1 = fn(&Tensor) -> Tensor;
    let fns: [(&str, Fn1); 7] = [
        ("softmax", |t| t.softmax(1)),
        ("log_softmax", |t| t.log_softmax(1)),
        ("gelu", F::gelu),
        ("gelu_tanh", F::gelu_tanh),
        ("silu", F::silu),
        ("layer_norm", |t| F::layer_norm(t, 1, None, None, 1e-5)),
        ("logsumexp", |t| t.logsumexp(1, false)),
    ];
    for (name, f) in fns {
        let z = leaf(&sd, "fn.z");
        let r = f(&z);
        close(name, &r, &get(&sd, &format!("fn.{name}.out")));
        (&r * &get(&sd, &format!("fn.{name}.g"))).sum().backward();
        close(&format!("{name} grad"), &z.grad().unwrap(), &get(&sd, &format!("fn.{name}.grad")));
    }
}

#[test]
fn optimisers_match_pytorch() {
    let sd = fixtures();
    let grads: Vec<Tensor> = (0..3).map(|i| get(&sd, &format!("opt.grad{i}"))).collect();
    let run = |name: &str, make: &dyn Fn(Vec<Tensor>) -> Box<dyn Optimizer>| {
        let p = leaf(&sd, "opt.p0");
        let mut opt = make(vec![p.clone()]);
        for g in &grads {
            p.set_grad(Some(g.clone()));
            opt.step();
        }
        close(name, &p, &get(&sd, &format!("opt.{name}")));
    };
    run("sgd", &|ps| Box::new(SGD::new(ps, 0.1).momentum(0.9).nesterov(true).weight_decay(0.01)));
    run("adam", &|ps| Box::new(Adam::new(ps, 0.01).betas(0.8, 0.99).weight_decay(0.1)));
    run("adamw", &|ps| Box::new(AdamW::new(ps, 0.01).weight_decay(0.1)));
}
