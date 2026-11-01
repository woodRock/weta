//! Every differentiable op against central finite differences in f64,
//! including broadcast and non-contiguous inputs.

#![allow(clippy::cloned_ref_to_slice_refs)]

use weta_autograd::{gradcheck, no_grad, BackwardOptions, GradcheckOptions, IndexOp, RawTensor, Reduction, Tensor};
use weta_core::{DType, Device, Im2ColParams};

fn leaf(shape: &[usize]) -> Tensor {
    Tensor::from_raw(RawTensor::randn(shape, DType::F64, Device::Cpu)).requires_grad_(true)
}

/// Strictly positive inputs, for log, sqrt, pow.
fn pos(shape: &[usize]) -> Tensor {
    Tensor::from_raw(RawTensor::rand(shape, DType::F64, Device::Cpu).add_scalar(0.5)).requires_grad_(true)
}

/// A non-contiguous leaf: the transpose of a random `[b, a]` matrix.
fn transposed(a: usize, b: usize) -> Tensor {
    Tensor::from_raw(RawTensor::randn(&[b, a], DType::F64, Device::Cpu).t()).requires_grad_(true)
}

#[track_caller]
fn check(f: impl Fn(&[Tensor]) -> Tensor, inputs: &[Tensor]) {
    if let Err(e) = gradcheck(f, inputs, GradcheckOptions::default()) {
        panic!("{e}");
    }
}

#[test]
fn unary_ops() {
    let x = leaf(&[3, 4]);
    check(|x| -&x[0], &[x.clone()]);
    check(|x| x[0].exp(), &[x.clone()]);
    check(|x| x[0].tanh(), &[x.clone()]);
    check(|x| x[0].sigmoid(), &[x.clone()]);
    check(|x| x[0].relu(), &[x.clone()]);
    check(|x| x[0].abs(), &[x.clone()]);
    check(|x| x[0].erf(), &[x.clone()]);
    check(|x| x[0].sin(), &[x.clone()]);
    check(|x| x[0].cos(), &[x.clone()]);
    check(|x| x[0].affine(3.0, -1.0), &[x.clone()]);
    check(|x| x[0].clamp(-0.5, 0.5), &[x.clone()]);
    let p = pos(&[3, 4]);
    check(|x| x[0].log(), &[p.clone()]);
    check(|x| x[0].sqrt(), &[p.clone()]);
    check(|x| x[0].reciprocal(), &[p.clone()]);
    check(|x| x[0].powf(2.5), &[p.clone()]);
    check(|x| 2.0 / &x[0], &[p]);
}

#[test]
fn binary_ops_with_broadcasting() {
    let (a, b, c) = (leaf(&[4, 3]), leaf(&[3]), leaf(&[4, 1]));
    check(|x| &x[0] + &x[1], &[a.clone(), b.clone()]);
    check(|x| &x[0] - &x[1], &[a.clone(), c.clone()]);
    check(|x| &x[0] * &x[1], &[a.clone(), b.clone()]);
    check(|x| &x[0] * &x[1], &[b.clone(), c.clone()]); // both broadcast: [3] × [4, 1]
    check(|x| x[0].maximum(&x[1]), &[a.clone(), b.clone()]);
    check(|x| x[0].minimum(&x[1]), &[a.clone(), c.clone()]);
    let (p, q) = (pos(&[4, 3]), pos(&[3]));
    check(|x| &x[0] / &x[1], &[a.clone(), q.clone()]);
    check(|x| x[0].pow(&x[1]), &[p, q]);
}

#[test]
fn non_contiguous_inputs() {
    let (a, b) = (transposed(3, 4), leaf(&[4, 2]));
    assert!(!a.is_contiguous());
    check(|x| x[0].matmul(&x[1]), &[a.clone(), b]);
    check(|x| (&x[0] * &x[0]).sum_dims(&[0], false), &[a.clone()]);
    check(|x| x[0].softmax(1), &[a.clone()]);
    check(|x| x[0].reshape(&[-1]), &[a]);
}

#[test]
fn reductions() {
    let x = leaf(&[2, 3, 4]);
    check(|x| x[0].sum(), &[x.clone()]);
    check(|x| x[0].sum_dims(&[1], false), &[x.clone()]);
    check(|x| x[0].sum_dims(&[0, 2], true), &[x.clone()]);
    check(|x| x[0].mean_dims(&[-1], false), &[x.clone()]);
    check(|x| x[0].max_dims(&[1], false), &[x.clone()]);
    check(|x| x[0].min_dims(&[2], true), &[x.clone()]);
    check(|x| x[0].var_dims(&[0, 2], true, false), &[x.clone()]);
    check(|x| x[0].logsumexp(1, false), &[x.clone()]);
    check(|x| x[0].max(), &[x]);
}

#[test]
fn softmax_family() {
    let x = leaf(&[4, 5]);
    // The full Jacobian, so wrong-class gradients are checked too.
    check(|x| x[0].softmax(1), &[x.clone()]);
    check(|x| x[0].softmax(0), &[x.clone()]);
    check(|x| x[0].log_softmax(1), &[x.clone()]);
    let y = Tensor::from_vec(vec![1i64, 0, 4, 2], &[4]);
    for red in [Reduction::Mean, Reduction::Sum, Reduction::None] {
        check(|x| x[0].cross_entropy(&y, red, 0.0), &[x.clone()]);
    }
    check(|x| x[0].cross_entropy(&y, Reduction::Mean, 0.1), &[x.clone()]);
    // cross_entropy matches log_softmax + NLL.
    let a = x.cross_entropy(&y, Reduction::Mean, 0.0).item();
    let b = -(x.log_softmax(1) * y.one_hot(5).to_dtype(DType::F64)).sum().item() / 4.0;
    assert!((a - b).abs() < 1e-12);
}

#[test]
fn matmul_variants() {
    check(|x| x[0].matmul(&x[1]), &[leaf(&[3, 4]), leaf(&[4, 2])]);
    check(|x| x[0].matmul(&x[1]), &[leaf(&[2, 3, 4]), leaf(&[4, 5])]); // broadcast batch
    check(|x| x[0].matmul(&x[1]), &[leaf(&[2, 1, 3, 4]), leaf(&[3, 4, 2])]);
    check(|x| x[0].matmul(&x[1]), &[leaf(&[4]), leaf(&[4, 3])]); // vector × matrix
    check(|x| x[0].matmul(&x[1]), &[leaf(&[3, 4]), leaf(&[4])]); // matrix × vector
    check(|x| x[0].matmul(&x[1].t()), &[leaf(&[3, 4]), leaf(&[5, 4])]);
}

#[test]
fn views_and_shapes() {
    let x = leaf(&[2, 3, 4]);
    check(|x| x[0].reshape(&[6, -1]), &[x.clone()]);
    check(|x| x[0].permute(&[2, 0, 1]), &[x.clone()]);
    check(|x| x[0].transpose(0, 2), &[x.clone()]);
    check(|x| x[0].narrow(2, 1, 2), &[x.clone()]);
    check(|x| x[0].i((.., 1)), &[x.clone()]);
    check(|x| x[0].i((1, 1.., ..3)), &[x.clone()]);
    check(|x| x[0].unsqueeze(1).squeeze(1), &[x.clone()]);
    check(|x| x[0].flatten(1, 2), &[x.clone()]);
    check(|x| x[0].i((.., .., 0..1)).broadcast_to(&[2, 3, 4]), &[x.clone()]);
    check(|x| x[0].contiguous(), &[x.clone()]);
    let y = leaf(&[2, 1, 4]);
    check(|x| Tensor::cat(&[&x[0], &x[1]], 1), &[x.clone(), y]);
    check(|x| Tensor::stack(&[&x[0], &x[0]], 0), &[x]);
}

#[test]
fn im2col_and_col2im() {
    let x = leaf(&[2, 2, 5, 5]);
    let p = Im2ColParams { kernel: (3, 3), stride: (2, 2), padding: (1, 1), dilation: (1, 1) };
    check(|x| x[0].im2col(p), &[x.clone()]);
    let cols = leaf(&[2, 2 * 9, 9]);
    check(|x| x[0].col2im(p, (2, 5, 5)), &[cols]);
}

#[test]
fn gradients_accumulate_for_reused_tensors() {
    // y = x·x + x uses x three times; dy/dx = 2x + 1, each path counted once.
    let x = Tensor::from_vec(vec![1.0f64, -2.0, 3.0], &[3]).requires_grad_(true);
    let h = &x * &x;
    let y = (&h + &x).sum();
    y.backward();
    assert_eq!(x.grad().unwrap().to_vec::<f64>(), vec![3.0, -3.0, 7.0]);
    // A diamond: two branches from one intermediate.
    let x = Tensor::from_vec(vec![2.0f64], &[1]).requires_grad_(true);
    let a = x.exp();
    let y = (&a * 2.0 + &a * 3.0).sum(); // 5 eˣ
    y.backward();
    assert!((x.grad().unwrap().item() - 5.0 * 2f64.exp()).abs() < 1e-12);
}

#[test]
fn deep_graphs_do_not_overflow_the_stack() {
    let x = Tensor::from_vec(vec![1.0f32], &[1]).requires_grad_(true);
    let mut y = x.clone();
    for _ in 0..100_000 {
        y = &y * 1.0;
    }
    y.sum().backward();
    assert_eq!(x.grad().unwrap().item(), 1.0);
}

#[test]
fn grads_accumulate_across_backward_calls() {
    let x = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).requires_grad_(true);
    (&x * 3.0).sum().backward();
    (&x * 3.0).sum().backward();
    assert_eq!(x.grad().unwrap().to_vec::<f32>(), vec![6.0, 6.0]);
    x.zero_grad();
    assert!(x.grad().is_none());
}

#[test]
#[should_panic(expected = "backward through the graph a second time")]
fn second_backward_needs_retain_graph() {
    let x = Tensor::from_vec(vec![1.0f32], &[1]).requires_grad_(true);
    let y = x.exp().sum();
    y.backward();
    y.backward();
}

#[test]
fn retain_graph_allows_second_backward() {
    let x = Tensor::from_vec(vec![1.0f32], &[1]).requires_grad_(true);
    let y = x.exp().sum();
    y.backward_with(BackwardOptions { retain_graph: true, ..Default::default() });
    y.backward();
    assert!((x.grad().unwrap().item() - 2.0 * 1f64.exp()).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "modified by an in-place operation")]
fn in_place_change_of_saved_tensor_is_detected() {
    let x = Tensor::from_vec(vec![1.0f32, 2.0], &[2]).requires_grad_(true);
    let h = &x * 2.0;
    let y = (&h * &h).sum();
    {
        let _g = no_grad();
        h.add_scalar_(1.0);
    }
    y.backward();
}

#[test]
fn no_grad_and_detach_stop_recording() {
    let x = Tensor::ones(&[2]).requires_grad_(true);
    {
        let _g = no_grad();
        assert!(!(&x * 2.0).requires_grad());
    }
    assert!((&x * 2.0).requires_grad());
    assert!(!(&x * 2.0).detach().requires_grad());
}

#[test]
fn second_order_gradients() {
    // d²/dx² of x³ = 6x, via create_graph.
    let x = Tensor::from_vec(vec![2.0f64], &[1]).requires_grad_(true);
    let y = (&x * &x * &x).sum();
    y.backward_with(BackwardOptions { create_graph: true, retain_graph: true });
    let g = x.grad().unwrap();
    assert!((g.item() - 12.0).abs() < 1e-12);
    x.zero_grad();
    g.sum().backward();
    assert!((x.grad().unwrap().item() - 12.0).abs() < 1e-12);
}

#[test]
fn tensor_macro_and_display() {
    let x = weta_autograd::tensor!([[1., 2.], [3., 4.]]);
    assert_eq!(x.shape(), &[2, 2]);
    assert_eq!(x.dtype(), DType::F32);
    let w = x.requires_grad_(true);
    let y = &w * 2.0;
    assert!(y.to_string().ends_with("grad_fn=<affineBackward>)"), "{y}");
}
