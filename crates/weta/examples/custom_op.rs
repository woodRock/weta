//! Adding a differentiable op without touching weta's core: implement `Op`
//! (the backward pass) and record the forward result with `Tensor::from_op`.
//! Here, softplus `log(1 + eˣ)`, whose derivative is `σ(x)`.
//!
//! ```text
//! cargo run --example custom_op
//! ```

use weta::autograd::{gradcheck, BackwardCtx, GradcheckOptions, Op, RawTensor};
use weta::prelude::*;

struct SoftplusBackward {
    x: Tensor,
}

impl Op for SoftplusBackward {
    fn name(&self) -> &'static str {
        "softplus"
    }
    fn backward(&self, _ctx: &BackwardCtx, grad_out: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(grad_out * &self.x.sigmoid())]
    }
}

fn softplus(x: &Tensor) -> Tensor {
    // Stable form: max(x, 0) + log(1 + e^{−|x|}), on raw tensors.
    let d = x.data();
    let out = d.relu().add(&d.abs().neg().exp().add_scalar(1.0).log());
    Tensor::from_op(out, SoftplusBackward { x: x.clone() }, &[x])
}

fn main() {
    let x = Tensor::from_raw(RawTensor::randn(&[3, 4], DType::F64, Device::Cpu)).requires_grad_(true);
    match gradcheck(|t| softplus(&t[0]), std::slice::from_ref(&x), GradcheckOptions::default()) {
        Ok(()) => println!("softplus gradcheck passed"),
        Err(e) => panic!("{e}"),
    }
    let y = softplus(&x).sum();
    y.backward();
    println!("x = {x}\ngrad = {}", x.grad().unwrap());
}
