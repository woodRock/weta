//! Gradient checking: compare an op's analytic gradient with central finite
//! differences,
//!
//! ```text
//! ∂L/∂xᵢ ≈ (L(x + h eᵢ) − L(x − h eᵢ)) / 2h,   h = 1e-5
//! ```
//!
//! in `f64`, where the approximation error (O(h²) ≈ 1e-10) and rounding
//! error (≈ 1e-16 / h ≈ 1e-11) are both far below the tolerance. To check a
//! function with a non-scalar output `y`, we differentiate the scalar
//! `L = Σ r ⊙ y` with fixed random weights `r`, which exercises every output
//! element with a different weight. Every differentiable op in weta has a
//! gradcheck test; this is what would have caught the diagonal-only softmax
//! bug in ai-accelerate in seconds.

use weta_core::DType;

use crate::{no_grad, RawTensor, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct GradcheckOptions {
    pub eps: f64,
    /// Largest allowed `|analytic − numeric| / max(1, |analytic|, |numeric|)`.
    pub tol: f64,
}

impl Default for GradcheckOptions {
    fn default() -> Self {
        GradcheckOptions { eps: 1e-5, tol: 1e-6 }
    }
}

/// A one-element view of element `flat` (row-major) of `t`.
fn element(t: &RawTensor, mut flat: usize) -> RawTensor {
    let mut v = t.clone();
    for d in (0..t.ndim()).rev() {
        let n = t.shape()[d];
        v = v.narrow(d as isize, flat % n, 1);
        flat /= n;
    }
    v
}

/// Checks `f`'s gradients with respect to every input that requires grad.
/// Inputs must be `F64`. Returns a description of the first mismatch.
pub fn gradcheck(f: impl Fn(&[Tensor]) -> Tensor, inputs: &[Tensor], opts: GradcheckOptions) -> Result<(), String> {
    for (i, x) in inputs.iter().enumerate() {
        if x.dtype() != DType::F64 {
            return Err(format!("gradcheck: input {i} is {}, expected f64", x.dtype()));
        }
    }
    let y = f(inputs);
    let r = Tensor::from_raw(RawTensor::randn(y.shape(), DType::F64, y.device()));
    let loss = |xs: &[Tensor]| (&f(xs) * &r).sum();

    for x in inputs {
        x.zero_grad();
    }
    loss(inputs).backward();

    for (i, x) in inputs.iter().enumerate() {
        if !x.requires_grad() {
            continue;
        }
        let analytic = match x.grad() {
            Some(g) => g.to_vec::<f64>(),
            None => vec![0.0; x.numel()],
        };
        let _guard = no_grad();
        for (k, &a) in analytic.iter().enumerate() {
            let e = element(x.data(), k);
            let orig = e.item();
            e.fill_(orig + opts.eps);
            let plus = loss(inputs).item();
            e.fill_(orig - opts.eps);
            let minus = loss(inputs).item();
            e.fill_(orig);
            let numeric = (plus - minus) / (2.0 * opts.eps);
            let err = (a - numeric).abs() / 1f64.max(a.abs()).max(numeric.abs());
            // Written so a NaN error fails too.
            if err.is_nan() || err > opts.tol {
                return Err(format!(
                    "gradcheck: input {i} (shape {:?}), element {k}: analytic {a:.10e}, numeric {numeric:.10e}, relative error {err:.3e}",
                    x.shape()
                ));
            }
        }
    }
    Ok(())
}
