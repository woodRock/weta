//! Differentiable ops. Each one computes its forward result with
//! `weta-tensor`, and implements [`Op`] with a backward pass written in
//! ordinary [`Tensor`] ops, right next to it.
//!
//! The maths for each backward is the vector–Jacobian product: given
//! `g = ∂L/∂y`, return `∂L/∂x = gᵀ ∂y/∂x`.

use weta_core::{BinaryOp, DType, Device, Im2ColParams, UnaryOp};

use crate::graph::{BackwardCtx, Op};
use crate::{RawTensor, Tensor};

/// A tensor saved for backward, with the version it had when saved. If an
/// in-place op changes it before backward runs, the gradient would be
/// silently wrong, so [`Saved::get`] fails loudly instead (as PyTorch does).
struct Saved {
    t: Tensor,
    version: u64,
}

impl Saved {
    fn new(t: &Tensor) -> Saved {
        Saved { t: t.clone(), version: t.data().version() }
    }
    /// Saves without the graph: for op outputs, which would otherwise form a
    /// cycle (output → node → saved output).
    fn detached(t: &RawTensor) -> Saved {
        Saved { t: Tensor::from_raw(t.clone()), version: t.version() }
    }
    #[track_caller]
    fn get(&self, op: &str) -> &Tensor {
        let now = self.t.data().version();
        assert_eq!(
            now, self.version,
            "{op}: one of the tensors needed for gradient computation has been modified by an in-place operation \
             (version {now}, expected {}). Use .copy() before modifying it",
            self.version
        );
        &self.t
    }
}

/// Sums `g` down to `shape`, undoing broadcasting: the gradient of a
/// broadcast input is the sum of the gradients of every element it was
/// copied to.
pub fn sum_to_shape(g: &Tensor, shape: &[usize]) -> Tensor {
    if g.shape() == shape {
        return g.clone();
    }
    let extra = g.ndim() - shape.len();
    let mut dims: Vec<isize> = (0..extra as isize).collect();
    for (i, &d) in shape.iter().enumerate() {
        if d == 1 && g.shape()[extra + i] != 1 {
            dims.push((extra + i) as isize);
        }
    }
    g.sum_dims(&dims, true).reshape_to(shape)
}

// ---------------------------------------------------------------------------
// Element-wise unary ops
// ---------------------------------------------------------------------------

struct UnaryBw {
    op: UnaryOp,
    x: Saved,
}

impl Op for UnaryBw {
    fn name(&self) -> &'static str {
        self.op.name()
    }

    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let x = self.x.get(self.name());
        // Derivatives recompute from x, which keeps them differentiable
        // (for higher-order gradients) without saving outputs.
        let gx = match self.op {
            UnaryOp::Neg => -g,
            UnaryOp::Exp => g * &x.exp(),
            UnaryOp::Log => g / x,
            UnaryOp::Sqrt => g * &x.sqrt().reciprocal() * 0.5,
            UnaryOp::Abs => g * &x.sign(),
            UnaryOp::Sign | UnaryOp::Step | UnaryOp::GtScalar(_) => g.zeros_like(),
            UnaryOp::Relu => g * &x.step(),
            // d tanh = 1 − tanh²
            UnaryOp::Tanh => {
                let t = x.tanh();
                g * &(1.0 - &(&t * &t))
            }
            // dσ = σ(1 − σ)
            UnaryOp::Sigmoid => {
                let s = x.sigmoid();
                g * &(&s * &(1.0 - &s))
            }
            // d erf = 2/√π · exp(−x²)
            UnaryOp::Erf => g * &(-(x * x)).exp() * std::f64::consts::FRAC_2_SQRT_PI,
            UnaryOp::Sin => g * &x.cos(),
            UnaryOp::Cos => -(g * &x.sin()),
            UnaryOp::Recip => -(g / &(x * x)),
            UnaryOp::Affine { mul, .. } => g * mul,
            UnaryOp::Powf(e) => g * &x.powf(e - 1.0) * e,
            UnaryOp::MaxScalar(v) => g * &x.gt_scalar(v),
            UnaryOp::MinScalar(v) => g * &(-x).gt_scalar(-v),
        };
        vec![Some(gx)]
    }
}

impl Tensor {
    pub fn unary(&self, op: UnaryOp) -> Tensor {
        let out = self.data().unary(op);
        Tensor::from_op(out, UnaryBw { op, x: Saved::new(self) }, &[self])
    }
}

macro_rules! unary_methods {
    ($($(#[$m:meta])* $name:ident => $op:expr),* $(,)?) => {
        impl Tensor {
            $($(#[$m])* pub fn $name(&self) -> Tensor { self.unary($op) })*
        }
    };
}

unary_methods! {
    exp => UnaryOp::Exp,
    /// Natural logarithm.
    log => UnaryOp::Log,
    sqrt => UnaryOp::Sqrt,
    abs => UnaryOp::Abs,
    /// −1, 0 or 1 (zero gradient).
    sign => UnaryOp::Sign,
    relu => UnaryOp::Relu,
    /// 1 where x > 0, else 0 (zero gradient).
    step => UnaryOp::Step,
    tanh => UnaryOp::Tanh,
    sigmoid => UnaryOp::Sigmoid,
    erf => UnaryOp::Erf,
    sin => UnaryOp::Sin,
    cos => UnaryOp::Cos,
    reciprocal => UnaryOp::Recip,
}

impl Tensor {
    pub fn neg(&self) -> Tensor {
        self.unary(UnaryOp::Neg)
    }
    /// `self * mul + add` in one kernel.
    pub fn affine(&self, mul: f64, add: f64) -> Tensor {
        self.unary(UnaryOp::Affine { mul, add })
    }
    pub fn add_scalar(&self, v: f64) -> Tensor {
        self.affine(1.0, v)
    }
    pub fn mul_scalar(&self, v: f64) -> Tensor {
        self.affine(v, 0.0)
    }
    pub fn powf(&self, e: f64) -> Tensor {
        self.unary(UnaryOp::Powf(e))
    }
    pub fn square(&self) -> Tensor {
        self * self
    }
    pub fn clamp_min(&self, v: f64) -> Tensor {
        self.unary(UnaryOp::MaxScalar(v))
    }
    pub fn clamp_max(&self, v: f64) -> Tensor {
        self.unary(UnaryOp::MinScalar(v))
    }
    pub fn clamp(&self, lo: f64, hi: f64) -> Tensor {
        self.clamp_min(lo).clamp_max(hi)
    }
    /// 1 where x > v, else 0 (zero gradient).
    pub fn gt_scalar(&self, v: f64) -> Tensor {
        self.unary(UnaryOp::GtScalar(v))
    }
}

// ---------------------------------------------------------------------------
// Element-wise binary ops, with broadcasting
// ---------------------------------------------------------------------------

struct BinaryBw {
    op: BinaryOp,
    a: Saved,
    b: Saved,
}

impl Op for BinaryBw {
    fn name(&self) -> &'static str {
        self.op.name()
    }

    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let (a, b) = (self.a.get(self.name()), self.b.get(self.name()));
        let (ga, gb) = match self.op {
            BinaryOp::Add => (g.clone(), g.clone()),
            BinaryOp::Sub => (g.clone(), -g),
            BinaryOp::Mul => (g * b, g * a),
            // d(a/b) = 1/b da − a/b² db
            BinaryOp::Div => (g / b, -(g * a / &(b * b))),
            // d(aᵇ) = b aᵇ⁻¹ da + aᵇ ln a db
            BinaryOp::Pow => {
                let p = a.pow(b);
                (g * b * &a.pow(&(b - 1.0)), g * &p * &a.log())
            }
            // The gradient goes to whichever input won (split evenly on ties).
            BinaryOp::Maximum | BinaryOp::Minimum => {
                let (wins, ties) = if self.op == BinaryOp::Maximum { (a.gt(b), a.eq(b)) } else { (a.lt(b), a.eq(b)) };
                let ma = &wins + &(&ties * 0.5);
                (g * &ma, g * &(1.0 - &ma))
            }
            _ => return vec![None, None], // comparisons have zero gradient
        };
        vec![
            ctx.needs_grad(0).then(|| sum_to_shape(&ga, ctx.input_shape(0))),
            ctx.needs_grad(1).then(|| sum_to_shape(&gb, ctx.input_shape(1))),
        ]
    }
}

impl Tensor {
    pub fn try_binary(&self, op: BinaryOp, other: &Tensor) -> crate::Result<Tensor> {
        let out = self.data().try_binary(op, other.data())?;
        if matches!(op, BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge) {
            return Ok(Tensor::from_raw(out));
        }
        Ok(Tensor::from_op(out, BinaryBw { op, a: Saved::new(self), b: Saved::new(other) }, &[self, other]))
    }

    #[track_caller]
    pub fn binary(&self, op: BinaryOp, other: &Tensor) -> Tensor {
        self.try_binary(op, other).unwrap_or_else(|e| panic!("{e}"))
    }
}

macro_rules! binary_methods {
    ($($name:ident, $try_name:ident => $op:expr),* $(,)?) => {
        impl Tensor {
            $(
                #[track_caller]
                pub fn $name(&self, other: &Tensor) -> Tensor { self.binary($op, other) }
                pub fn $try_name(&self, other: &Tensor) -> crate::Result<Tensor> { self.try_binary($op, other) }
            )*
        }
    };
}

binary_methods! {
    add, try_add => BinaryOp::Add, sub, try_sub => BinaryOp::Sub,
    mul, try_mul => BinaryOp::Mul, div, try_div => BinaryOp::Div,
    pow, try_pow => BinaryOp::Pow, maximum, try_maximum => BinaryOp::Maximum,
    minimum, try_minimum => BinaryOp::Minimum,
    eq, try_eq => BinaryOp::Eq, ne, try_ne => BinaryOp::Ne, lt, try_lt => BinaryOp::Lt,
    le, try_le => BinaryOp::Le, gt, try_gt => BinaryOp::Gt, ge, try_ge => BinaryOp::Ge,
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Re-inserts reduced dims (as size 1) so a gradient broadcasts back.
fn unsqueeze_dims(g: &Tensor, in_shape: &[usize], dims: &[usize], keepdim: bool) -> Tensor {
    if keepdim {
        return g.clone();
    }
    let mut shape = in_shape.to_vec();
    for &d in dims {
        shape[d] = 1;
    }
    g.reshape_to(&shape)
}

struct SumBw {
    dims: Vec<usize>,
    keepdim: bool,
}

impl Op for SumBw {
    fn name(&self) -> &'static str {
        "sum"
    }
    // Every input element contributes once to its sum: the gradient is g,
    // broadcast back over the reduced dims.
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let shape = ctx.input_shape(0);
        vec![Some(unsqueeze_dims(g, shape, &self.dims, self.keepdim).broadcast_to(shape).contiguous())]
    }
}

struct MaxBw {
    x: Saved,
    out: Saved,
    dims: Vec<usize>,
    keepdim: bool,
}

impl Op for MaxBw {
    fn name(&self) -> &'static str {
        "max"
    }
    // The gradient flows to the element(s) that attained the max, split
    // evenly between ties.
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let shape = ctx.input_shape(0);
        let x = self.x.get("max");
        let m = unsqueeze_dims(self.out.get("max"), shape, &self.dims, self.keepdim);
        let mask = x.eq(&m).to_dtype(x.dtype());
        let dims: Vec<isize> = self.dims.iter().map(|&d| d as isize).collect();
        let count = mask.sum_dims(&dims, true);
        let g = unsqueeze_dims(g, shape, &self.dims, self.keepdim);
        vec![Some(&(&mask / &count) * &g)]
    }
}

impl Tensor {
    fn norm_dims(&self, dims: &[isize]) -> Vec<usize> {
        let mut d: Vec<usize> = dims.iter().map(|&d| self.data().dim_index(d)).collect();
        d.sort_unstable();
        d.dedup();
        d
    }

    fn all_dims(&self) -> Vec<isize> {
        (0..self.ndim() as isize).collect()
    }

    /// Sum over `dims` (negative counts from the end).
    pub fn sum_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        let out = self.data().sum_dims(dims, keepdim);
        Tensor::from_op(out, SumBw { dims: self.norm_dims(dims), keepdim }, &[self])
    }

    /// Sum of all elements, as a 0-d tensor.
    pub fn sum(&self) -> Tensor {
        self.sum_dims(&self.all_dims(), false)
    }

    pub fn mean_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        let n: usize = self.norm_dims(dims).iter().map(|&d| self.shape()[d]).product();
        self.sum_dims(dims, keepdim) * (1.0 / n as f64)
    }

    pub fn mean(&self) -> Tensor {
        self.sum() * (1.0 / self.numel() as f64)
    }

    /// Max over `dims` (values only; see [`argmax`](Self::argmax)).
    pub fn max_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        let out = self.data().max_dims(dims, keepdim);
        let op = MaxBw { x: Saved::new(self), out: Saved::detached(&out), dims: self.norm_dims(dims), keepdim };
        Tensor::from_op(out, op, &[self])
    }

    pub fn max(&self) -> Tensor {
        self.max_dims(&self.all_dims(), false)
    }

    pub fn min_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        -&(-self).max_dims(dims, keepdim)
    }

    pub fn min(&self) -> Tensor {
        -&(-self).max()
    }

    /// Indices of the max along `dim`, as `I64` (no gradient).
    pub fn argmax(&self, dim: isize, keepdim: bool) -> Tensor {
        Tensor::from_raw(self.data().argmax(dim, keepdim))
    }

    pub fn argmin(&self, dim: isize, keepdim: bool) -> Tensor {
        Tensor::from_raw(self.data().argmin(dim, keepdim))
    }

    /// Variance over `dims`; `unbiased` divides by n − 1 (Bessel's correction).
    pub fn var_dims(&self, dims: &[isize], unbiased: bool, keepdim: bool) -> Tensor {
        let n: usize = self.norm_dims(dims).iter().map(|&d| self.shape()[d]).product();
        let c = self - &self.mean_dims(dims, true);
        let denom = if unbiased { n.saturating_sub(1).max(1) } else { n };
        (&c * &c).sum_dims(dims, keepdim) * (1.0 / denom as f64)
    }

    /// `log Σ exp(x)` over `dim`, shifted by the max so it can't overflow.
    pub fn logsumexp(&self, dim: isize, keepdim: bool) -> Tensor {
        let m = self.max_dims(&[dim], true).detach();
        let out = (self - &m).exp().sum_dims(&[dim], true).log() + &m;
        if keepdim {
            out
        } else {
            out.squeeze(dim)
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax family. Fused ops, because their Jacobians simplify.
// ---------------------------------------------------------------------------

struct SoftmaxBw {
    s: Saved,
    dim: isize,
}

impl Op for SoftmaxBw {
    fn name(&self) -> &'static str {
        "softmax"
    }
    // The full Jacobian-vector product, not just its diagonal:
    //   ∂L/∂z = s ⊙ (g − Σⱼ sⱼ gⱼ)
    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let s = self.s.get("softmax");
        let dot = (s * g).sum_dims(&[self.dim], true);
        vec![Some(s * &(g - &dot))]
    }
}

struct LogSoftmaxBw {
    out: Saved,
    dim: isize,
}

impl Op for LogSoftmaxBw {
    fn name(&self) -> &'static str {
        "log_softmax"
    }
    // y = z − logsumexp(z), so ∂L/∂z = g − softmax(z) · Σⱼ gⱼ.
    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let s = self.out.get("log_softmax").exp();
        vec![Some(g - &(&s * &g.sum_dims(&[self.dim], true)))]
    }
}

/// How a loss reduces over the batch, as PyTorch's `reduction=` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reduction {
    None,
    #[default]
    Mean,
    Sum,
}

struct CrossEntropyBw {
    probs: Saved,
    target: RawTensor,
    reduction: Reduction,
    label_smoothing: f64,
}

impl Op for CrossEntropyBw {
    fn name(&self) -> &'static str {
        "cross_entropy"
    }
    // With p = softmax(z) and a one-hot (or smoothed) target y, the
    // gradient of −Σ y log p is simply p − y. No softmax Jacobian needed.
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let p = self.probs.get("cross_entropy");
        let (n, c) = (p.size(0), p.size(1));
        let y = smoothed_targets(&self.target, c, self.label_smoothing, p.dtype());
        let diff = p - &Tensor::from_raw(y);
        let gx = match self.reduction {
            Reduction::Mean => &diff * &(g * (1.0 / n as f64)),
            Reduction::Sum => &diff * g,
            Reduction::None => &diff * &g.unsqueeze(1),
        };
        debug_assert_eq!(gx.shape(), ctx.input_shape(0));
        vec![Some(gx), None]
    }
}

fn smoothed_targets(target: &RawTensor, classes: usize, smoothing: f64, dtype: DType) -> RawTensor {
    let y = target.one_hot(classes, dtype);
    if smoothing > 0.0 {
        y.affine(1.0 - smoothing, smoothing / classes as f64)
    } else {
        y
    }
}

impl Tensor {
    /// Softmax along `dim`.
    pub fn softmax(&self, dim: isize) -> Tensor {
        let out = self.data().softmax(dim);
        let op = SoftmaxBw { s: Saved::detached(&out), dim };
        Tensor::from_op(out, op, &[self])
    }

    /// `log(softmax(x))`, computed stably.
    pub fn log_softmax(&self, dim: isize) -> Tensor {
        let out = self.data().log_softmax(dim);
        let op = LogSoftmaxBw { out: Saved::detached(&out), dim };
        Tensor::from_op(out, op, &[self])
    }

    /// Cross-entropy between logits `[N, C]` and `I64` class indices `[N]`:
    /// log-softmax and negative log-likelihood fused into one op.
    #[track_caller]
    pub fn cross_entropy(&self, target: &Tensor, reduction: Reduction, label_smoothing: f64) -> Tensor {
        assert_eq!(self.ndim(), 2, "cross_entropy: expected logits [N, C], got {:?}", self.shape());
        assert_eq!(target.dtype(), DType::I64, "cross_entropy: expected I64 class indices, got {}", target.dtype());
        assert_eq!(
            target.shape(),
            &self.shape()[..1],
            "cross_entropy: target shape {:?} doesn't match batch size of logits {:?}",
            target.shape(),
            self.shape()
        );
        let lsm = self.data().log_softmax(1);
        let y = smoothed_targets(target.data(), self.size(1), label_smoothing, self.dtype());
        let per_sample = lsm.mul(&y).sum_dims(&[1], false).neg();
        let out = match reduction {
            Reduction::None => per_sample,
            Reduction::Mean => per_sample.mean(),
            Reduction::Sum => per_sample.sum(),
        };
        let op = CrossEntropyBw { probs: Saved::detached(&lsm.exp()), target: target.data().clone(), reduction, label_smoothing };
        Tensor::from_op(out, op, &[self, target])
    }
}

// ---------------------------------------------------------------------------
// Views and shape ops
// ---------------------------------------------------------------------------

/// Backward for any op that only rearranges elements without changing their
/// order in memory: reshape, view, squeeze, unsqueeze, flatten.
struct ReshapeBw;

impl Op for ReshapeBw {
    fn name(&self) -> &'static str {
        "reshape"
    }
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(g.reshape_to(ctx.input_shape(0)))]
    }
}

struct PermuteBw {
    dims: Vec<usize>,
}

impl Op for PermuteBw {
    fn name(&self) -> &'static str {
        "permute"
    }
    // Undo the permutation with its inverse.
    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let mut inv = vec![0; self.dims.len()];
        for (i, &d) in self.dims.iter().enumerate() {
            inv[d] = i;
        }
        vec![Some(g.permute(&inv))]
    }
}

struct BroadcastBw;

impl Op for BroadcastBw {
    fn name(&self) -> &'static str {
        "expand"
    }
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(sum_to_shape(g, ctx.input_shape(0)))]
    }
}

struct NarrowBw {
    dim: usize,
    start: usize,
}

impl Op for NarrowBw {
    fn name(&self) -> &'static str {
        "narrow"
    }
    // The slice's gradient, placed back where it came from; zero elsewhere.
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(g.pad_into(ctx.input_shape(0), self.dim, self.start))]
    }
}

/// The adjoint of narrow: embeds a tensor in zeros (its backward is narrow).
struct PadBw {
    dim: usize,
    start: usize,
}

impl Op for PadBw {
    fn name(&self) -> &'static str {
        "pad"
    }
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(g.narrow(self.dim as isize, self.start, ctx.input_shape(0)[self.dim]))]
    }
}

struct IdentityBw(&'static str);

impl Op for IdentityBw {
    fn name(&self) -> &'static str {
        self.0
    }
    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(g.clone())]
    }
}

struct CatBw {
    dim: usize,
    sizes: Vec<usize>,
}

impl Op for CatBw {
    fn name(&self) -> &'static str {
        "cat"
    }
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let mut start = 0;
        self.sizes
            .iter()
            .enumerate()
            .map(|(i, &len)| {
                let s = start;
                start += len;
                ctx.needs_grad(i).then(|| g.narrow(self.dim as isize, s, len))
            })
            .collect()
    }
}

struct CastBw(DType);

impl Op for CastBw {
    fn name(&self) -> &'static str {
        "to_dtype"
    }
    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(g.to_dtype(self.0))]
    }
}

struct ToDeviceBw(Device);

impl Op for ToDeviceBw {
    fn name(&self) -> &'static str {
        "to"
    }
    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(g.to(self.0))]
    }
}

impl Tensor {
    fn view_op(&self, out: RawTensor) -> Tensor {
        Tensor::from_op(out, ReshapeBw, &[self])
    }

    pub fn try_reshape(&self, shape: &[isize]) -> crate::Result<Tensor> {
        Ok(self.view_op(self.data().try_reshape(shape)?))
    }

    /// Reshapes; one dim may be `-1` to infer it. Copies only if needed.
    #[track_caller]
    pub fn reshape(&self, shape: &[isize]) -> Tensor {
        self.try_reshape(shape).unwrap_or_else(|e| panic!("{e}"))
    }

    /// PyTorch's `view` (here the same as `reshape`).
    #[track_caller]
    pub fn view(&self, shape: &[isize]) -> Tensor {
        self.reshape(shape)
    }

    #[track_caller]
    pub fn reshape_to(&self, shape: &[usize]) -> Tensor {
        self.view_op(self.data().reshape_to(shape))
    }

    #[track_caller]
    pub fn unsqueeze(&self, dim: isize) -> Tensor {
        self.view_op(self.data().unsqueeze(dim))
    }

    #[track_caller]
    pub fn squeeze(&self, dim: isize) -> Tensor {
        self.view_op(self.data().squeeze(dim))
    }

    #[track_caller]
    pub fn flatten(&self, start: isize, end: isize) -> Tensor {
        self.view_op(self.data().flatten(start, end))
    }

    #[track_caller]
    pub fn permute(&self, dims: &[usize]) -> Tensor {
        let out = self.data().permute(dims);
        Tensor::from_op(out, PermuteBw { dims: dims.to_vec() }, &[self])
    }

    #[track_caller]
    pub fn transpose(&self, d0: isize, d1: isize) -> Tensor {
        let (a, b) = (self.data().dim_index(d0), self.data().dim_index(d1));
        let mut dims: Vec<usize> = (0..self.ndim()).collect();
        dims.swap(a, b);
        self.permute(&dims)
    }

    /// Swaps the last two dims (`x.T` for a matrix).
    #[track_caller]
    pub fn t(&self) -> Tensor {
        assert!(self.ndim() >= 2, "t: expected at least 2 dims, got shape {:?}", self.shape());
        self.transpose(-2, -1)
    }

    #[track_caller]
    pub fn broadcast_to(&self, shape: &[usize]) -> Tensor {
        Tensor::from_op(self.data().broadcast_to(shape), BroadcastBw, &[self])
    }

    /// PyTorch's `expand` (here the same as `broadcast_to`).
    #[track_caller]
    pub fn expand(&self, shape: &[usize]) -> Tensor {
        self.broadcast_to(shape)
    }

    #[track_caller]
    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Tensor {
        let d = self.data().dim_index(dim);
        Tensor::from_op(self.data().narrow(dim, start, len), NarrowBw { dim: d, start }, &[self])
    }

    /// Index `i` along `dim`, removing that dim: `x[:, i]`.
    #[track_caller]
    pub fn select(&self, dim: isize, i: usize) -> Tensor {
        let d = self.data().dim_index(dim) as isize;
        self.narrow(d, i, 1).squeeze(d)
    }

    /// Places this tensor in a zero tensor of `shape` at `start` along `dim`.
    pub fn pad_into(&self, shape: &[usize], dim: usize, start: usize) -> Tensor {
        let out = RawTensor::zeros(shape, self.dtype(), self.device());
        out.narrow(dim as isize, start, self.shape()[dim]).assign(self.data());
        Tensor::from_op(out, PadBw { dim, start }, &[self])
    }

    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        Tensor::from_op(self.data().contiguous(), IdentityBw("contiguous"), &[self])
    }

    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if dtype == self.dtype() {
            return self.clone();
        }
        let out = self.data().to_dtype(dtype);
        if !dtype.is_float() {
            return Tensor::from_raw(out);
        }
        Tensor::from_op(out, CastBw(self.dtype()), &[self])
    }

    /// Moves to a device. For a leaf (e.g. a parameter), the result is a new
    /// leaf with the same `requires_grad`, as `nn.Module.to` expects.
    #[track_caller]
    pub fn to(&self, device: Device) -> Tensor {
        if device == self.device() {
            return self.clone();
        }
        let out = self.data().to(device);
        if self.is_leaf() {
            return Tensor::from_raw(out).requires_grad_(self.requires_grad());
        }
        Tensor::from_op(out, ToDeviceBw(self.device()), &[self])
    }

    pub fn try_cat(tensors: &[&Tensor], dim: isize) -> crate::Result<Tensor> {
        let raws: Vec<&RawTensor> = tensors.iter().map(|t| t.data()).collect();
        let out = RawTensor::try_cat(&raws, dim)?;
        let d = raws[0].dim_index(dim);
        let sizes = tensors.iter().map(|t| t.shape()[d]).collect();
        Ok(Tensor::from_op(out, CatBw { dim: d, sizes }, tensors))
    }

    /// Concatenates along an existing dim (`torch.cat`).
    #[track_caller]
    pub fn cat(tensors: &[&Tensor], dim: isize) -> Tensor {
        Tensor::try_cat(tensors, dim).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Stacks along a new dim (`torch.stack`).
    #[track_caller]
    pub fn stack(tensors: &[&Tensor], dim: isize) -> Tensor {
        let us: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();
        Tensor::cat(&us.iter().collect::<Vec<_>>(), dim)
    }

    /// One-hot encoding of `I64` indices (no gradient).
    pub fn one_hot(&self, classes: usize) -> Tensor {
        Tensor::from_raw(self.data().one_hot(classes, DType::F32))
    }
}

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

struct MatmulBw {
    a: Saved,
    b: Saved,
}

impl Op for MatmulBw {
    fn name(&self) -> &'static str {
        "matmul"
    }
    // For Y = A B:  ∂L/∂A = G Bᵀ  and  ∂L/∂B = Aᵀ G, summed over any
    // broadcast batch dims.
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let (a, b) = (self.a.get("matmul"), self.b.get("matmul"));
        vec![
            ctx.needs_grad(0).then(|| sum_to_shape(&g.matmul(&b.t()), ctx.input_shape(0))),
            ctx.needs_grad(1).then(|| sum_to_shape(&a.t().matmul(g), ctx.input_shape(1))),
        ]
    }
}

impl Tensor {
    /// Matrix product with `torch.matmul` semantics (vectors and broadcast
    /// batch dims included).
    pub fn try_matmul(&self, other: &Tensor) -> crate::Result<Tensor> {
        // Vectors are promoted to matrices so backward only sees ≥ 2-D.
        if self.ndim() == 1 && other.ndim() >= 1 {
            return Ok(self.unsqueeze(0).try_matmul(other)?.squeeze(-2));
        }
        if other.ndim() == 1 && self.ndim() >= 2 {
            return Ok(self.try_matmul(&other.unsqueeze(1))?.squeeze(-1));
        }
        let out = self.data().try_matmul(other.data())?;
        Ok(Tensor::from_op(out, MatmulBw { a: Saved::new(self), b: Saved::new(other) }, &[self, other]))
    }

    #[track_caller]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        self.try_matmul(other).unwrap_or_else(|e| panic!("{e}"))
    }
}

// ---------------------------------------------------------------------------
// im2col / col2im: the basis of convolution and pooling
// ---------------------------------------------------------------------------

struct Im2ColBw {
    p: Im2ColParams,
}

impl Op for Im2ColBw {
    fn name(&self) -> &'static str {
        "im2col"
    }
    // im2col copies each pixel into every column whose window covers it, so
    // the adjoint sums those copies back: col2im.
    fn backward(&self, ctx: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        let s = ctx.input_shape(0);
        vec![Some(g.col2im(self.p, (s[1], s[2], s[3])))]
    }
}

struct Col2ImBw {
    p: Im2ColParams,
}

impl Op for Col2ImBw {
    fn name(&self) -> &'static str {
        "col2im"
    }
    fn backward(&self, _: &BackwardCtx, g: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(g.im2col(self.p))]
    }
}

impl Tensor {
    /// `[N, C, H, W]` → `[N, C·kh·kw, OH·OW]`.
    #[track_caller]
    pub fn im2col(&self, p: Im2ColParams) -> Tensor {
        Tensor::from_op(self.data().im2col(p), Im2ColBw { p }, &[self])
    }

    /// `[N, C·kh·kw, L]` → `[N, C, H, W]`, summing overlapping patches.
    #[track_caller]
    pub fn col2im(&self, p: Im2ColParams, chw: (usize, usize, usize)) -> Tensor {
        Tensor::from_op(self.data().col2im(p, chw), Col2ImBw { p }, &[self])
    }
}

// ---------------------------------------------------------------------------
// In-place ops. These change data without recording; like PyTorch, they
// refuse to touch a leaf that requires grad while grad mode is on.
// ---------------------------------------------------------------------------

impl Tensor {
    #[track_caller]
    fn check_inplace(&self, op: &str) {
        if crate::is_grad_enabled() && self.is_leaf() && self.requires_grad() {
            panic!("{op}: a leaf tensor that requires grad is being used in an in-place operation; wrap it in no_grad()");
        }
    }

    /// Copies `src` into this tensor's elements (`copy_`).
    #[track_caller]
    pub fn copy_(&self, src: &Tensor) {
        self.check_inplace("copy_");
        self.data().assign(src.data());
    }
    #[track_caller]
    pub fn zero_(&self) {
        self.check_inplace("zero_");
        self.data().zero_();
    }
    #[track_caller]
    pub fn fill_(&self, v: f64) {
        self.check_inplace("fill_");
        self.data().fill_(v);
    }
    #[track_caller]
    pub fn add_(&self, other: &Tensor) {
        self.check_inplace("add_");
        self.data().add_(other.data());
    }
    #[track_caller]
    pub fn sub_(&self, other: &Tensor) {
        self.check_inplace("sub_");
        self.data().sub_(other.data());
    }
    #[track_caller]
    pub fn mul_(&self, other: &Tensor) {
        self.check_inplace("mul_");
        self.data().mul_(other.data());
    }
    #[track_caller]
    pub fn div_(&self, other: &Tensor) {
        self.check_inplace("div_");
        self.data().div_(other.data());
    }
    #[track_caller]
    pub fn mul_scalar_(&self, v: f64) {
        self.check_inplace("mul_");
        self.data().mul_scalar_(v);
    }
    #[track_caller]
    pub fn add_scalar_(&self, v: f64) {
        self.check_inplace("add_");
        self.data().add_scalar_(v);
    }
}

// ---------------------------------------------------------------------------
// Operators: + − × ÷ on tensors and f64 scalars, owned or borrowed.
// ---------------------------------------------------------------------------

macro_rules! impl_op {
    ($trait:ident, $method:ident, $tensor_fn:ident, $scalar:expr, $rscalar:expr) => {
        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$tensor_fn(self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$tensor_fn(&self, &rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                Tensor::$tensor_fn(&self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                Tensor::$tensor_fn(self, &rhs)
            }
        }
        impl std::ops::$trait<f64> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f64) -> Tensor {
                let f: fn(&Tensor, f64) -> Tensor = $scalar;
                f(self, rhs)
            }
        }
        impl std::ops::$trait<f64> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: f64) -> Tensor {
                let f: fn(&Tensor, f64) -> Tensor = $scalar;
                f(&self, rhs)
            }
        }
        impl std::ops::$trait<&Tensor> for f64 {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                let f: fn(f64, &Tensor) -> Tensor = $rscalar;
                f(self, rhs)
            }
        }
        impl std::ops::$trait<Tensor> for f64 {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                let f: fn(f64, &Tensor) -> Tensor = $rscalar;
                f(self, &rhs)
            }
        }
    };
}

impl_op!(Add, add, add, |t, s| t.affine(1.0, s), |s, t| t.affine(1.0, s));
impl_op!(Sub, sub, sub, |t, s| t.affine(1.0, -s), |s, t| t.affine(-1.0, s));
impl_op!(Mul, mul, mul, |t, s| t.affine(s, 0.0), |s, t| t.affine(s, 0.0));
impl_op!(Div, div, div, |t, s| t.affine(1.0 / s, 0.0), |s, t| t.reciprocal().affine(s, 0.0));

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.unary(UnaryOp::Neg)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.unary(UnaryOp::Neg)
    }
}

impl std::ops::AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        *self = &*self + rhs;
    }
}
