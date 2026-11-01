//! Stateless functions, as `torch.nn.functional` (import it as `F`).
//!
//! Most are a few lines of differentiable tensor ops, so their backward
//! passes come for free; the exceptions (softmax, log-softmax and
//! cross-entropy) are fused ops in `weta-autograd`, because their Jacobians
//! simplify.

use weta_autograd::{Reduction, Tensor};
use weta_core::Im2ColParams;

pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

pub fn sigmoid(x: &Tensor) -> Tensor {
    x.sigmoid()
}

pub fn tanh(x: &Tensor) -> Tensor {
    x.tanh()
}

/// `max(0, x) + slope · min(0, x)`.
pub fn leaky_relu(x: &Tensor, negative_slope: f64) -> Tensor {
    x.relu() - (-x).relu() * negative_slope
}

/// `x · σ(x)`, also called swish.
pub fn silu(x: &Tensor) -> Tensor {
    x * &x.sigmoid()
}

/// GELU, exact form: `x · Φ(x) = ½ x (1 + erf(x / √2))`.
pub fn gelu(x: &Tensor) -> Tensor {
    x * &((x * std::f64::consts::FRAC_1_SQRT_2).erf() + 1.0) * 0.5
}

/// GELU, tanh approximation (`approximate='tanh'` in PyTorch).
pub fn gelu_tanh(x: &Tensor) -> Tensor {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let inner = (x + &(x * x * x * 0.044715)) * c;
    x * &(inner.tanh() + 1.0) * 0.5
}

pub fn softmax(x: &Tensor, dim: isize) -> Tensor {
    x.softmax(dim)
}

pub fn log_softmax(x: &Tensor, dim: isize) -> Tensor {
    x.log_softmax(dim)
}

/// `x Wᵀ + b`, with `W` stored `[out, in]` as in PyTorch.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let y = x.matmul(&weight.t());
    match bias {
        Some(b) => y + b,
        None => y,
    }
}

/// Zeroes each element with probability `p` and scales the rest by
/// `1 / (1 − p)`, so the expected value is unchanged. Identity in eval mode.
pub fn dropout(x: &Tensor, p: f64, training: bool) -> Tensor {
    assert!((0.0..=1.0).contains(&p), "dropout probability has to be between 0 and 1, but got {p}");
    if !training || p == 0.0 {
        return x.clone();
    }
    if p == 1.0 {
        return x * 0.0;
    }
    let mask = weta_tensor::Tensor::rand(x.shape(), x.dtype(), x.device()).gt_scalar(p).mul_scalar(1.0 / (1.0 - p));
    x * &Tensor::from_raw(mask)
}

/// 2-D convolution as `im2col` + one matrix multiply:
/// `x [N, C, H, W]`, `weight [O, C, kh, kw]`, `bias [O]` → `[N, O, OH, OW]`.
#[track_caller]
pub fn conv2d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> Tensor {
    conv2d_ext(x, weight, bias, (stride, stride), (padding, padding), (1, 1))
}

#[track_caller]
pub fn conv2d_ext(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor {
    assert_eq!(x.ndim(), 4, "conv2d: expected input [N, C, H, W], got {:?}", x.shape());
    assert_eq!(weight.ndim(), 4, "conv2d: expected weight [O, C, kh, kw], got {:?}", weight.shape());
    let (n, c, h, w) = (x.size(0), x.size(1), x.size(2), x.size(3));
    let (o, wc, kh, kw) = (weight.size(0), weight.size(1), weight.size(2), weight.size(3));
    assert_eq!(c, wc, "conv2d: input has {c} channels but weight {:?} expects {wc}", weight.shape());
    let p = Im2ColParams { kernel: (kh, kw), stride, padding, dilation };
    let (oh, ow) = p.out_size(h, w);
    assert!(oh > 0 && ow > 0, "conv2d: kernel {kh}×{kw} is larger than the padded input {h}×{w}");
    let cols = x.im2col(p); // [N, C·kh·kw, OH·OW]
    let mut y = weight.reshape(&[o as isize, -1]).matmul(&cols); // [N, O, OH·OW]
    if let Some(b) = bias {
        y = y + b.reshape(&[o as isize, 1]);
    }
    y.reshape_to(&[n, o, oh, ow])
}

/// Max over each `kernel × kernel` window. The gradient goes to the max.
#[track_caller]
pub fn max_pool2d(x: &Tensor, kernel: usize, stride: usize) -> Tensor {
    pool2d(x, kernel, stride, 0, true)
}

/// Mean over each `kernel × kernel` window (zero padding counts, as in PyTorch).
#[track_caller]
pub fn avg_pool2d(x: &Tensor, kernel: usize, stride: usize, padding: usize) -> Tensor {
    pool2d(x, kernel, stride, padding, false)
}

#[track_caller]
fn pool2d(x: &Tensor, kernel: usize, stride: usize, padding: usize, max: bool) -> Tensor {
    assert_eq!(x.ndim(), 4, "pool2d: expected [N, C, H, W], got {:?}", x.shape());
    let (n, c, h, w) = (x.size(0), x.size(1), x.size(2), x.size(3));
    let p = Im2ColParams { kernel: (kernel, kernel), stride: (stride, stride), padding: (padding, padding), dilation: (1, 1) };
    let (oh, ow) = p.out_size(h, w);
    // Treat channels as batch so each window is one column.
    let cols = x.reshape_to(&[n * c, 1, h, w]).im2col(p); // [N·C, k·k, OH·OW]
    let y = if max { cols.max_dims(&[1], false) } else { cols.mean_dims(&[1], false) };
    y.reshape_to(&[n, c, oh, ow])
}

/// Average over the whole spatial extent: `[N, C, H, W]` → `[N, C, 1, 1]`.
pub fn adaptive_avg_pool2d_1(x: &Tensor) -> Tensor {
    x.mean_dims(&[2, 3], true)
}

/// Batch normalisation over every dim except 1 (channels).
///
/// Training: normalise with the batch mean and (biased) variance, and move
/// the running estimates toward them (`running = (1 − m)·running + m·batch`,
/// with the unbiased variance, as PyTorch does). Eval: use the running
/// estimates.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm(
    x: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> Tensor {
    let c = x.size(1);
    let dims: Vec<isize> = (0..x.ndim() as isize).filter(|&d| d != 1).collect();
    let mut bshape = vec![1usize; x.ndim()];
    bshape[1] = c;
    let (mean, var) = match (running_mean, running_var) {
        (Some(rm), Some(rv)) if !training => (rm.reshape_to(&bshape), rv.reshape_to(&bshape)),
        (rm, rv) => {
            let mean = x.mean_dims(&dims, true);
            let var = x.var_dims(&dims, false, true);
            if let (Some(rm), Some(rv)) = (rm, rv) {
                let n = x.numel() / c;
                let unbiased = var.data().mul_scalar(n as f64 / (n.max(2) - 1) as f64);
                rm.data().assign(&rm.data().mul_scalar(1.0 - momentum).add(&mean.data().reshape_to(&[c]).mul_scalar(momentum)));
                rv.data().assign(&rv.data().mul_scalar(1.0 - momentum).add(&unbiased.reshape_to(&[c]).mul_scalar(momentum)));
            }
            (mean, var)
        }
    };
    let mut y = (x - &mean) * (var + eps).sqrt().reciprocal();
    if let Some(w) = weight {
        y = y * w.reshape_to(&bshape);
    }
    if let Some(b) = bias {
        y = y + b.reshape_to(&bshape);
    }
    y
}

/// Layer normalisation over the last `normalized_dims` dims.
pub fn layer_norm(x: &Tensor, normalized_dims: usize, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f64) -> Tensor {
    let dims: Vec<isize> = (x.ndim() - normalized_dims..x.ndim()).map(|d| d as isize).collect();
    let mean = x.mean_dims(&dims, true);
    let var = x.var_dims(&dims, false, true);
    let mut y = (x - &mean) * (var + eps).sqrt().reciprocal();
    if let Some(w) = weight {
        y = y * w;
    }
    if let Some(b) = bias {
        y += b;
    }
    y
}

/// Cross-entropy on raw logits `[N, C]` and `I64` class indices `[N]`,
/// averaged over the batch (PyTorch's default).
#[track_caller]
pub fn cross_entropy(logits: &Tensor, target: &Tensor) -> Tensor {
    logits.cross_entropy(target, Reduction::Mean, 0.0)
}

/// Cross-entropy with a reduction and label smoothing.
#[track_caller]
pub fn cross_entropy_with(logits: &Tensor, target: &Tensor, reduction: Reduction, label_smoothing: f64) -> Tensor {
    logits.cross_entropy(target, reduction, label_smoothing)
}

/// Negative log-likelihood of `log_probs [N, C]` at class indices `[N]`.
pub fn nll_loss(log_probs: &Tensor, target: &Tensor) -> Tensor {
    let picked = (log_probs * &target.one_hot(log_probs.size(1)).to_dtype(log_probs.dtype())).sum_dims(&[1], false);
    -picked.mean()
}

fn reduce(x: Tensor, reduction: Reduction) -> Tensor {
    match reduction {
        Reduction::None => x,
        Reduction::Mean => x.mean(),
        Reduction::Sum => x.sum(),
    }
}

pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    mse_loss_with(input, target, Reduction::Mean)
}

pub fn mse_loss_with(input: &Tensor, target: &Tensor, reduction: Reduction) -> Tensor {
    let d = input - target;
    reduce(&d * &d, reduction)
}

pub fn l1_loss(input: &Tensor, target: &Tensor) -> Tensor {
    (input - target).abs().mean()
}

/// Binary cross-entropy on logits, in the stable form
/// `max(x, 0) − x·y + log(1 + e^{−|x|})`.
pub fn binary_cross_entropy_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    binary_cross_entropy_with_logits_with(logits, target, Reduction::Mean)
}

pub fn binary_cross_entropy_with_logits_with(logits: &Tensor, target: &Tensor, reduction: Reduction) -> Tensor {
    let loss = logits.relu() - logits * target + ((-logits.abs()).exp() + 1.0).log();
    reduce(loss, reduction)
}

pub fn flatten(x: &Tensor, start: isize, end: isize) -> Tensor {
    x.flatten(start, end)
}

pub fn one_hot(target: &Tensor, classes: usize) -> Tensor {
    target.one_hot(classes)
}
