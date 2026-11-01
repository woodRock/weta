//! Initialisation schemes, as `torch.nn.init`. Each fills a tensor in place
//! (hence the trailing `_`) without recording on the graph.
//!
//! The scale of the initial weights matters: if each layer multiplies the
//! variance of its input by more (or less) than one, activations explode (or
//! vanish) with depth. For a layer with `fan_in` inputs:
//!
//! - **Xavier/Glorot** keeps variance steady for tanh-like units:
//!   `Var(w) = 2 / (fan_in + fan_out)`.
//! - **Kaiming/He** compensates for ReLU zeroing half its inputs:
//!   `Var(w) = 2 / fan_in`.

use weta_autograd::{no_grad, Tensor};
use weta_tensor::Tensor as Raw;

/// Which fan to scale by: `FanIn` preserves the forward pass's variance,
/// `FanOut` the backward pass's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FanMode {
    #[default]
    FanIn,
    FanOut,
}

/// The nonlinearity after the layer, for [`calculate_gain`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nonlinearity {
    Linear,
    Sigmoid,
    Tanh,
    Relu,
    LeakyRelu(f64),
}

/// The recommended gain for a nonlinearity (`torch.nn.init.calculate_gain`).
pub fn calculate_gain(nl: Nonlinearity) -> f64 {
    match nl {
        Nonlinearity::Linear | Nonlinearity::Sigmoid => 1.0,
        Nonlinearity::Tanh => 5.0 / 3.0,
        Nonlinearity::Relu => 2f64.sqrt(),
        Nonlinearity::LeakyRelu(a) => (2.0 / (1.0 + a * a)).sqrt(),
    }
}

/// `(fan_in, fan_out)` of a weight: for `[out, in, k...]`, `fan_in = in·Πk`.
#[track_caller]
pub fn fans(t: &Tensor) -> (usize, usize) {
    assert!(t.ndim() >= 2, "fan in and fan out can't be computed for a tensor with fewer than 2 dims ({:?})", t.shape());
    let receptive: usize = t.shape()[2..].iter().product();
    (t.size(1) * receptive, t.size(0) * receptive)
}

fn fill(t: &Tensor, src: Raw) {
    let _g = no_grad();
    t.data().assign(&src);
}

pub fn uniform_(t: &Tensor, a: f64, b: f64) {
    fill(t, Raw::uniform(t.shape(), a, b, t.dtype(), t.device()));
}

pub fn normal_(t: &Tensor, mean: f64, std: f64) {
    fill(t, Raw::normal(t.shape(), mean, std, t.dtype(), t.device()));
}

pub fn constant_(t: &Tensor, v: f64) {
    let _g = no_grad();
    t.data().fill_(v);
}

pub fn zeros_(t: &Tensor) {
    constant_(t, 0.0)
}

pub fn ones_(t: &Tensor) {
    constant_(t, 1.0)
}

/// `U(−a, a)` with `a = gain · √(6 / (fan_in + fan_out))`.
pub fn xavier_uniform_(t: &Tensor, gain: f64) {
    let (fi, fo) = fans(t);
    let a = gain * (6.0 / (fi + fo) as f64).sqrt();
    uniform_(t, -a, a);
}

/// `N(0, σ²)` with `σ = gain · √(2 / (fan_in + fan_out))`.
pub fn xavier_normal_(t: &Tensor, gain: f64) {
    let (fi, fo) = fans(t);
    normal_(t, 0.0, gain * (2.0 / (fi + fo) as f64).sqrt());
}

fn fan(t: &Tensor, mode: FanMode) -> f64 {
    let (fi, fo) = fans(t);
    (if mode == FanMode::FanIn { fi } else { fo }) as f64
}

/// `U(−b, b)` with `b = gain · √(3 / fan)`.
pub fn kaiming_uniform_(t: &Tensor, mode: FanMode, nl: Nonlinearity) {
    let b = calculate_gain(nl) * (3.0 / fan(t, mode)).sqrt();
    uniform_(t, -b, b);
}

/// `N(0, σ²)` with `σ = gain / √fan`.
pub fn kaiming_normal_(t: &Tensor, mode: FanMode, nl: Nonlinearity) {
    normal_(t, 0.0, calculate_gain(nl) / fan(t, mode).sqrt());
}
