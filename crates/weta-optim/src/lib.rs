//! # weta-optim
//!
//! Optimisers with PyTorch's names, defaults and update rules, used in the
//! usual loop:
//!
//! ```ignore
//! let mut opt = optim::Adam::new(model.parameters(), 1e-3);
//! opt.zero_grad();
//! loss.backward();
//! opt.step();
//! ```
//!
//! `step` updates parameters in place on their raw data, so nothing is
//! recorded on the graph and the model's handles see the new values.
//! State uses PyTorch's names (`exp_avg`, `exp_avg_sq`, `momentum_buffer`)
//! and round-trips through [`Optimizer::state_dict`].

use weta_autograd::Tensor;
use weta_core::{Error, Result};
use weta_io::StateDict;
use weta_tensor::Tensor as Raw;

pub mod lr_scheduler;

/// What every optimiser provides.
pub trait Optimizer {
    /// One update of every parameter that has a gradient.
    fn step(&mut self);
    /// Clears the gradients of the optimised parameters.
    fn zero_grad(&mut self);
    fn lr(&self) -> f64;
    fn set_lr(&mut self, lr: f64);
    fn state_dict(&self) -> StateDict;
    fn load_state_dict(&mut self, sd: &StateDict) -> Result<()>;
}

fn zero_grads(params: &[Tensor]) {
    for p in params {
        p.zero_grad();
    }
}

/// The gradient as a raw tensor, or `None` if backward didn't reach `p`.
fn grad_of(p: &Tensor) -> Option<Raw> {
    p.grad().map(|g| g.data().clone())
}

fn save_state(sd: &mut StateDict, i: usize, name: &str, t: &Option<Raw>) {
    if let Some(t) = t {
        sd.insert(format!("state.{i}.{name}"), t.clone());
    }
}

fn load_state(sd: &StateDict, i: usize, name: &str, like: &Tensor) -> Option<Raw> {
    sd.get(&format!("state.{i}.{name}")).map(|t| t.to(like.device()).to_dtype(like.dtype()).copy())
}

fn lr_from(sd: &StateDict) -> Result<f64> {
    sd.metadata
        .get("lr")
        .ok_or_else(|| Error::Msg("optimizer state dict has no lr".into()))?
        .parse()
        .map_err(|_| Error::Msg("optimizer state dict has an invalid lr".into()))
}

// ---------------------------------------------------------------------------

/// Stochastic gradient descent, with optional momentum (heavy-ball or
/// Nesterov) and L2 weight decay:
///
/// ```text
/// g ← ∇ + λθ;   b ← μ b + (1 − τ) g;   θ ← θ − η (nesterov ? g + μ b : b)
/// ```
pub struct SGD {
    params: Vec<Tensor>,
    lr: f64,
    momentum: f64,
    dampening: f64,
    weight_decay: f64,
    nesterov: bool,
    momentum_buffer: Vec<Option<Raw>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        SGD { params, lr, momentum: 0.0, dampening: 0.0, weight_decay: 0.0, nesterov: false, momentum_buffer: vec![None; n] }
    }
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
    pub fn dampening(mut self, d: f64) -> Self {
        self.dampening = d;
        self
    }
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
    pub fn nesterov(mut self, on: bool) -> Self {
        self.nesterov = on;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        let _g = weta_autograd::no_grad();
        for (p, buf) in self.params.iter().zip(&mut self.momentum_buffer) {
            let Some(mut g) = grad_of(p) else { continue };
            if self.weight_decay != 0.0 {
                g = g.add(&p.data().mul_scalar(self.weight_decay));
            }
            if self.momentum != 0.0 {
                let b = match buf.take() {
                    None => g.copy(),
                    Some(b) => b.mul_scalar(self.momentum).add(&g.mul_scalar(1.0 - self.dampening)),
                };
                g = if self.nesterov { g.add(&b.mul_scalar(self.momentum)) } else { b.clone() };
                *buf = Some(b);
            }
            p.data().sub_(&g.mul_scalar(self.lr));
        }
    }
    fn zero_grad(&mut self) {
        zero_grads(&self.params)
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for (i, b) in self.momentum_buffer.iter().enumerate() {
            save_state(&mut sd, i, "momentum_buffer", b);
        }
        sd.metadata.insert("lr".into(), self.lr.to_string());
        sd
    }
    fn load_state_dict(&mut self, sd: &StateDict) -> Result<()> {
        self.lr = lr_from(sd)?;
        for (i, p) in self.params.iter().enumerate() {
            self.momentum_buffer[i] = load_state(sd, i, "momentum_buffer", p);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------

/// Adam (Kingma & Ba, 2015): per-parameter step sizes from running
/// estimates of the gradient's first and second moments, bias-corrected
/// for their zero initialisation:
///
/// ```text
/// m ← β₁m + (1 − β₁)g;   v ← β₂v + (1 − β₂)g²
/// θ ← θ − η · (m / (1 − β₁ᵗ)) / (√(v / (1 − β₂ᵗ)) + ε)
/// ```
///
/// `weight_decay` adds `λθ` to the gradient (L2, as `torch.optim.Adam`);
/// [`AdamW`] decouples it instead.
pub struct Adam {
    params: Vec<Tensor>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    decoupled: bool,
    step: Vec<u64>,
    exp_avg: Vec<Option<Raw>>,
    exp_avg_sq: Vec<Option<Raw>>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let n = params.len();
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled: false,
            step: vec![0; n],
            exp_avg: vec![None; n],
            exp_avg_sq: vec![None; n],
        }
    }
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        let _g = weta_autograd::no_grad();
        for (i, p) in self.params.iter().enumerate() {
            let Some(mut g) = grad_of(p) else { continue };
            let theta = p.data();
            if self.weight_decay != 0.0 {
                if self.decoupled {
                    theta.mul_scalar_(1.0 - self.lr * self.weight_decay);
                } else {
                    g = g.add(&theta.mul_scalar(self.weight_decay));
                }
            }
            self.step[i] += 1;
            let t = self.step[i] as i32;
            let (b1, b2) = (self.beta1, self.beta2);
            let m = match self.exp_avg[i].take() {
                Some(m) => m.mul_scalar(b1).add(&g.mul_scalar(1.0 - b1)),
                None => g.mul_scalar(1.0 - b1),
            };
            let v = match self.exp_avg_sq[i].take() {
                Some(v) => v.mul_scalar(b2).add(&g.mul(&g).mul_scalar(1.0 - b2)),
                None => g.mul(&g).mul_scalar(1.0 - b2),
            };
            let bc1 = 1.0 - b1.powi(t);
            let bc2 = 1.0 - b2.powi(t);
            let denom = v.sqrt().mul_scalar(1.0 / bc2.sqrt()).add_scalar(self.eps);
            theta.sub_(&m.div(&denom).mul_scalar(self.lr / bc1));
            self.exp_avg[i] = Some(m);
            self.exp_avg_sq[i] = Some(v);
        }
    }
    fn zero_grad(&mut self) {
        zero_grads(&self.params)
    }
    fn lr(&self) -> f64 {
        self.lr
    }
    fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for i in 0..self.params.len() {
            if self.step[i] > 0 {
                sd.insert(format!("state.{i}.step"), Raw::from_vec(vec![self.step[i] as i64], &[]));
            }
            save_state(&mut sd, i, "exp_avg", &self.exp_avg[i]);
            save_state(&mut sd, i, "exp_avg_sq", &self.exp_avg_sq[i]);
        }
        sd.metadata.insert("lr".into(), self.lr.to_string());
        sd
    }
    fn load_state_dict(&mut self, sd: &StateDict) -> Result<()> {
        self.lr = lr_from(sd)?;
        for (i, p) in self.params.iter().enumerate() {
            self.step[i] = sd.get(&format!("state.{i}.step")).map_or(0, |t| t.item() as u64);
            self.exp_avg[i] = load_state(sd, i, "exp_avg", p);
            self.exp_avg_sq[i] = load_state(sd, i, "exp_avg_sq", p);
        }
        Ok(())
    }
}

/// AdamW (Loshchilov & Hutter, 2019): Adam with decoupled weight decay,
/// `θ ← θ(1 − ηλ)` before the Adam step, instead of adding `λθ` to the
/// gradient. The default `weight_decay` is 0.01, as in PyTorch.
pub struct AdamW;

impl AdamW {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(params: Vec<Tensor>, lr: f64) -> Adam {
        Adam { decoupled: true, weight_decay: 0.01, ..Adam::new(params, lr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimises (θ − 3)² and checks θ → 3.
    fn converges(mut opt: impl Optimizer, p: &Tensor, steps: usize) -> f64 {
        for _ in 0..steps {
            opt.zero_grad();
            let loss = ((p - 3.0) * (p - 3.0)).sum();
            loss.backward();
            opt.step();
        }
        p.item()
    }

    #[test]
    fn optimisers_converge() {
        let p = Tensor::zeros(&[1]).requires_grad_(true);
        assert!((converges(SGD::new(vec![p.clone()], 0.1), &p, 100) - 3.0).abs() < 1e-3);
        let p = Tensor::zeros(&[1]).requires_grad_(true);
        assert!((converges(SGD::new(vec![p.clone()], 0.05).momentum(0.9).nesterov(true), &p, 200) - 3.0).abs() < 1e-3);
        let p = Tensor::zeros(&[1]).requires_grad_(true);
        assert!((converges(Adam::new(vec![p.clone()], 0.1), &p, 300) - 3.0).abs() < 1e-2);
        let p = Tensor::zeros(&[1]).requires_grad_(true);
        assert!((converges(AdamW::new(vec![p.clone()], 0.1).weight_decay(0.0), &p, 300) - 3.0).abs() < 1e-2);
    }

    #[test]
    fn adam_first_step_matches_pytorch() {
        // With bias correction, Adam's first step is lr · sign(g) (up to eps).
        let p = Tensor::from_vec(vec![1.0f32, -2.0], &[2]).requires_grad_(true);
        let mut opt = Adam::new(vec![p.clone()], 0.01);
        (&p * &Tensor::from_vec(vec![3.0f32, -0.5], &[2])).sum().backward();
        opt.step();
        let v = p.to_vec::<f32>();
        assert!((v[0] - 0.99).abs() < 1e-6 && (v[1] - -1.99).abs() < 1e-6, "{v:?}");
    }

    #[test]
    fn state_dict_round_trips() {
        let p = Tensor::from_vec(vec![1.0f32], &[1]).requires_grad_(true);
        let mut opt = Adam::new(vec![p.clone()], 0.01);
        (&p * 2.0).sum().backward();
        opt.step();
        let sd = StateDict::from_bytes(&opt.state_dict().to_bytes()).unwrap();
        let mut opt2 = Adam::new(vec![p.clone()], 1.0);
        opt2.load_state_dict(&sd).unwrap();
        assert_eq!(opt2.lr(), 0.01);
        assert_eq!(opt2.step[0], 1);
        assert_eq!(opt2.exp_avg[0].as_ref().unwrap().to_vec::<f32>(), opt.exp_avg[0].as_ref().unwrap().to_vec::<f32>());
    }
}
