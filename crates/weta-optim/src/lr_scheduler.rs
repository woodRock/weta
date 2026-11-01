//! Learning-rate schedules, as `torch.optim.lr_scheduler`. Call `step()`
//! once per epoch, after the optimiser's steps.

use crate::Optimizer;

/// Multiplies the learning rate by `gamma` every `step_size` epochs.
pub struct StepLR {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
    epoch: usize,
}

impl StepLR {
    pub fn new(opt: &impl Optimizer, step_size: usize, gamma: f64) -> Self {
        StepLR { base_lr: opt.lr(), step_size, gamma, epoch: 0 }
    }

    pub fn step(&mut self, opt: &mut impl Optimizer) {
        self.epoch += 1;
        opt.set_lr(self.base_lr * self.gamma.powi((self.epoch / self.step_size) as i32));
    }
}

/// Anneals the learning rate from its initial value to `eta_min` along half
/// a cosine over `t_max` epochs:
/// `η_t = η_min + ½(η₀ − η_min)(1 + cos(π t / T))`.
pub struct CosineAnnealingLR {
    base_lr: f64,
    t_max: usize,
    eta_min: f64,
    epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(opt: &impl Optimizer, t_max: usize, eta_min: f64) -> Self {
        CosineAnnealingLR { base_lr: opt.lr(), t_max, eta_min, epoch: 0 }
    }

    pub fn step(&mut self, opt: &mut impl Optimizer) {
        self.epoch += 1;
        let t = (self.epoch.min(self.t_max)) as f64 / self.t_max as f64;
        opt.set_lr(self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * t).cos()));
    }
}
