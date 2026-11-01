//! Loss modules, as in PyTorch (`nn.CrossEntropyLoss()(logits, target)`).
//! They take two inputs, so they have their own `forward` rather than the
//! single-input [`Module`](crate::Module) one.

use weta_autograd::{Reduction, Tensor};

use crate::functional as F;

#[derive(Clone, Copy, Debug, Default)]
pub struct CrossEntropyLoss {
    pub reduction: Reduction,
    pub label_smoothing: f64,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn reduction(mut self, r: Reduction) -> Self {
        self.reduction = r;
        self
    }
    pub fn label_smoothing(mut self, s: f64) -> Self {
        self.label_smoothing = s;
        self
    }
    #[track_caller]
    pub fn forward(&self, logits: &Tensor, target: &Tensor) -> Tensor {
        F::cross_entropy_with(logits, target, self.reduction, self.label_smoothing)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MSELoss {
    pub reduction: Reduction,
}

impl MSELoss {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        F::mse_loss_with(input, target, self.reduction)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BCEWithLogitsLoss {
    pub reduction: Reduction,
}

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn forward(&self, logits: &Tensor, target: &Tensor) -> Tensor {
        F::binary_cross_entropy_with_logits_with(logits, target, self.reduction)
    }
}
