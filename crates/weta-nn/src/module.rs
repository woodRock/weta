use weta_autograd::{no_grad, Tensor};
use weta_core::{DType, Device, Error, Result};
use weta_io::StateDict;

/// A trainable parameter is an ordinary tensor that requires grad.
pub type Parameter = Tensor;

/// Whether a module's tensor is optimised (a parameter) or only state
/// (a buffer, like BatchNorm's running statistics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorKind {
    Parameter,
    Buffer,
}

/// `"fc1"` + `"weight"` → `"fc1.weight"`.
pub fn join_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// A layer or model, as PyTorch's `nn.Module`. You write `forward` (usually
/// via `#[derive(Module)]`, which also writes `visit`, `visit_mut` and
/// `set_training`); everything else is provided.
pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Calls `f` with the full name and kind of every tensor in this module.
    fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind));

    /// As [`visit`](Module::visit), but lets `f` replace tensors (for `to`).
    fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind));

    /// Switches train/eval behaviour (dropout, batch norm) here and in submodules.
    fn set_training(&mut self, _training: bool) {}

    /// A one-line description of this layer for `weta-infer` (e.g.
    /// `conv2d stride=1 padding=1`), or `None` if it has no inference
    /// equivalent. Used by [`export`](crate::utils::export).
    fn describe(&self) -> Option<String> {
        None
    }

    fn named_parameters(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        self.visit("", &mut |n, t, k| {
            if k == TensorKind::Parameter {
                out.push((n.to_string(), t.clone()));
            }
        });
        out
    }

    /// Handles to every parameter, for an optimiser.
    fn parameters(&self) -> Vec<Parameter> {
        self.named_parameters().into_iter().map(|(_, p)| p).collect()
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        self.visit("", &mut |n, t, k| {
            if k == TensorKind::Buffer {
                out.push((n.to_string(), t.clone()));
            }
        });
        out
    }

    /// Total number of parameter elements.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Parameters and buffers by name, sharing storage with the module.
    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        self.visit("", &mut |n, t, _| sd.insert(n, t.data().clone()));
        sd
    }

    /// Copies values from `sd` into this module's tensors in place, so
    /// optimisers holding the parameters keep working. With `strict`, every
    /// key must match exactly, as in PyTorch.
    fn load_state_dict(&mut self, sd: &StateDict, strict: bool) -> Result<()> {
        let _g = no_grad();
        let mut missing = Vec::new();
        let mut errors = Vec::new();
        let mut seen = Vec::new();
        self.visit("", &mut |n, t, _| match sd.get(n) {
            None => missing.push(n.to_string()),
            Some(src) if src.shape() != t.shape() => errors.push(format!(
                "size mismatch for {n}: copying a param with shape {:?}, the shape in the current model is {:?}",
                src.shape(),
                t.shape()
            )),
            Some(src) => {
                t.data().assign(src);
                seen.push(n.to_string());
            }
        });
        if strict {
            if !missing.is_empty() {
                errors.push(format!("missing keys: {missing:?}"));
            }
            let unexpected: Vec<&str> = sd.keys().filter(|k| !seen.iter().any(|s| s == k)).collect();
            if !unexpected.is_empty() {
                errors.push(format!("unexpected keys: {unexpected:?}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Msg(format!("error(s) in load_state_dict: {}", errors.join("; "))))
        }
    }

    fn train(&mut self) {
        self.set_training(true);
    }

    fn eval(&mut self) {
        self.set_training(false);
    }

    /// Clears every parameter's gradient.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    /// Moves every parameter and buffer to `device` (call before creating
    /// the optimiser, as in PyTorch).
    fn to(mut self, device: Device) -> Self
    where
        Self: Sized,
    {
        self.visit_mut("", &mut |_, t, _| *t = t.to(device));
        self
    }

    /// Converts floating-point parameters and buffers to `dtype`.
    fn to_dtype(mut self, dtype: DType) -> Self
    where
        Self: Sized,
    {
        self.visit_mut("", &mut |_, t, _| {
            if t.dtype().is_float() {
                *t = Tensor::from_raw(t.data().to_dtype(dtype)).requires_grad_(t.requires_grad());
            }
        });
        self
    }
}

impl<M: Module + ?Sized> Module for Box<M> {
    fn forward(&self, x: &Tensor) -> Tensor {
        (**self).forward(x)
    }
    fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        (**self).visit(prefix, f)
    }
    fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        (**self).visit_mut(prefix, f)
    }
    fn set_training(&mut self, training: bool) {
        (**self).set_training(training)
    }
    fn describe(&self) -> Option<String> {
        (**self).describe()
    }
}

/// What `#[derive(Module)]` calls on each field: implemented for every
/// module, for a bare `Tensor` (a parameter), and for `Option`s and `Vec`s
/// of them.
pub trait Visit {
    fn visit_tensors(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind));
    fn visit_tensors_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind));
    fn set_train_mode(&mut self, _training: bool) {}
}

impl<M: Module + ?Sized> Visit for M {
    fn visit_tensors(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        self.visit(prefix, f)
    }
    fn visit_tensors_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        self.visit_mut(prefix, f)
    }
    fn set_train_mode(&mut self, training: bool) {
        self.set_training(training)
    }
}

impl Visit for Tensor {
    fn visit_tensors(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f(prefix, self, TensorKind::Parameter)
    }
    fn visit_tensors_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        f(prefix, self, TensorKind::Parameter)
    }
}

impl<T: Visit> Visit for Option<T> {
    fn visit_tensors(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        if let Some(t) = self {
            t.visit_tensors(prefix, f)
        }
    }
    fn visit_tensors_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        if let Some(t) = self {
            t.visit_tensors_mut(prefix, f)
        }
    }
    fn set_train_mode(&mut self, training: bool) {
        if let Some(t) = self {
            t.set_train_mode(training)
        }
    }
}

impl<T: Visit> Visit for Vec<T> {
    fn visit_tensors(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        for (i, t) in self.iter().enumerate() {
            t.visit_tensors(&join_name(prefix, &i.to_string()), f)
        }
    }
    fn visit_tensors_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        for (i, t) in self.iter_mut().enumerate() {
            t.visit_tensors_mut(&join_name(prefix, &i.to_string()), f)
        }
    }
    fn set_train_mode(&mut self, training: bool) {
        for t in self {
            t.set_train_mode(training)
        }
    }
}
