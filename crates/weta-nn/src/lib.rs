//! # weta-nn
//!
//! Layers, losses and initialisers, with PyTorch's names. A model is a
//! struct of layers with a `forward` method and `#[derive(Module)]`:
//!
//! ```ignore
//! #[derive(Module)]
//! struct Net { fc1: nn::Linear, fc2: nn::Linear }
//!
//! impl Net {
//!     fn forward(&self, x: &Tensor) -> Tensor {
//!         self.fc2.forward(&F::relu(&self.fc1.forward(x)))
//!     }
//! }
//! ```
//!
//! The derive finds parameters in the fields, so `named_parameters()` gives
//! `fc1.weight`, `fc1.bias`, ... exactly as PyTorch would.

pub mod functional;
pub mod init;
mod layers;
mod loss;
mod module;

pub use layers::*;
pub use loss::{BCEWithLogitsLoss, CrossEntropyLoss, MSELoss};
pub use module::{join_name, Module, Parameter, TensorKind, Visit};
pub use weta_autograd::{Reduction, Tensor};
pub use weta_derive::Module;
pub use weta_io::StateDict;

/// Utilities, as `torch.nn.utils`.
pub mod utils {
    use weta_autograd::{no_grad, Tensor};

    /// Rescales gradients so their combined L2 norm is at most `max_norm`;
    /// returns the norm before clipping (`clip_grad_norm_`).
    pub fn clip_grad_norm_(parameters: &[Tensor], max_norm: f64) -> f64 {
        let _g = no_grad();
        let total: f64 = parameters.iter().filter_map(|p| p.grad()).map(|g| (&g * &g).sum().item()).sum::<f64>().sqrt();
        let scale = max_norm / (total + 1e-6);
        if scale < 1.0 {
            // Replace rather than scale in place: gradients can share storage
            // (both inputs of `a + b` get the same one), and an in-place
            // scale would hit a shared gradient twice.
            for p in parameters {
                if let Some(g) = p.grad() {
                    p.set_grad(Some(&g * scale));
                }
            }
        }
        total
    }

    /// Saves a model for `weta-infer`: its state dict as safetensors, with
    /// the architecture (from [`Module::describe`](crate::Module::describe))
    /// in the file's metadata, so one file is the whole deployable model.
    pub fn export(model: &impl crate::Module, path: impl AsRef<std::path::Path>) -> weta_core::Result<()> {
        let arch = model.describe().ok_or_else(|| {
            weta_core::Error::Msg("export: this model can't be described for weta-infer; use nn::Sequential of built-in layers".into())
        })?;
        let mut sd = model.state_dict();
        sd.metadata.insert("weta.arch".into(), arch);
        sd.save(path)
    }
}
