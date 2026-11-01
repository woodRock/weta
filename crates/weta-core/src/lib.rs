//! # weta-core
//!
//! The vocabulary every other weta crate shares: [`DType`], [`Device`],
//! [`Layout`] (shape, strides and offset), the primitive op enums, and the
//! [`Backend`] trait that CPU and GPU backends implement.
//!
//! Nothing here allocates tensor data or knows about gradients. It is the
//! bottom of the dependency graph:
//!
//! ```text
//! weta-core ← weta-cpu / weta-cuda ← weta-tensor ← weta-autograd ← weta-nn, weta-optim, weta-data
//!                                        ↑
//!                                 weta-io ← weta-infer
//! ```

mod backend;
mod dtype;
mod error;
mod layout;
mod ops;

pub use backend::{Backend, HostData, View};
pub use dtype::{DType, Element, Float};
pub use error::{Error, Result};
pub use layout::{broadcast_shapes, resolve_shape, Layout, StridedOffsets};
pub use ops::{splitmix64, BinaryOp, Distribution, Im2ColParams, ReduceOp, UnaryOp};

use std::fmt;

/// Where a tensor's data lives. Like PyTorch's `torch.device`, it is a
/// runtime value, so the same model code runs anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// A CUDA GPU by ordinal. Needs the `cuda` Cargo feature.
    Cuda(usize),
}

impl Device {
    pub fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }
    pub fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda(_))
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}

/// Formats a shape the way error messages print it: `[64, 784]`.
pub fn shape_str(shape: &[usize]) -> String {
    format!("{shape:?}")
}
