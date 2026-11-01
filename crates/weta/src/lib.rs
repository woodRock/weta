//! # weta
//!
//! A small, readable deep learning framework in Rust with a PyTorch-style
//! API: N-dimensional tensors, reverse-mode autodiff, the standard layers
//! and optimisers, a CPU backend that runs anywhere, and CUDA behind the
//! `cuda` feature.
//!
//! ```no_run
//! use weta::prelude::*;
//! use weta::nn::functional as F;
//!
//! #[derive(Module)]
//! struct Net {
//!     fc1: nn::Linear,
//!     fc2: nn::Linear,
//!     drop: nn::Dropout,
//! }
//!
//! impl Net {
//!     fn new() -> Self {
//!         Net { fc1: nn::Linear::new(784, 128), fc2: nn::Linear::new(128, 10), drop: nn::Dropout::new(0.1) }
//!     }
//!
//!     fn forward(&self, x: &Tensor) -> Tensor {
//!         let x = F::relu(&self.fc1.forward(&x.flatten(1, -1)));
//!         self.fc2.forward(&self.drop.forward(&x)) // logits
//!     }
//! }
//!
//! fn main() -> weta::Result<()> {
//!     let device = Device::cuda_if_available();
//!     let mut model = Net::new().to(device);
//!     let mut opt = optim::Adam::new(model.parameters(), 1e-3);
//!     let train = data::DataLoader::new(data::datasets::Mnist::train("data")?, 64).shuffle(true);
//!
//!     for epoch in 0..5 {
//!         model.train();
//!         let mut last = 0.0;
//!         for (x, y) in &train {
//!             let (x, y) = (x.to(device), y.to(device));
//!             let loss = F::cross_entropy(&model.forward(&x), &y);
//!             opt.zero_grad();
//!             loss.backward();
//!             opt.step();
//!             last = loss.item();
//!         }
//!         println!("epoch {epoch}: last loss {last:.4}");
//!     }
//!     model.state_dict().save("mnist.safetensors")?;
//!     Ok(())
//! }
//! ```
//!
//! The crates underneath, from the bottom up:
//!
//! | crate | what it holds |
//! | --- | --- |
//! | `weta-core` | `DType`, `Device`, `Layout`, the `Backend` trait |
//! | `weta-cpu`, `weta-cuda` | backends: a dozen primitive kernels each |
//! | `weta-tensor` | strided N-d tensors with no autograd |
//! | `weta-autograd` | the autograd `Tensor`, the graph, `backward`, `gradcheck` |
//! | `weta-nn`, `weta-optim`, `weta-data` | layers, optimisers, datasets |
//! | `weta-io`, `weta-infer` | safetensors, and inference without autograd |

pub use weta_core::{DType, Device, Error, Result};
pub use weta_tensor::{cuda_device_count, cuda_is_available, cuda_profile_report, manual_seed};

#[cfg(feature = "train")]
pub use weta_autograd::{
    enable_grad, gradcheck, is_grad_enabled, no_grad, set_grad_enabled, tensor, BackwardOptions, GradcheckOptions, IndexOp, Tensor,
};

/// Without the `train` feature, `weta::Tensor` is the plain array type.
#[cfg(not(feature = "train"))]
pub use weta_tensor::Tensor;

/// The autograd internals, for writing new differentiable ops.
#[cfg(feature = "train")]
pub mod autograd {
    pub use weta_autograd::*;
}

/// Plain tensors without autograd (the layer `weta-infer` builds on).
pub mod raw {
    pub use weta_tensor::*;
}

/// Layers, losses, init schemes (`torch.nn`); `nn::functional` is `F`.
#[cfg(feature = "train")]
pub mod nn {
    pub use weta_nn::*;
}

/// Optimisers and learning-rate schedulers (`torch.optim`).
#[cfg(feature = "train")]
pub mod optim {
    pub use weta_optim::*;
}

/// Datasets and loaders (`torch.utils.data`, `torchvision.datasets`).
#[cfg(feature = "train")]
pub mod data {
    pub use weta_data::*;
}

/// safetensors state dicts.
pub mod io {
    pub use weta_io::*;
}

/// Inference without autograd.
pub mod infer {
    pub use weta_infer::*;
}

/// `Device::cuda_if_available()`, as PyTorch's
/// `"cuda" if torch.cuda.is_available() else "cpu"`.
pub trait DeviceExt {
    fn cuda_if_available() -> Device;
}

impl DeviceExt for Device {
    fn cuda_if_available() -> Device {
        if cuda_is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }
}

/// `use weta::prelude::*;` brings in `Tensor`, `Device`, `DType`, `nn`,
/// `optim`, `data`, the `Module` trait and derive, and `no_grad`.
pub mod prelude {
    pub use crate::{manual_seed, DType, Device, DeviceExt, Tensor};

    #[cfg(feature = "train")]
    pub use crate::{data, nn, no_grad, optim, tensor, IndexOp};
    #[cfg(feature = "train")]
    pub use weta_nn::Module;
    #[cfg(feature = "train")]
    pub use weta_optim::Optimizer;
}
