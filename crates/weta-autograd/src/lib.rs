//! # weta-autograd
//!
//! Reverse-mode automatic differentiation. [`Tensor`] here is the
//! user-facing tensor: a `weta_tensor::Tensor` plus what autograd needs
//! (whether it requires grad, its `.grad()`, and the [`Node`] that made it).
//!
//! ```
//! use weta_autograd::Tensor;
//!
//! let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).requires_grad_(true);
//! let y = (&x * &x).sum(); // y = Σ x²
//! y.backward();
//! assert_eq!(x.grad().unwrap().to_vec::<f32>(), vec![2.0, 4.0, 6.0]); // dy/dx = 2x
//! ```

mod gradcheck;
mod graph;
mod index;
mod ops;
mod tensor_macro;

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use smallvec::SmallVec;
pub use weta_core::{DType, Device, Error, Result};
pub use weta_tensor::{cuda_device_count, cuda_is_available, manual_seed, randperm, with_rng, Rng};

pub use gradcheck::{gradcheck, GradcheckOptions};
pub use graph::{
    enable_grad, is_grad_enabled, no_grad, set_grad_enabled, BackwardCtx, BackwardOptions, GradModeGuard, InputMeta, Node, Op,
};
pub use index::{IndexOp, TensorIndexer};
pub use tensor_macro::NestedData;

pub use ops::{sum_to_shape, Reduction};

use graph::Edge;

/// The raw array type, without autograd.
pub type RawTensor = weta_tensor::Tensor;

pub(crate) struct Inner {
    data: RawTensor,
    grad_fn: Option<Arc<Node>>,
    requires_grad: AtomicBool,
    grad: Mutex<Option<Tensor>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        if let Some(n) = self.grad_fn.take() {
            graph::defer_drop(n);
        }
    }
}

/// An N-dimensional array that can record the operations applied to it and
/// compute gradients with [`backward`](Tensor::backward). Cloning is cheap
/// and shares data, gradient and graph, like assigning a PyTorch tensor to
/// a new name.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<Inner>,
}

impl Tensor {
    fn new_inner(data: RawTensor, grad_fn: Option<Arc<Node>>) -> Tensor {
        Tensor { inner: Arc::new(Inner { data, grad_fn, requires_grad: AtomicBool::new(false), grad: Mutex::new(None) }) }
    }

    pub(crate) fn from_inner(inner: Arc<Inner>) -> Tensor {
        Tensor { inner }
    }

    /// Wraps a raw tensor as a constant (not requiring grad).
    pub fn from_raw(data: RawTensor) -> Tensor {
        Tensor::new_inner(data, None)
    }

    /// Records `data`, the result of `op` applied to `inputs`, on the graph
    /// if grad mode is on and any input requires grad. This is the one hook
    /// a new differentiable op needs.
    pub fn from_op(data: RawTensor, op: impl Op + 'static, inputs: &[&Tensor]) -> Tensor {
        if !graph::is_grad_enabled() || !inputs.iter().any(|t| t.requires_grad()) {
            return Tensor::from_raw(data);
        }
        let edges: SmallVec<[Edge; 2]> = inputs.iter().map(|t| t.edge()).collect();
        let meta = inputs
            .iter()
            .map(|t| InputMeta { shape: t.shape().to_vec(), dtype: t.dtype(), device: t.device(), requires_grad: t.requires_grad() })
            .collect();
        let node = Node::new(Box::new(op), edges, meta);
        Tensor::new_inner(data, Some(Arc::new(node)))
    }

    fn edge(&self) -> Edge {
        if let Some(n) = &self.inner.grad_fn {
            Edge::Node(n.clone())
        } else if self.requires_grad() {
            Edge::Leaf(self.inner.clone())
        } else {
            Edge::None
        }
    }

    /// The underlying array (no autograd).
    pub fn data(&self) -> &RawTensor {
        &self.inner.data
    }

    pub fn grad_fn(&self) -> Option<Arc<Node>> {
        self.inner.grad_fn.clone()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.grad_fn.is_some() || self.inner.requires_grad.load(Ordering::Relaxed)
    }

    /// True for tensors not made by a recorded op (inputs and parameters).
    pub fn is_leaf(&self) -> bool {
        self.inner.grad_fn.is_none()
    }

    /// Marks a leaf as requiring grad, as PyTorch's `requires_grad_()`. It
    /// changes this tensor (and its clones) and returns it for chaining.
    #[track_caller]
    pub fn requires_grad_(&self, requires_grad: bool) -> Tensor {
        assert!(self.is_leaf() || requires_grad, "you can only change requires_grad flags of leaf variables; use .detach() for a non-leaf");
        if requires_grad {
            assert!(self.dtype().is_float(), "only floating point tensors can require gradients, got {}", self.dtype());
        }
        self.inner.requires_grad.store(requires_grad, Ordering::Relaxed);
        self.clone()
    }

    /// The accumulated gradient, or `None` if backward hasn't reached it.
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.grad.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn set_grad(&self, g: Option<Tensor>) {
        *self.inner.grad.lock().unwrap_or_else(|e| e.into_inner()) = g;
    }

    /// Clears the gradient (`p.grad = None`, as `zero_grad(set_to_none=True)`).
    pub fn zero_grad(&self) {
        self.set_grad(None);
    }

    pub(crate) fn accumulate_grad(&self, g: Tensor) {
        let g = if graph::is_grad_enabled() { g } else { g.detach() };
        let mut slot = self.inner.grad.lock().unwrap_or_else(|e| e.into_inner());
        *slot = Some(match slot.take() {
            Some(prev) => &prev + &g,
            None => g,
        });
    }

    /// Computes the gradient of this scalar with respect to every leaf that
    /// requires grad, accumulating into their `.grad()`.
    #[track_caller]
    pub fn backward(&self) {
        self.backward_with(BackwardOptions::default())
    }

    #[track_caller]
    pub fn backward_with(&self, opts: BackwardOptions) {
        assert_eq!(
            self.numel(),
            1,
            "grad can be implicitly created only for scalar outputs; this tensor has shape {:?} (use backward_with_grad)",
            self.shape()
        );
        let seed = Tensor::from_raw(self.data().ones_like());
        graph::run_backward(self, seed, opts);
    }

    /// Backward from a non-scalar tensor, given `dL/dself`.
    pub fn backward_with_grad(&self, grad: &Tensor, opts: BackwardOptions) {
        graph::run_backward(self, grad.clone(), opts);
    }

    /// The same data, cut out of the graph.
    pub fn detach(&self) -> Tensor {
        Tensor::from_raw(self.data().clone())
    }
}

// ---------------------------------------------------------------------------
// Constructors. Defaults follow PyTorch: f32 on the CPU.
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn from_vec<T: weta_tensor::IntoHost>(data: Vec<T>, shape: &[usize]) -> Tensor {
        Tensor::from_raw(RawTensor::from_vec(data, shape))
    }
    pub fn from_slice<T: weta_tensor::IntoHost>(data: &[T], shape: &[usize]) -> Tensor {
        Tensor::from_raw(RawTensor::from_slice(data, shape))
    }
    pub fn scalar(v: f64) -> Tensor {
        Tensor::from_raw(RawTensor::scalar(v, DType::F32, Device::Cpu))
    }
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::from_raw(RawTensor::zeros(shape, DType::F32, Device::Cpu))
    }
    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::from_raw(RawTensor::ones(shape, DType::F32, Device::Cpu))
    }
    pub fn full(shape: &[usize], v: f64) -> Tensor {
        Tensor::from_raw(RawTensor::full(shape, v, DType::F32, Device::Cpu))
    }
    /// Zeros with an explicit dtype and device.
    pub fn zeros_on(shape: &[usize], dtype: DType, device: Device) -> Tensor {
        Tensor::from_raw(RawTensor::zeros(shape, dtype, device))
    }
    pub fn zeros_like(&self) -> Tensor {
        Tensor::from_raw(self.data().zeros_like())
    }
    pub fn ones_like(&self) -> Tensor {
        Tensor::from_raw(self.data().ones_like())
    }
    pub fn rand(shape: &[usize]) -> Tensor {
        Tensor::from_raw(RawTensor::rand(shape, DType::F32, Device::Cpu))
    }
    pub fn randn(shape: &[usize]) -> Tensor {
        Tensor::from_raw(RawTensor::randn(shape, DType::F32, Device::Cpu))
    }
    pub fn rand_like(&self) -> Tensor {
        Tensor::from_raw(RawTensor::rand(self.shape(), self.dtype(), self.device()))
    }
    pub fn randn_like(&self) -> Tensor {
        Tensor::from_raw(RawTensor::randn(self.shape(), self.dtype(), self.device()))
    }
    pub fn randint(low: i64, high: i64, shape: &[usize]) -> Tensor {
        Tensor::from_raw(RawTensor::randint(low, high, shape, Device::Cpu))
    }
    /// `torch.arange(start, end, step)` as f32.
    pub fn arange(start: f64, end: f64, step: f64) -> Tensor {
        Tensor::from_raw(RawTensor::arange(start, end, step, DType::F32, Device::Cpu))
    }
    pub fn eye(n: usize) -> Tensor {
        Tensor::from_raw(RawTensor::eye(n, DType::F32, Device::Cpu))
    }
    /// A tensor from nested arrays; used by the [`tensor!`] macro.
    pub fn from_nested<T: NestedData>(v: T) -> Tensor {
        tensor_macro::from_nested(v)
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        self.data().shape()
    }
    pub fn dims(&self) -> &[usize] {
        self.data().shape()
    }
    #[track_caller]
    pub fn size(&self, dim: isize) -> usize {
        self.data().size(dim)
    }
    pub fn ndim(&self) -> usize {
        self.data().ndim()
    }
    pub fn dim(&self) -> usize {
        self.ndim()
    }
    pub fn numel(&self) -> usize {
        self.data().numel()
    }
    pub fn dtype(&self) -> DType {
        self.data().dtype()
    }
    pub fn device(&self) -> Device {
        self.data().device()
    }
    pub fn is_contiguous(&self) -> bool {
        self.data().is_contiguous()
    }
    #[track_caller]
    pub fn item(&self) -> f64 {
        self.data().item()
    }
    pub fn to_vec<T: weta_tensor::FromHost>(&self) -> Vec<T> {
        self.data().to_vec()
    }
    /// True if both are the same tensor (share data, grad and graph).
    pub fn ptr_eq(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.data().to_string();
        let s = s.strip_suffix(')').unwrap_or(&s);
        match &self.inner.grad_fn {
            Some(n) => write!(f, "{s}, grad_fn=<{}Backward>)", n.name()),
            None if self.requires_grad() => write!(f, "{s}, requires_grad=true)"),
            None => write!(f, "{s})"),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor[{:?}, {}, {}", self.shape(), self.dtype(), self.device())?;
        if self.requires_grad() {
            write!(f, ", requires_grad")?;
        }
        write!(f, "]")
    }
}

impl From<RawTensor> for Tensor {
    fn from(t: RawTensor) -> Tensor {
        Tensor::from_raw(t)
    }
}
