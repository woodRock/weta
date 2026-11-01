use std::sync::Arc;

use weta_core::{
    broadcast_shapes, resolve_shape, BinaryOp, DType, Device, Element, Error, HostData, Im2ColParams, Layout, ReduceOp, Result, UnaryOp,
};

use crate::storage::{Storage, StorageCell};

/// An N-dimensional array: a cheap, clonable handle to shared storage plus
/// a [`Layout`]. Cloning a tensor, or taking a view of it (transpose, narrow,
/// reshape of contiguous data, broadcast), never copies elements.
///
/// This is the plain array type, with no gradients; `weta-autograd` wraps it
/// to record operations. Inference code uses it directly.
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<StorageCell>,
    layout: Layout,
    dtype: DType,
    device: Device,
}

/// Panics with an error's message, pointing at the caller's line.
#[track_caller]
pub(crate) fn ok<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}

/// A tensor's data copied out to the host: implemented for `f32`, `f64`
/// and `i64` so `x.to_vec::<f32>()` works whatever the tensor's dtype.
pub trait FromHost: Element {
    fn from_host(h: HostData) -> Vec<Self>;
}

impl FromHost for f32 {
    fn from_host(h: HostData) -> Vec<f32> {
        match h {
            HostData::F32(v) => v,
            HostData::F64(v) => v.into_iter().map(|x| x as f32).collect(),
            HostData::I64(v) => v.into_iter().map(|x| x as f32).collect(),
        }
    }
}
impl FromHost for f64 {
    fn from_host(h: HostData) -> Vec<f64> {
        h.to_f64_vec()
    }
}
impl FromHost for i64 {
    fn from_host(h: HostData) -> Vec<i64> {
        match h {
            HostData::I64(v) => v,
            HostData::F32(v) => v.into_iter().map(|x| x as i64).collect(),
            HostData::F64(v) => v.into_iter().map(|x| x as i64).collect(),
        }
    }
}

/// Converts a host vector of any supported element type into [`HostData`].
pub trait IntoHost: Element {
    fn into_host(v: Vec<Self>) -> HostData;
}
impl IntoHost for f32 {
    fn into_host(v: Vec<f32>) -> HostData {
        HostData::F32(v)
    }
}
impl IntoHost for f64 {
    fn into_host(v: Vec<f64>) -> HostData {
        HostData::F64(v)
    }
}
impl IntoHost for i64 {
    fn into_host(v: Vec<i64>) -> HostData {
        HostData::I64(v)
    }
}

// ---------------------------------------------------------------------------
// Construction and accessors
// ---------------------------------------------------------------------------

impl Tensor {
    pub(crate) fn from_storage(storage: Storage, shape: &[usize], dtype: DType, device: Device) -> Tensor {
        Tensor { storage: Arc::new(StorageCell::new(storage)), layout: Layout::contiguous(shape), dtype, device }
    }

    fn view_with(&self, layout: Layout) -> Tensor {
        Tensor { storage: self.storage.clone(), layout, dtype: self.dtype, device: self.device }
    }

    /// A new tensor from host data, on the CPU.
    pub fn try_from_host(data: HostData, shape: &[usize], device: Device) -> Result<Tensor> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(Error::shape("from_vec", format!("{} elements can't have shape {shape:?}", data.len())));
        }
        let dtype = data.dtype();
        Ok(Tensor::from_storage(Storage::from_host(data, device)?, shape, dtype, device))
    }

    /// `Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2])`, like `torch.tensor(...).reshape(2, 2)`.
    #[track_caller]
    pub fn from_vec<T: IntoHost>(data: Vec<T>, shape: &[usize]) -> Tensor {
        ok(Tensor::try_from_host(T::into_host(data), shape, Device::Cpu))
    }

    pub fn from_slice<T: IntoHost>(data: &[T], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape)
    }

    /// A 0-dimensional tensor holding one value.
    pub fn scalar(v: f64, dtype: DType, device: Device) -> Tensor {
        Tensor::full(&[], v, dtype, device)
    }

    #[track_caller]
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Tensor {
        let n = shape.iter().product();
        Tensor::from_storage(ok(Storage::zeros(n, dtype, device)), shape, dtype, device)
    }

    pub fn ones(shape: &[usize], dtype: DType, device: Device) -> Tensor {
        Tensor::full(shape, 1.0, dtype, device)
    }

    pub fn full(shape: &[usize], value: f64, dtype: DType, device: Device) -> Tensor {
        Tensor::zeros(shape, dtype, device).affine(0.0, value)
    }

    pub fn zeros_like(&self) -> Tensor {
        Tensor::zeros(self.shape(), self.dtype, self.device)
    }

    pub fn ones_like(&self) -> Tensor {
        Tensor::ones(self.shape(), self.dtype, self.device)
    }

    /// `[start, start + step, ...)` up to `end`, as in `torch.arange`.
    pub fn arange(start: f64, end: f64, step: f64, dtype: DType, device: Device) -> Tensor {
        let n = ((end - start) / step).ceil().max(0.0) as usize;
        let vals: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Tensor::from_vec(vals, &[n]).to_dtype(dtype).to(device)
    }

    pub fn eye(n: usize, dtype: DType, device: Device) -> Tensor {
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        Tensor::from_vec(v, &[n, n]).to_dtype(dtype).to(device)
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }
    /// The size of one dimension; negative indices count from the end.
    #[track_caller]
    pub fn size(&self, dim: isize) -> usize {
        self.shape()[self.dim_index(dim)]
    }
    pub fn dims(&self) -> &[usize] {
        self.shape()
    }
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }
    /// Bumped by every in-place write to this tensor's storage.
    pub fn version(&self) -> u64 {
        self.storage.version()
    }
    /// True if both tensors view the same storage.
    pub fn same_storage(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Resolves a possibly negative dimension index.
    #[track_caller]
    pub fn dim_index(&self, dim: isize) -> usize {
        let n = self.ndim() as isize;
        let d = if dim < 0 { dim + n } else { dim };
        assert!(d >= 0 && d < n.max(1), "dimension {dim} out of range for a {n}-d tensor of shape {:?}", self.shape());
        d as usize
    }

    pub fn to_host(&self) -> HostData {
        self.storage.read().to_host(&self.layout, self.dtype)
    }

    /// Copies the elements to a `Vec` in row-major order, converting to `T`.
    pub fn to_vec<T: FromHost>(&self) -> Vec<T> {
        T::from_host(self.to_host())
    }

    /// The value of a one-element tensor, as PyTorch's `.item()`.
    #[track_caller]
    pub fn item(&self) -> f64 {
        assert_eq!(
            self.numel(),
            1,
            "item: tensor with {} elements (shape {:?}) can't be converted to a scalar",
            self.numel(),
            self.shape()
        );
        self.to_host().to_f64_vec()[0]
    }

    /// The same data on another device (a no-op if already there).
    #[track_caller]
    pub fn to(&self, device: Device) -> Tensor {
        ok(self.try_to(device))
    }

    pub fn try_to(&self, device: Device) -> Result<Tensor> {
        if device == self.device {
            return Ok(self.clone());
        }
        Tensor::try_from_host(self.to_host(), self.shape(), device)
    }

    pub fn to_dtype(&self, dtype: DType) -> Tensor {
        if dtype == self.dtype {
            return self.clone();
        }
        let s = self.storage.read().cast(&self.layout, self.dtype, dtype);
        Tensor::from_storage(s, self.shape(), dtype, self.device)
    }

    /// A contiguous copy, or `self` if already contiguous.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        self.copy()
    }

    /// Always a fresh copy (PyTorch's `clone()`; Rust's `clone()` shares storage).
    pub fn copy(&self) -> Tensor {
        let s = self.storage.read().cast(&self.layout, self.dtype, self.dtype);
        Tensor::from_storage(s, self.shape(), self.dtype, self.device)
    }
}

// ---------------------------------------------------------------------------
// Views: new layouts over the same storage
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn try_reshape(&self, shape: &[isize]) -> Result<Tensor> {
        let shape = resolve_shape("reshape", self.numel(), shape)?;
        Ok(match self.layout.reshape(&shape) {
            Some(l) => self.view_with(l),
            None => self.copy().view_with(Layout::contiguous(&shape)),
        })
    }

    /// Reshapes, copying only if the data isn't contiguous. `-1` is inferred.
    #[track_caller]
    pub fn reshape(&self, shape: &[isize]) -> Tensor {
        ok(self.try_reshape(shape))
    }

    /// `reshape` with a `usize` shape.
    #[track_caller]
    pub fn reshape_to(&self, shape: &[usize]) -> Tensor {
        let s: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
        self.reshape(&s)
    }

    pub fn try_transpose(&self, d0: isize, d1: isize) -> Result<Tensor> {
        let (a, b) = (self.dim_index(d0), self.dim_index(d1));
        Ok(self.view_with(self.layout.transpose(a, b)?))
    }

    #[track_caller]
    pub fn transpose(&self, d0: isize, d1: isize) -> Tensor {
        ok(self.try_transpose(d0, d1))
    }

    /// Transposes the last two dimensions (PyTorch's `.T` / `.mT` for matrices).
    #[track_caller]
    pub fn t(&self) -> Tensor {
        assert!(self.ndim() >= 2, "t: expected at least 2 dims, got shape {:?}", self.shape());
        self.transpose(-2, -1)
    }

    #[track_caller]
    pub fn permute(&self, dims: &[usize]) -> Tensor {
        ok(self.layout.permute(dims).map(|l| self.view_with(l)))
    }

    pub fn try_narrow(&self, dim: isize, start: usize, len: usize) -> Result<Tensor> {
        let d = self.dim_index(dim);
        Ok(self.view_with(self.layout.narrow(d, start, len)?))
    }

    /// `len` elements of `dim` starting at `start`: `x[:, start:start+len]`.
    #[track_caller]
    pub fn narrow(&self, dim: isize, start: usize, len: usize) -> Tensor {
        ok(self.try_narrow(dim, start, len))
    }

    /// Every `step`-th element of `dim`.
    pub fn step_by(&self, dim: isize, step: usize) -> Tensor {
        let d = self.dim_index(dim);
        self.view_with(self.layout.step(d, step))
    }

    /// Index `i` of `dim`, removing that dimension: `x[:, i]`.
    #[track_caller]
    pub fn select(&self, dim: isize, i: usize) -> Tensor {
        let d = self.dim_index(dim);
        self.narrow(d as isize, i, 1).squeeze(d as isize)
    }

    #[track_caller]
    pub fn unsqueeze(&self, dim: isize) -> Tensor {
        let n = self.ndim() as isize + 1;
        let d = if dim < 0 { dim + n } else { dim };
        ok(self.layout.unsqueeze(d as usize).map(|l| self.view_with(l)))
    }

    #[track_caller]
    pub fn squeeze(&self, dim: isize) -> Tensor {
        let d = self.dim_index(dim);
        ok(self.layout.squeeze(d).map(|l| self.view_with(l)))
    }

    /// Flattens dims `start..=end` into one, as `torch.flatten`.
    #[track_caller]
    pub fn flatten(&self, start: isize, end: isize) -> Tensor {
        if self.ndim() == 0 {
            return self.reshape(&[1]);
        }
        let (s, e) = (self.dim_index(start), self.dim_index(end));
        let mut shape: Vec<isize> = self.shape()[..s].iter().map(|&d| d as isize).collect();
        shape.push(self.shape()[s..=e].iter().product::<usize>() as isize);
        shape.extend(self.shape()[e + 1..].iter().map(|&d| d as isize));
        self.reshape(&shape)
    }

    pub fn try_broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        Ok(self.view_with(self.layout.broadcast_to(shape)?))
    }

    /// A stride-0 view with a larger shape (`torch.expand`/`broadcast_to`).
    #[track_caller]
    pub fn broadcast_to(&self, shape: &[usize]) -> Tensor {
        ok(self.try_broadcast_to(shape))
    }
}

// ---------------------------------------------------------------------------
// Element-wise ops
// ---------------------------------------------------------------------------

impl Tensor {
    pub fn unary(&self, op: UnaryOp) -> Tensor {
        let s = self.storage.read().unary(op, &self.layout, self.dtype);
        let dtype = self.dtype;
        Tensor::from_storage(s, self.shape(), dtype, self.device)
    }

    fn check_same(&self, op: &'static str, other: &Tensor) -> Result<()> {
        if self.device != other.device {
            return Err(Error::Device { op, a: self.device, b: other.device });
        }
        if self.dtype != other.dtype {
            return Err(Error::dtype(op, self.dtype, other.dtype));
        }
        Ok(())
    }

    /// Element-wise op with NumPy broadcasting.
    pub fn try_binary(&self, op: BinaryOp, other: &Tensor) -> Result<Tensor> {
        self.check_same(op.name(), other)?;
        let shape = broadcast_shapes(op.name(), self.shape(), other.shape())?;
        let la = self.layout.broadcast_to(&shape)?;
        let lb = other.layout.broadcast_to(&shape)?;
        let s = {
            let (sa, sb) = (self.storage.read(), other.storage.read());
            sa.binary(op, &la, &sb, &lb, self.dtype)
        };
        Ok(Tensor::from_storage(s, &shape, self.dtype, self.device))
    }

    #[track_caller]
    pub fn binary(&self, op: BinaryOp, other: &Tensor) -> Tensor {
        ok(self.try_binary(op, other))
    }

    pub fn affine(&self, mul: f64, add: f64) -> Tensor {
        self.unary(UnaryOp::Affine { mul, add })
    }
}

macro_rules! unary_methods {
    ($($name:ident => $op:expr),* $(,)?) => {
        impl Tensor {
            $(pub fn $name(&self) -> Tensor { self.unary($op) })*
        }
    };
}

unary_methods! {
    neg => UnaryOp::Neg, exp => UnaryOp::Exp, log => UnaryOp::Log, sqrt => UnaryOp::Sqrt,
    abs => UnaryOp::Abs, sign => UnaryOp::Sign, relu => UnaryOp::Relu, step => UnaryOp::Step,
    tanh => UnaryOp::Tanh, sigmoid => UnaryOp::Sigmoid, erf => UnaryOp::Erf, sin => UnaryOp::Sin,
    cos => UnaryOp::Cos, reciprocal => UnaryOp::Recip,
}

macro_rules! binary_methods {
    ($($name:ident, $try_name:ident => $op:expr),* $(,)?) => {
        impl Tensor {
            $(
                #[track_caller]
                pub fn $name(&self, other: &Tensor) -> Tensor { self.binary($op, other) }
                pub fn $try_name(&self, other: &Tensor) -> Result<Tensor> { self.try_binary($op, other) }
            )*
        }
    };
}

binary_methods! {
    add, try_add => BinaryOp::Add, sub, try_sub => BinaryOp::Sub,
    mul, try_mul => BinaryOp::Mul, div, try_div => BinaryOp::Div,
    pow, try_pow => BinaryOp::Pow, maximum, try_maximum => BinaryOp::Maximum,
    minimum, try_minimum => BinaryOp::Minimum,
    eq, try_eq => BinaryOp::Eq, ne, try_ne => BinaryOp::Ne, lt, try_lt => BinaryOp::Lt,
    le, try_le => BinaryOp::Le, gt, try_gt => BinaryOp::Gt, ge, try_ge => BinaryOp::Ge,
}

impl Tensor {
    pub fn add_scalar(&self, v: f64) -> Tensor {
        self.affine(1.0, v)
    }
    pub fn mul_scalar(&self, v: f64) -> Tensor {
        self.affine(v, 0.0)
    }
    pub fn powf(&self, e: f64) -> Tensor {
        self.unary(UnaryOp::Powf(e))
    }
    pub fn clamp_min(&self, v: f64) -> Tensor {
        self.unary(UnaryOp::MaxScalar(v))
    }
    pub fn clamp_max(&self, v: f64) -> Tensor {
        self.unary(UnaryOp::MinScalar(v))
    }
    pub fn clamp(&self, lo: f64, hi: f64) -> Tensor {
        self.clamp_min(lo).clamp_max(hi)
    }
    pub fn gt_scalar(&self, v: f64) -> Tensor {
        self.unary(UnaryOp::GtScalar(v))
    }
    /// `cond ? a : b` element-wise, where `self` is a 0/1 mask.
    pub fn where_cond(&self, a: &Tensor, b: &Tensor) -> Tensor {
        let m = self.to_dtype(a.dtype);
        m.mul(a).add(&m.affine(-1.0, 1.0).mul(b))
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

impl Tensor {
    fn reduce_axis(&self, op: ReduceOp, axis: usize) -> Tensor {
        let s = self.storage.read().reduce(op, &self.layout, self.dtype, axis);
        let mut shape = self.shape().to_vec();
        shape.remove(axis);
        let dtype = if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) { DType::I64 } else { self.dtype };
        Tensor::from_storage(s, &shape, dtype, self.device)
    }

    fn reduce(&self, op: ReduceOp, dims: &[isize], keepdim: bool) -> Tensor {
        let mut axes: Vec<usize> = dims.iter().map(|&d| self.dim_index(d)).collect();
        axes.sort_unstable();
        axes.dedup();
        let mut out = self.clone();
        for &a in axes.iter().rev() {
            out = out.reduce_axis(op, a);
        }
        if keepdim {
            for &a in &axes {
                out = out.unsqueeze(a as isize);
            }
        }
        out
    }

    fn all_dims(&self) -> Vec<isize> {
        (0..self.ndim() as isize).collect()
    }

    /// Sum over `dims` (`torch.sum(x, dims, keepdim)`).
    pub fn sum_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        self.reduce(ReduceOp::Sum, dims, keepdim)
    }
    /// Sum of every element, as a 0-d tensor.
    pub fn sum(&self) -> Tensor {
        self.sum_dims(&self.all_dims(), false)
    }
    pub fn mean_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        let n: usize = dims.iter().map(|&d| self.size(d)).product();
        self.sum_dims(dims, keepdim).mul_scalar(1.0 / n as f64)
    }
    pub fn mean(&self) -> Tensor {
        self.sum().mul_scalar(1.0 / self.numel() as f64)
    }
    pub fn max_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        self.reduce(ReduceOp::Max, dims, keepdim)
    }
    pub fn max(&self) -> Tensor {
        self.max_dims(&self.all_dims(), false)
    }
    pub fn min_dims(&self, dims: &[isize], keepdim: bool) -> Tensor {
        self.reduce(ReduceOp::Min, dims, keepdim)
    }
    pub fn min(&self) -> Tensor {
        self.min_dims(&self.all_dims(), false)
    }
    #[track_caller]
    pub fn argmax(&self, dim: isize, keepdim: bool) -> Tensor {
        self.reduce(ReduceOp::ArgMax, &[dim], keepdim)
    }
    #[track_caller]
    pub fn argmin(&self, dim: isize, keepdim: bool) -> Tensor {
        self.reduce(ReduceOp::ArgMin, &[dim], keepdim)
    }
    /// Variance over `dims`, with Bessel's correction if `unbiased`.
    pub fn var_dims(&self, dims: &[isize], unbiased: bool, keepdim: bool) -> Tensor {
        let n: usize = dims.iter().map(|&d| self.size(d)).product();
        let centered = self.sub(&self.mean_dims(dims, true));
        let denom = if unbiased { n.saturating_sub(1).max(1) } else { n };
        centered.mul(&centered).sum_dims(dims, keepdim).mul_scalar(1.0 / denom as f64)
    }

    /// Numerically stable `log Σ exp(x)` over `dim`.
    pub fn logsumexp(&self, dim: isize, keepdim: bool) -> Tensor {
        let m = self.max_dims(&[dim], true);
        let out = self.sub(&m).exp().sum_dims(&[dim], true).log().add(&m);
        if keepdim {
            out
        } else {
            out.squeeze(dim)
        }
    }

    pub fn softmax(&self, dim: isize) -> Tensor {
        let e = self.sub(&self.max_dims(&[dim], true)).exp();
        e.div(&e.sum_dims(&[dim], true))
    }

    pub fn log_softmax(&self, dim: isize) -> Tensor {
        self.sub(&self.logsumexp(dim, true))
    }
}

// ---------------------------------------------------------------------------
// Matrix multiply, convolution helpers, concatenation
// ---------------------------------------------------------------------------

/// Tries to view the leading `nb` dims of a layout as one batch dimension,
/// giving a `[B, M, K]` layout without copying.
fn merge_batch(l: &Layout) -> Option<Layout> {
    let nb = l.ndim() - 2;
    let (shape, strides) = (l.shape(), l.strides());
    let mut b = 1;
    let mut stride = 0;
    // Batch dims merge if each one steps over exactly the dims after it.
    let mut first = true;
    for i in (0..nb).rev() {
        if shape[i] == 1 {
            continue;
        }
        if first {
            stride = strides[i];
            first = false;
        } else if strides[i] != stride * b {
            return None;
        }
        b *= shape[i];
    }
    Some(Layout::new(vec![b, shape[nb], shape[nb + 1]], vec![stride, strides[nb], strides[nb + 1]], l.offset()))
}

impl Tensor {
    /// Matrix product with PyTorch's `torch.matmul` semantics: 1-D inputs are
    /// treated as vectors, and leading (batch) dimensions broadcast.
    pub fn try_matmul(&self, other: &Tensor) -> Result<Tensor> {
        self.check_same("matmul", other)?;
        if self.ndim() == 0 || other.ndim() == 0 {
            return Err(Error::shape("matmul", "both arguments need at least 1 dimension"));
        }
        let a = if self.ndim() == 1 { self.unsqueeze(0) } else { self.clone() };
        let b = if other.ndim() == 1 { other.unsqueeze(1) } else { other.clone() };
        let (m, k) = (a.size(-2), a.size(-1));
        let (k2, n) = (b.size(-2), b.size(-1));
        if k != k2 {
            return Err(Error::shape("matmul", format!("{:?} × {:?}: inner dims {k} ≠ {k2}", self.shape(), other.shape())));
        }
        let batch = broadcast_shapes("matmul", &a.shape()[..a.ndim() - 2], &b.shape()[..b.ndim() - 2])?;
        let full = |t: &Tensor, r: usize, c: usize| -> Result<Layout> {
            let mut s = batch.clone();
            s.extend([r, c]);
            let mut l = t.layout.broadcast_to(&s)?;
            if batch.is_empty() {
                l = l.unsqueeze(0)?;
            }
            Ok(l)
        };
        let (fa, fb) = (full(&a, m, k)?, full(&b, k, n)?);
        // Batch dims that can't be merged into one stride need a copy first.
        let (sa, la) = match merge_batch(&fa) {
            Some(l) => (a.storage.clone(), l),
            None => {
                let c = a.view_with(fa.clone()).copy();
                let l = merge_batch(&c.layout).unwrap();
                (c.storage, l)
            }
        };
        let (sb, lb) = match merge_batch(&fb) {
            Some(l) => (b.storage.clone(), l),
            None => {
                let c = b.view_with(fb.clone()).copy();
                let l = merge_batch(&c.layout).unwrap();
                (c.storage, l)
            }
        };
        let s = sa.read().matmul(&la, &sb.read(), &lb, self.dtype);
        let mut shape = batch;
        if self.ndim() > 1 {
            shape.push(m);
        }
        if other.ndim() > 1 {
            shape.push(n);
        }
        Ok(Tensor::from_storage(s, &shape, self.dtype, self.device))
    }

    #[track_caller]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        ok(self.try_matmul(other))
    }

    /// `[N, C, H, W]` → `[N, C·kh·kw, OH·OW]` (`torch.nn.functional.unfold`).
    #[track_caller]
    pub fn im2col(&self, p: Im2ColParams) -> Tensor {
        assert_eq!(self.ndim(), 4, "im2col: expected [N, C, H, W], got {:?}", self.shape());
        let s = self.shape();
        let (oh, ow) = p.out_size(s[2], s[3]);
        let st = self.storage.read().im2col(&self.layout, self.dtype, p);
        Tensor::from_storage(st, &[s[0], s[1] * p.kernel.0 * p.kernel.1, oh * ow], self.dtype, self.device)
    }

    /// The adjoint of [`im2col`](Self::im2col) (`torch.nn.functional.fold`).
    #[track_caller]
    pub fn col2im(&self, p: Im2ColParams, (c, h, w): (usize, usize, usize)) -> Tensor {
        assert_eq!(self.ndim(), 3, "col2im: expected [N, C·kh·kw, L], got {:?}", self.shape());
        let st = self.storage.read().col2im(&self.layout, self.dtype, p, (c, h, w));
        Tensor::from_storage(st, &[self.shape()[0], c, h, w], self.dtype, self.device)
    }

    /// 2-D convolution: `x [N, C, H, W]`, `w [O, C, kh, kw]`, optional `b [O]`.
    #[track_caller]
    pub fn conv2d(&self, w: &Tensor, b: Option<&Tensor>, stride: usize, padding: usize) -> Tensor {
        let (n, h, wd) = (self.size(0), self.size(2), self.size(3));
        let (o, kh, kw) = (w.size(0), w.size(2), w.size(3));
        let p = Im2ColParams { kernel: (kh, kw), stride: (stride, stride), padding: (padding, padding), dilation: (1, 1) };
        let (oh, ow) = p.out_size(h, wd);
        let cols = self.im2col(p);
        let mut out = w.reshape(&[o as isize, -1]).matmul(&cols);
        if let Some(b) = b {
            out = out.add(&b.reshape(&[o as isize, 1]));
        }
        out.reshape_to(&[n, o, oh, ow])
    }

    /// 2-D max pooling with a square window.
    #[track_caller]
    pub fn max_pool2d(&self, kernel: usize, stride: usize) -> Tensor {
        let (n, c, h, w) = (self.size(0), self.size(1), self.size(2), self.size(3));
        let p = Im2ColParams { kernel: (kernel, kernel), stride: (stride, stride), padding: (0, 0), dilation: (1, 1) };
        let (oh, ow) = p.out_size(h, w);
        let cols = self.reshape_to(&[n * c, 1, h, w]).im2col(p);
        cols.max_dims(&[1], false).reshape_to(&[n, c, oh, ow])
    }

    pub fn try_cat(tensors: &[&Tensor], dim: isize) -> Result<Tensor> {
        let first = *tensors.first().ok_or_else(|| Error::shape("cat", "expected at least one tensor"))?;
        let d = first.dim_index(dim);
        let mut shape = first.shape().to_vec();
        shape[d] = 0;
        for t in tensors {
            first.check_same("cat", t)?;
            let ok = t.ndim() == first.ndim() && t.shape().iter().zip(first.shape()).enumerate().all(|(i, (a, b))| i == d || a == b);
            if !ok {
                return Err(Error::shape("cat", format!("can't concatenate {:?} and {:?} along dim {d}", first.shape(), t.shape())));
            }
            shape[d] += t.shape()[d];
        }
        let out = Tensor::zeros(&shape, first.dtype, first.device);
        let mut start = 0;
        for t in tensors {
            let len = t.shape()[d];
            out.narrow(d as isize, start, len).assign(t);
            start += len;
        }
        Ok(out)
    }

    /// Concatenates along `dim` (`torch.cat`).
    #[track_caller]
    pub fn cat(tensors: &[&Tensor], dim: isize) -> Tensor {
        ok(Tensor::try_cat(tensors, dim))
    }

    /// Stacks along a new dimension (`torch.stack`).
    #[track_caller]
    pub fn stack(tensors: &[&Tensor], dim: isize) -> Tensor {
        let us: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();
        Tensor::cat(&us.iter().collect::<Vec<_>>(), dim)
    }

    /// One-hot encoding of `I64` class indices: `[N]` → `[N, classes]`.
    pub fn one_hot(&self, classes: usize, dtype: DType) -> Tensor {
        let ids = Tensor::arange(0.0, classes as f64, 1.0, DType::I64, self.device);
        self.unsqueeze(-1).eq(&ids).to_dtype(dtype)
    }

    /// Picks `self[i, index[i]]` for each row: `[N, C]`, `[N]` → `[N]`.
    pub fn gather_rows(&self, index: &Tensor) -> Tensor {
        self.mul(&index.one_hot(self.size(-1), self.dtype)).sum_dims(&[-1], false)
    }
}

// ---------------------------------------------------------------------------
// In-place ops (trailing underscore, as in PyTorch)
// ---------------------------------------------------------------------------

impl Tensor {
    /// Copies `src` (broadcast to this shape) into this tensor's elements,
    /// converting dtype and device as needed (`copy_` in PyTorch).
    #[track_caller]
    pub fn assign(&self, src: &Tensor) {
        let src = src.to(self.device).to_dtype(self.dtype);
        let ls = ok(src.layout.broadcast_to(self.shape()));
        if self.same_storage(&src) {
            // Read and write the same storage: go through a copy.
            let tmp = src.view_with(ls).copy();
            self.storage.write().copy_from(&self.layout, &tmp.storage.read(), &tmp.layout, self.dtype);
        } else {
            let s = src.storage.read();
            self.storage.write().copy_from(&self.layout, &s, &ls, self.dtype);
        }
    }

    #[track_caller]
    pub fn copy_(&self, src: &Tensor) {
        self.assign(src)
    }
    pub fn zero_(&self) {
        self.assign(&Tensor::scalar(0.0, self.dtype, self.device));
    }
    pub fn fill_(&self, v: f64) {
        self.assign(&Tensor::scalar(v, self.dtype, self.device));
    }
    #[track_caller]
    pub fn add_(&self, other: &Tensor) {
        self.assign(&self.add(other))
    }
    #[track_caller]
    pub fn sub_(&self, other: &Tensor) {
        self.assign(&self.sub(other))
    }
    #[track_caller]
    pub fn mul_(&self, other: &Tensor) {
        self.assign(&self.mul(other))
    }
    #[track_caller]
    pub fn div_(&self, other: &Tensor) {
        self.assign(&self.div(other))
    }
    pub fn mul_scalar_(&self, v: f64) {
        self.assign(&self.mul_scalar(v))
    }
    pub fn add_scalar_(&self, v: f64) {
        self.assign(&self.add_scalar(v))
    }
}

/// Element-wise closeness check used by tests: `|a − b| ≤ atol + rtol·|b|`.
pub fn allclose(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) -> bool {
    a.shape() == b.shape() && a.to_vec::<f64>().iter().zip(b.to_vec::<f64>()).all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs())
}
