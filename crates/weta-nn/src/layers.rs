//! The standard layers. Each is a small struct holding its parameters, with
//! the same name, constructor arguments and default initialisation as in
//! PyTorch. Optional settings are builder methods: PyTorch's
//! `nn.Conv2d(1, 16, 3, padding=1)` is `nn::Conv2d::new(1, 16, 3).padding(1)`.

use weta_autograd::Tensor;
use weta_core::DType;

use crate::functional as F;
use crate::module::{join_name, Module, TensorKind, Visit};

fn param(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape).requires_grad_(true)
}

/// PyTorch's default for `Linear` and `Conv2d`: `kaiming_uniform_(a=√5)` on
/// the weight, which works out to `U(−1/√fan_in, 1/√fan_in)`, and the same
/// range for the bias.
fn default_init(weight: &Tensor, bias: Option<&Tensor>) {
    let (fan_in, _) = crate::init::fans(weight);
    let bound = 1.0 / (fan_in as f64).sqrt();
    crate::init::uniform_(weight, -bound, bound);
    if let Some(b) = bias {
        crate::init::uniform_(b, -bound, bound);
    }
}

macro_rules! visit_fields {
    ($($field:ident),*) => {
        fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
            $(Visit::visit_tensors(&self.$field, &join_name(prefix, stringify!($field)), f);)*
        }
        fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
            $(Visit::visit_tensors_mut(&mut self.$field, &join_name(prefix, stringify!($field)), f);)*
        }
    };
}

macro_rules! no_tensors {
    () => {
        fn visit(&self, _: &str, _: &mut dyn FnMut(&str, &Tensor, TensorKind)) {}
        fn visit_mut(&mut self, _: &str, _: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {}
    };
}

// ---------------------------------------------------------------------------

/// `y = x Wᵀ + b`, with `W: [out, in]`.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let l = Linear { weight: param(&[out_features, in_features]), bias: Some(param(&[out_features])) };
        default_init(&l.weight, l.bias.as_ref());
        l
    }

    /// Drops (`false`) or keeps the bias, as PyTorch's `bias=` argument.
    pub fn bias(mut self, bias: bool) -> Self {
        if !bias {
            self.bias = None;
        }
        self
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::linear(x, &self.weight, self.bias.as_ref())
    }
    visit_fields!(weight, bias);
    fn describe(&self) -> Option<String> {
        Some("linear".into())
    }
}

/// 2-D convolution over `[N, C, H, W]` images, with weight `[O, C, k, k]`.
pub struct Conv2d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let c = Conv2d {
            weight: param(&[out_channels, in_channels, kernel_size, kernel_size]),
            bias: Some(param(&[out_channels])),
            stride: 1,
            padding: 0,
            dilation: 1,
        };
        default_init(&c.weight, c.bias.as_ref());
        c
    }
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }
    pub fn padding(mut self, p: usize) -> Self {
        self.padding = p;
        self
    }
    pub fn dilation(mut self, d: usize) -> Self {
        self.dilation = d;
        self
    }
    pub fn bias(mut self, bias: bool) -> Self {
        if !bias {
            self.bias = None;
        }
        self
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (s, p, d) = (self.stride, self.padding, self.dilation);
        F::conv2d_ext(x, &self.weight, self.bias.as_ref(), (s, s), (p, p), (d, d))
    }
    visit_fields!(weight, bias);
    fn describe(&self) -> Option<String> {
        Some(format!("conv2d stride={} padding={} dilation={}", self.stride, self.padding, self.dilation))
    }
}

/// Max over `k × k` windows; the stride defaults to `k`.
pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize) -> Self {
        MaxPool2d { kernel_size, stride: kernel_size }
    }
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::max_pool2d(x, self.kernel_size, self.stride)
    }
    no_tensors!();
    fn describe(&self) -> Option<String> {
        Some(format!("maxpool2d kernel={} stride={}", self.kernel_size, self.stride))
    }
}

/// Mean over `k × k` windows; the stride defaults to `k`.
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        AvgPool2d { kernel_size, stride: kernel_size, padding: 0 }
    }
    pub fn stride(mut self, s: usize) -> Self {
        self.stride = s;
        self
    }
    pub fn padding(mut self, p: usize) -> Self {
        self.padding = p;
        self
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::avg_pool2d(x, self.kernel_size, self.stride, self.padding)
    }
    no_tensors!();
    fn describe(&self) -> Option<String> {
        Some(format!("avgpool2d kernel={} stride={} padding={}", self.kernel_size, self.stride, self.padding))
    }
}

/// Flattens dims `start_dim..=end_dim` (by default all but the batch dim).
pub struct Flatten {
    pub start_dim: isize,
    pub end_dim: isize,
}

impl Default for Flatten {
    fn default() -> Self {
        Flatten { start_dim: 1, end_dim: -1 }
    }
}

impl Flatten {
    pub fn new(start_dim: isize, end_dim: isize) -> Self {
        Flatten { start_dim, end_dim }
    }
}

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.flatten(self.start_dim, self.end_dim)
    }
    no_tensors!();
    fn describe(&self) -> Option<String> {
        Some(format!("flatten start={} end={}", self.start_dim, self.end_dim))
    }
}

/// Batch normalisation (Ioffe & Szegedy, 2015) over channels: dim 1 of
/// `[N, C]` (`BatchNorm1d`) or `[N, C, H, W]` (`BatchNorm2d`).
pub struct BatchNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub num_batches_tracked: Tensor,
    pub eps: f64,
    pub momentum: f64,
    pub training: bool,
    ndim: usize,
}

pub type BatchNorm1d = BatchNorm;
pub type BatchNorm2d = BatchNorm;

impl BatchNorm {
    fn with_ndim(num_features: usize, ndim: usize) -> Self {
        BatchNorm {
            weight: Some(Tensor::ones(&[num_features]).requires_grad_(true)),
            bias: Some(param(&[num_features])),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            num_batches_tracked: Tensor::zeros_on(&[], DType::I64, Default::default()),
            eps: 1e-5,
            momentum: 0.1,
            training: true,
            ndim,
        }
    }

    /// For `[N, C, H, W]` inputs (`nn.BatchNorm2d`).
    pub fn new(num_features: usize) -> Self {
        Self::with_ndim(num_features, 4)
    }

    /// For `[N, C]` inputs (`nn.BatchNorm1d`).
    pub fn new_1d(num_features: usize) -> Self {
        Self::with_ndim(num_features, 2)
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = m;
        self
    }
    /// Without learnable scale and shift (`affine=False`).
    pub fn affine(mut self, affine: bool) -> Self {
        if !affine {
            self.weight = None;
            self.bias = None;
        }
        self
    }
}

impl Module for BatchNorm {
    #[track_caller]
    fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), self.ndim, "BatchNorm: expected a {}-d input, got {:?}", self.ndim, x.shape());
        if self.training {
            self.num_batches_tracked.data().add_scalar_(1.0);
        }
        F::batch_norm(
            x,
            Some(&self.running_mean),
            Some(&self.running_var),
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.training,
            self.momentum,
            self.eps,
        )
    }

    fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        self.weight.visit_tensors(&join_name(prefix, "weight"), f);
        self.bias.visit_tensors(&join_name(prefix, "bias"), f);
        f(&join_name(prefix, "running_mean"), &self.running_mean, TensorKind::Buffer);
        f(&join_name(prefix, "running_var"), &self.running_var, TensorKind::Buffer);
        f(&join_name(prefix, "num_batches_tracked"), &self.num_batches_tracked, TensorKind::Buffer);
    }

    fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        self.weight.visit_tensors_mut(&join_name(prefix, "weight"), f);
        self.bias.visit_tensors_mut(&join_name(prefix, "bias"), f);
        f(&join_name(prefix, "running_mean"), &mut self.running_mean, TensorKind::Buffer);
        f(&join_name(prefix, "running_var"), &mut self.running_var, TensorKind::Buffer);
        f(&join_name(prefix, "num_batches_tracked"), &mut self.num_batches_tracked, TensorKind::Buffer);
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn describe(&self) -> Option<String> {
        Some(format!("batchnorm eps={}", self.eps))
    }
}

/// Layer normalisation (Ba et al., 2016) over the trailing dims in `shape`.
pub struct LayerNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub normalized_shape: Vec<usize>,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(normalized_shape: &[usize]) -> Self {
        LayerNorm {
            weight: Some(Tensor::ones(normalized_shape).requires_grad_(true)),
            bias: Some(param(normalized_shape)),
            normalized_shape: normalized_shape.to_vec(),
            eps: 1e-5,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::layer_norm(x, self.normalized_shape.len(), self.weight.as_ref(), self.bias.as_ref(), self.eps)
    }
    visit_fields!(weight, bias);
    fn describe(&self) -> Option<String> {
        Some(format!("layernorm dims={} eps={}", self.normalized_shape.len(), self.eps))
    }
}

/// Randomly zeroes elements with probability `p` while training.
pub struct Dropout {
    pub p: f64,
    pub training: bool,
}

impl Dropout {
    #[track_caller]
    pub fn new(p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "dropout probability has to be between 0 and 1, but got {p}");
        Dropout { p, training: true }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::dropout(x, self.p, self.training)
    }
    no_tensors!();
    fn describe(&self) -> Option<String> {
        Some("identity".into())
    }
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

macro_rules! activation {
    ($($(#[$m:meta])* $name:ident => $f:expr, $desc:literal),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;
        impl Module for $name {
            fn forward(&self, x: &Tensor) -> Tensor {
                let f: fn(&Tensor) -> Tensor = $f;
                f(x)
            }
            no_tensors!();
            fn describe(&self) -> Option<String> {
                Some($desc.into())
            }
        }
    )*};
}

activation! {
    /// `max(0, x)`.
    ReLU => F::relu, "relu",
    Tanh => F::tanh, "tanh",
    Sigmoid => F::sigmoid, "sigmoid",
    /// Exact GELU, `x Φ(x)`.
    GELU => F::gelu, "gelu",
    /// `x σ(x)`.
    SiLU => F::silu, "silu",
    /// Passes its input through unchanged.
    Identity => |x| x.clone(), "identity",
}

/// `max(0, x) + slope · min(0, x)`.
#[derive(Clone, Copy, Debug)]
pub struct LeakyReLU {
    pub negative_slope: f64,
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        LeakyReLU { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        LeakyReLU { negative_slope: 0.01 }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        F::leaky_relu(x, self.negative_slope)
    }
    no_tensors!();
    fn describe(&self) -> Option<String> {
        Some(format!("leakyrelu slope={}", self.negative_slope))
    }
}

/// Softmax over a dim, as a module.
#[derive(Clone, Copy, Debug)]
pub struct Softmax {
    pub dim: isize,
}

impl Softmax {
    pub fn new(dim: isize) -> Self {
        Softmax { dim }
    }
}

impl Module for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.softmax(self.dim)
    }
    no_tensors!();
    fn describe(&self) -> Option<String> {
        Some(format!("softmax dim={}", self.dim))
    }
}

// ---------------------------------------------------------------------------
// Containers
// ---------------------------------------------------------------------------

/// Layers applied in order. Parameter names are indices: `0.weight`, ...
#[derive(Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new() -> Self {
        Sequential::default()
    }

    /// Appends a layer (builder style).
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, m: M) -> Self {
        self.layers.push(Box::new(m));
        self
    }

    pub fn push<M: Module + 'static>(&mut self, m: M) {
        self.layers.push(Box::new(m));
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&dyn Module> {
        self.layers.get(i).map(|b| b.as_ref())
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.layers.iter().fold(x.clone(), |h, l| l.forward(&h))
    }
    fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        self.layers.visit_tensors(prefix, f);
    }
    fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        self.layers.visit_tensors_mut(prefix, f);
    }
    fn set_training(&mut self, training: bool) {
        self.layers.set_train_mode(training);
    }
    /// One line per layer; layer `i`'s weights are under `i.`. Nested
    /// containers aren't describable.
    fn describe(&self) -> Option<String> {
        let lines: Option<Vec<String>> = self.layers.iter().map(|l| l.describe().filter(|d| !d.contains('\n'))).collect();
        lines.map(|l| l.join("\n"))
    }
}

/// A list of modules; you decide how to call them in your own `forward`.
#[derive(Default)]
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new() -> Self {
        ModuleList::default()
    }
    pub fn push<M: Module + 'static>(&mut self, m: M) {
        self.modules.push(Box::new(m));
    }
    pub fn len(&self) -> usize {
        self.modules.len()
    }
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = &dyn Module> {
        self.modules.iter().map(|b| b.as_ref())
    }
}

impl std::ops::Index<usize> for ModuleList {
    type Output = dyn Module;
    fn index(&self, i: usize) -> &(dyn Module + 'static) {
        self.modules[i].as_ref()
    }
}

impl Module for ModuleList {
    fn forward(&self, _: &Tensor) -> Tensor {
        panic!("ModuleList has no forward; iterate over it in your own forward")
    }
    fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        self.modules.visit_tensors(prefix, f);
    }
    fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        self.modules.visit_tensors_mut(prefix, f);
    }
    fn set_training(&mut self, training: bool) {
        self.modules.set_train_mode(training);
    }
}

/// Named modules, in insertion order.
#[derive(Default)]
pub struct ModuleDict {
    modules: Vec<(String, Box<dyn Module>)>,
}

impl ModuleDict {
    pub fn new() -> Self {
        ModuleDict::default()
    }
    pub fn insert<M: Module + 'static>(&mut self, name: impl Into<String>, m: M) {
        self.modules.push((name.into(), Box::new(m)));
    }
    pub fn get(&self, name: &str) -> Option<&dyn Module> {
        self.modules.iter().find(|(n, _)| n == name).map(|(_, m)| m.as_ref())
    }
}

impl Module for ModuleDict {
    fn forward(&self, _: &Tensor) -> Tensor {
        panic!("ModuleDict has no forward; look modules up in your own forward")
    }
    fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        for (n, m) in &self.modules {
            m.visit(&join_name(prefix, n), f);
        }
    }
    fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut Tensor, TensorKind)) {
        for (n, m) in &mut self.modules {
            m.visit_mut(&join_name(prefix, n), f);
        }
    }
    fn set_training(&mut self, training: bool) {
        for (_, m) in &mut self.modules {
            m.set_training(training);
        }
    }
}
