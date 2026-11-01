//! The primitive ops a backend has to implement. Everything else in weta
//! (softmax, layer norm, conv2d, pooling, losses) is built from these.

/// Element-wise ops on one tensor. The ones with parameters let scalar
/// arithmetic like `x * 2.0 + 1.0` run as one kernel with no extra tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Abs,
    /// -1, 0 or 1.
    Sign,
    Relu,
    /// 1 where x > 0, else 0: the derivative of ReLU.
    Step,
    Tanh,
    Sigmoid,
    Erf,
    Sin,
    Cos,
    Recip,
    /// `x * mul + add`.
    Affine {
        mul: f64,
        add: f64,
    },
    /// `x ^ e`.
    Powf(f64),
    /// 1 where x > v, else 0.
    GtScalar(f64),
    /// `max(x, v)`, `min(x, v)`.
    MaxScalar(f64),
    MinScalar(f64),
}

impl UnaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Abs => "abs",
            UnaryOp::Sign => "sign",
            UnaryOp::Relu => "relu",
            UnaryOp::Step => "step",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Erf => "erf",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Recip => "reciprocal",
            UnaryOp::Affine { .. } => "affine",
            UnaryOp::Powf(_) => "pow",
            UnaryOp::GtScalar(_) => "gt",
            UnaryOp::MaxScalar(_) => "clamp_min",
            UnaryOp::MinScalar(_) => "clamp_max",
        }
    }
}

/// Element-wise ops on two tensors of the same shape (broadcasting is done
/// beforehand, by giving the inputs stride-0 layouts). Comparisons return
/// 1 or 0 in the input dtype, as masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
            BinaryOp::Eq => "eq",
            BinaryOp::Ne => "ne",
            BinaryOp::Lt => "lt",
            BinaryOp::Le => "le",
            BinaryOp::Gt => "gt",
            BinaryOp::Ge => "ge",
        }
    }
}

/// Reductions over one axis. `ArgMax`/`ArgMin` return `I64` indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    ArgMax,
    ArgMin,
}

/// Parameters for `im2col`/`col2im`, the two kernels convolution and
/// pooling are built on.
///
/// `im2col` turns an image `[N, C, H, W]` into columns
/// `[N, C·kh·kw, OH·OW]`: each column holds the pixels one kernel position
/// sees. Convolution is then a single matrix multiply with the weights
/// reshaped to `[O, C·kh·kw]`. `col2im` is its adjoint (it sums overlapping
/// patches back into an image), which makes it the backward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Im2ColParams {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Im2ColParams {
    pub fn new(kernel: (usize, usize)) -> Self {
        Im2ColParams { kernel, stride: (1, 1), padding: (0, 0), dilation: (1, 1) }
    }

    /// Output spatial size: `⌊(H + 2p − d(k − 1) − 1) / s⌋ + 1`.
    pub fn out_size(&self, h: usize, w: usize) -> (usize, usize) {
        let f = |x: usize, k: usize, s: usize, p: usize, d: usize| {
            let span = d * (k - 1) + 1;
            if x + 2 * p < span {
                0
            } else {
                (x + 2 * p - span) / s + 1
            }
        };
        (
            f(h, self.kernel.0, self.stride.0, self.padding.0, self.dilation.0),
            f(w, self.kernel.1, self.stride.1, self.padding.1, self.dilation.1),
        )
    }
}

/// What [`Backend::random`](crate::Backend::random) draws from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distribution {
    Uniform { lo: f64, hi: f64 },
    Normal { mean: f64, std: f64 },
}

/// A counter-based random number generator: element `i` of a random tensor
/// is a pure function of `(seed, i)`, so any backend can fill elements in
/// parallel, in any order, and get the same numbers as every other backend.
/// The mixing function is SplitMix64's finaliser (Steele et al., 2014);
/// `kernels.rs` in `weta-cuda` has the same code in CUDA C.
pub fn splitmix64(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_add(i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[0, 1)` with 53 random bits.
fn unit(seed: u64, i: u64) -> f64 {
    (splitmix64(seed, i) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

impl Distribution {
    /// Element `i` of a random tensor drawn with `seed`. Normals use the
    /// Box–Muller transform on the uniforms at counters `2i` and `2i + 1`.
    pub fn sample(self, seed: u64, i: u64) -> f64 {
        match self {
            Distribution::Uniform { lo, hi } => lo + (hi - lo) * unit(seed, i),
            Distribution::Normal { mean, std } => {
                let u1 = 1.0 - unit(seed, 2 * i); // (0, 1], so ln is finite
                let u2 = unit(seed, 2 * i + 1);
                mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }
        }
    }
}
