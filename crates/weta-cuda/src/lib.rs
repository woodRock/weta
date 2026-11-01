//! # weta-cuda
//!
//! The CUDA backend, through `cudarc`. Element-wise kernels, reductions,
//! copies and im2col are hand-written (see [`kernels`]) and compiled at
//! runtime with NVRTC; matrix multiplication goes through cuBLAS. The CUDA
//! libraries are loaded when first used, so building this crate needs no
//! CUDA toolkit, and `weta` without the `cuda` feature never touches it.
//!
//! Like the CPU backend, every kernel reads strided views directly. The
//! backend-parity tests check each kernel against the CPU reference.

mod kernels;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::{sys::cublasOperation_t, CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use weta_core::{Backend, BinaryOp, DType, Device, Error, HostData, Im2ColParams, Layout, ReduceOp, Result, UnaryOp, View};

/// Data on one GPU.
pub enum CudaData {
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
    I64(CudaSlice<i64>),
}

pub struct CudaStorage {
    data: CudaData,
    ordinal: usize,
}

impl CudaStorage {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
    pub fn dtype(&self) -> DType {
        match self.data {
            CudaData::F32(_) => DType::F32,
            CudaData::F64(_) => DType::F64,
            CudaData::I64(_) => DType::I64,
        }
    }
}

/// A strided layout as the kernels see it (must match `Strides` in the
/// CUDA source).
#[repr(C)]
#[derive(Clone, Copy)]
struct Strides {
    ndim: i32,
    pad: i32,
    shape: [i64; 8],
    stride: [i64; 8],
    offset: i64,
}

unsafe impl DeviceRepr for Strides {}

impl Strides {
    fn of(l: &Layout) -> Strides {
        assert!(l.ndim() <= 8, "the CUDA backend supports at most 8 dims, got shape {:?}", l.shape());
        let mut s = Strides { ndim: l.ndim() as i32, pad: 0, shape: [1; 8], stride: [0; 8], offset: l.offset() as i64 };
        for i in 0..l.ndim() {
            s.shape[i] = l.shape()[i] as i64;
            s.stride[i] = l.strides()[i] as i64;
        }
        s
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvP {
    n: i64,
    c: i64,
    h: i64,
    w: i64,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    ph: i64,
    pw: i64,
    dh: i64,
    dw: i64,
    oh: i64,
    ow: i64,
}

unsafe impl DeviceRepr for ConvP {}

impl ConvP {
    fn new(p: Im2ColParams, (n, c, h, w): (usize, usize, usize, usize)) -> ConvP {
        let (oh, ow) = p.out_size(h, w);
        let i = |v: usize| v as i64;
        ConvP {
            n: i(n),
            c: i(c),
            h: i(h),
            w: i(w),
            kh: i(p.kernel.0),
            kw: i(p.kernel.1),
            sh: i(p.stride.0),
            sw: i(p.stride.1),
            ph: i(p.padding.0),
            pw: i(p.padding.1),
            dh: i(p.dilation.0),
            dw: i(p.dilation.1),
            oh: i(oh),
            ow: i(ow),
        }
    }
}

pub struct CudaBackend {
    ordinal: usize,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    funcs: HashMap<String, CudaFunction>,
}

static BACKENDS: OnceLock<Mutex<HashMap<usize, &'static CudaBackend>>> = OnceLock::new();

fn cuda_err(e: impl std::fmt::Debug) -> Error {
    Error::DeviceUnavailable(format!("CUDA: {e:?}"))
}

/// The backend for GPU `ordinal`, created (and its kernels compiled) on
/// first use and kept for the life of the process.
pub fn backend(ordinal: usize) -> Result<&'static CudaBackend> {
    let map = BACKENDS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = map.get(&ordinal) {
        return Ok(b);
    }
    let b: &'static CudaBackend = Box::leak(Box::new(CudaBackend::new(ordinal)?));
    map.insert(ordinal, b);
    Ok(b)
}

/// The number of CUDA devices, or 0 if the driver can't be loaded.
pub fn device_count() -> usize {
    // cudarc panics if libcuda can't be loaded; treat that as no GPU.
    std::panic::catch_unwind(|| CudaContext::device_count().map(|n| n.max(0) as usize).unwrap_or(0)).unwrap_or(0)
}

pub fn is_available() -> bool {
    device_count() > 0
}

impl CudaBackend {
    fn new(ordinal: usize) -> Result<CudaBackend> {
        if ordinal >= device_count() {
            return Err(Error::DeviceUnavailable(format!("no CUDA device {ordinal} (found {})", device_count())));
        }
        let ctx = CudaContext::new(ordinal).map_err(cuda_err)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_err)?;
        let ptx = cudarc::nvrtc::compile_ptx(kernels::source()).map_err(cuda_err)?;
        let module = ctx.load_module(ptx).map_err(cuda_err)?;
        let mut funcs = HashMap::new();
        for name in kernels::names() {
            let f = module.load_function(&name).map_err(cuda_err)?;
            funcs.insert(name, f);
        }
        Ok(CudaBackend { ordinal, stream, blas, funcs })
    }

    fn func(&self, name: &str) -> &CudaFunction {
        self.funcs.get(name).unwrap_or_else(|| panic!("CUDA kernel {name} not found"))
    }

    fn wrap(&self, data: CudaData) -> CudaStorage {
        CudaStorage { data, ordinal: self.ordinal }
    }

    fn zeros_of(&self, dtype: DType, n: usize) -> CudaData {
        // cudarc can't allocate zero bytes; keep one element instead.
        let n = n.max(1);
        let r = match dtype {
            DType::F32 => self.stream.alloc_zeros::<f32>(n).map(CudaData::F32),
            DType::F64 => self.stream.alloc_zeros::<f64>(n).map(CudaData::F64),
            DType::I64 => self.stream.alloc_zeros::<i64>(n).map(CudaData::I64),
        };
        r.expect("CUDA allocation failed")
    }
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

fn sfx(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I64 => "i64",
    }
}

fn unary_code(op: UnaryOp) -> (i32, f64, f64) {
    match op {
        UnaryOp::Neg => (0, 0.0, 0.0),
        UnaryOp::Exp => (1, 0.0, 0.0),
        UnaryOp::Log => (2, 0.0, 0.0),
        UnaryOp::Sqrt => (3, 0.0, 0.0),
        UnaryOp::Abs => (4, 0.0, 0.0),
        UnaryOp::Sign => (5, 0.0, 0.0),
        UnaryOp::Relu => (6, 0.0, 0.0),
        UnaryOp::Step => (7, 0.0, 0.0),
        UnaryOp::Tanh => (8, 0.0, 0.0),
        UnaryOp::Sigmoid => (9, 0.0, 0.0),
        UnaryOp::Erf => (10, 0.0, 0.0),
        UnaryOp::Sin => (11, 0.0, 0.0),
        UnaryOp::Cos => (12, 0.0, 0.0),
        UnaryOp::Recip => (13, 0.0, 0.0),
        UnaryOp::Affine { mul, add } => (14, mul, add),
        UnaryOp::Powf(e) => (15, e, 0.0),
        UnaryOp::GtScalar(v) => (16, v, 0.0),
        UnaryOp::MaxScalar(v) => (17, v, 0.0),
        UnaryOp::MinScalar(v) => (18, v, 0.0),
    }
}

fn binary_code(op: BinaryOp) -> i32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Pow => 4,
        BinaryOp::Maximum => 5,
        BinaryOp::Minimum => 6,
        BinaryOp::Eq => 7,
        BinaryOp::Ne => 8,
        BinaryOp::Lt => 9,
        BinaryOp::Le => 10,
        BinaryOp::Gt => 11,
        BinaryOp::Ge => 12,
    }
}

/// Runs `$body` with `$s` bound to the typed slice and `$out` to a fresh
/// output slice of the same type with `$n` elements.
macro_rules! typed {
    ($self:ident, $data:expr, $n:expr, |$s:ident, $out:ident| $body:expr) => {
        match $data {
            CudaData::F32($s) => {
                let mut $out = $self.stream.alloc_zeros::<f32>($n.max(1)).expect("CUDA alloc");
                $body;
                CudaData::F32($out)
            }
            CudaData::F64($s) => {
                let mut $out = $self.stream.alloc_zeros::<f64>($n.max(1)).expect("CUDA alloc");
                $body;
                CudaData::F64($out)
            }
            CudaData::I64($s) => {
                let mut $out = $self.stream.alloc_zeros::<i64>($n.max(1)).expect("CUDA alloc");
                $body;
                CudaData::I64($out)
            }
        }
    };
}

impl CudaBackend {
    /// A contiguous copy of a view (same dtype).
    fn contiguous(&self, x: View<'_, CudaStorage>) -> CudaStorage {
        self.cast(x, x.dtype)
    }

    /// Batched GEMM operand: `(transpose?, leading dim, batch stride)` if the
    /// row-major `[B, R, C]` view can be fed to cuBLAS without a copy.
    fn gemm_operand(l: &Layout) -> Option<(bool, i32, i64)> {
        let (r, c) = (l.shape()[1], l.shape()[2]);
        let (sr, sc) = (l.strides()[1], l.strides()[2]);
        // cuBLAS is column-major: a row-major R×C matrix with unit column
        // stride is its C×R transpose with leading dimension sr.
        if sc == 1 || c == 1 {
            let ld = if r == 1 { c.max(1) } else { sr };
            if ld >= c.max(1) {
                return Some((false, ld as i32, l.strides()[0] as i64));
            }
        }
        if sr == 1 || r == 1 {
            let ld = if c == 1 { r.max(1) } else { sc };
            if ld >= r.max(1) {
                return Some((true, ld as i32, l.strides()[0] as i64));
            }
        }
        None
    }

    fn matmul_t<T>(&self, a: &CudaSlice<T>, la: &Layout, b: &CudaSlice<T>, lb: &Layout) -> CudaSlice<T>
    where
        CudaBlas: Gemm<T>,
        T: DeviceRepr + cudarc::driver::ValidAsZeroBits + num_one::One,
    {
        let (bs, m, k) = (la.shape()[0], la.shape()[1], la.shape()[2]);
        let n = lb.shape()[2];
        let mut out = self.stream.alloc_zeros::<T>((bs * m * n).max(1)).expect("CUDA alloc");
        if bs * m * n == 0 {
            return out;
        }
        let (ta, lda_b, sa) = Self::gemm_operand(la).expect("checked by caller");
        let (tb, ldb_b, sb) = Self::gemm_operand(lb).expect("checked by caller");
        let op = |t: bool| if t { cublasOperation_t::CUBLAS_OP_T } else { cublasOperation_t::CUBLAS_OP_N };
        // Row-major C = A·B is column-major Cᵀ = Bᵀ·Aᵀ, so B goes first.
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: op(tb),
                transb: op(ta),
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: T::one(),
                lda: ldb_b,
                ldb: lda_b,
                beta: T::zero_value(),
                ldc: n as i32,
            },
            batch_size: bs as i32,
            stride_a: sb,
            stride_b: sa,
            stride_c: (m * n) as i64,
        };
        let av = a.slice(la.offset()..);
        let bv = b.slice(lb.offset()..);
        unsafe { self.blas.gemm_strided_batched(cfg, &bv, &av, &mut out) }.expect("cuBLAS gemm failed");
        out
    }
}

/// `1` and `0` for the GEMM scalars.
mod num_one {
    pub trait One {
        fn one() -> Self;
        fn zero_value() -> Self;
    }
    impl One for f32 {
        fn one() -> f32 {
            1.0
        }
        fn zero_value() -> f32 {
            0.0
        }
    }
    impl One for f64 {
        fn one() -> f64 {
            1.0
        }
        fn zero_value() -> f64 {
            0.0
        }
    }
}

impl Backend for CudaBackend {
    type Storage = CudaStorage;

    fn device(&self) -> Device {
        Device::Cuda(self.ordinal)
    }

    fn from_host(&self, data: HostData) -> CudaStorage {
        let r = match data {
            HostData::F32(v) if v.is_empty() => Ok(self.zeros_of(DType::F32, 0)),
            HostData::F64(v) if v.is_empty() => Ok(self.zeros_of(DType::F64, 0)),
            HostData::I64(v) if v.is_empty() => Ok(self.zeros_of(DType::I64, 0)),
            HostData::F32(v) => self.stream.clone_htod(&v).map(CudaData::F32),
            HostData::F64(v) => self.stream.clone_htod(&v).map(CudaData::F64),
            HostData::I64(v) => self.stream.clone_htod(&v).map(CudaData::I64),
        };
        self.wrap(r.expect("CUDA host-to-device copy failed"))
    }

    fn to_host(&self, x: View<'_, CudaStorage>) -> HostData {
        let n = x.layout.numel();
        let c = self.contiguous(x);
        let r = match &c.data {
            CudaData::F32(s) => self.stream.clone_dtoh(s).map(|mut v| {
                v.truncate(n);
                HostData::F32(v)
            }),
            CudaData::F64(s) => self.stream.clone_dtoh(s).map(|mut v| {
                v.truncate(n);
                HostData::F64(v)
            }),
            CudaData::I64(s) => self.stream.clone_dtoh(s).map(|mut v| {
                v.truncate(n);
                HostData::I64(v)
            }),
        };
        r.expect("CUDA device-to-host copy failed")
    }

    fn zeros(&self, len: usize, dtype: DType) -> CudaStorage {
        let d = self.zeros_of(dtype, len);
        self.wrap(d)
    }

    fn copy_strided(&self, src: View<'_, CudaStorage>, dst: &mut CudaStorage, dst_layout: &Layout) {
        let n = src.layout.numel();
        if n == 0 {
            return;
        }
        let (ss, ds) = (Strides::of(src.layout), Strides::of(dst_layout));
        let n64 = n as i64;
        let f = self.func(&format!("copy_{}", sfx(src.dtype)));
        let mut b = self.stream.launch_builder(f);
        match (&src.storage.data, &mut dst.data) {
            (CudaData::F32(s), CudaData::F32(d)) => unsafe { b.arg(d).arg(&ds).arg(s).arg(&ss).arg(&n64).launch(cfg(n)) },
            (CudaData::F64(s), CudaData::F64(d)) => unsafe { b.arg(d).arg(&ds).arg(s).arg(&ss).arg(&n64).launch(cfg(n)) },
            (CudaData::I64(s), CudaData::I64(d)) => unsafe { b.arg(d).arg(&ds).arg(s).arg(&ss).arg(&n64).launch(cfg(n)) },
            _ => panic!("copy: dtype mismatch"),
        }
        .expect("CUDA copy kernel failed");
    }

    fn unary(&self, op: UnaryOp, x: View<'_, CudaStorage>) -> CudaStorage {
        let n = x.layout.numel();
        let (code, p1, p2) = unary_code(op);
        let xs = Strides::of(x.layout);
        let n64 = n as i64;
        let f = self.func(&format!("unary_{}", sfx(x.dtype)));
        let d = typed!(self, &x.storage.data, n, |s, out| if n > 0 {
            unsafe { self.stream.launch_builder(f).arg(&mut out).arg(s).arg(&xs).arg(&n64).arg(&code).arg(&p1).arg(&p2).launch(cfg(n)) }
                .expect("CUDA unary kernel failed");
        });
        self.wrap(d)
    }

    fn binary(&self, op: BinaryOp, a: View<'_, CudaStorage>, b: View<'_, CudaStorage>) -> CudaStorage {
        let n = a.layout.numel();
        let (sa, sb) = (Strides::of(a.layout), Strides::of(b.layout));
        let (code, n64) = (binary_code(op), n as i64);
        let f = self.func(&format!("binary_{}", sfx(a.dtype)));
        macro_rules! go {
            ($x:expr, $y:expr, $v:ident, $t:ty) => {{
                let mut out = self.stream.alloc_zeros::<$t>(n.max(1)).expect("CUDA alloc");
                if n > 0 {
                    unsafe {
                        self.stream.launch_builder(f).arg(&mut out).arg($x).arg(&sa).arg($y).arg(&sb).arg(&n64).arg(&code).launch(cfg(n))
                    }
                    .expect("CUDA binary kernel failed");
                }
                CudaData::$v(out)
            }};
        }
        let d = match (&a.storage.data, &b.storage.data) {
            (CudaData::F32(x), CudaData::F32(y)) => go!(x, y, F32, f32),
            (CudaData::F64(x), CudaData::F64(y)) => go!(x, y, F64, f64),
            (CudaData::I64(x), CudaData::I64(y)) => go!(x, y, I64, i64),
            _ => panic!("{}: dtype mismatch", op.name()),
        };
        self.wrap(d)
    }

    fn reduce(&self, op: ReduceOp, x: View<'_, CudaStorage>, axis: usize) -> CudaStorage {
        let mut shape = x.layout.shape().to_vec();
        let mut strides = x.layout.strides().to_vec();
        let (len, stride) = (shape.remove(axis) as i64, strides.remove(axis) as i64);
        let outer = Layout::new(shape, strides, x.layout.offset());
        let n = outer.numel();
        let (os, n64) = (Strides::of(&outer), n as i64);
        let (arg, code) = match op {
            ReduceOp::Sum => (false, 0i32),
            ReduceOp::Max => (false, 1),
            ReduceOp::Min => (false, 2),
            ReduceOp::ArgMax => (true, 0),
            ReduceOp::ArgMin => (true, 1),
        };
        let name = format!("{}reduce_{}", if arg { "arg" } else { "" }, sfx(x.dtype));
        let f = self.func(&name);
        if arg {
            let mut out = self.stream.alloc_zeros::<i64>(n.max(1)).expect("CUDA alloc");
            if n > 0 {
                let mut b = self.stream.launch_builder(f);
                b.arg(&mut out);
                match &x.storage.data {
                    CudaData::F32(s) => b.arg(s),
                    CudaData::F64(s) => b.arg(s),
                    CudaData::I64(s) => b.arg(s),
                };
                unsafe { b.arg(&os).arg(&len).arg(&stride).arg(&n64).arg(&code).launch(cfg(n)) }.expect("CUDA argreduce failed");
            }
            return self.wrap(CudaData::I64(out));
        }
        let d = typed!(self, &x.storage.data, n, |s, out| if n > 0 {
            unsafe {
                self.stream.launch_builder(f).arg(&mut out).arg(s).arg(&os).arg(&len).arg(&stride).arg(&n64).arg(&code).launch(cfg(n))
            }
            .expect("CUDA reduce kernel failed");
        });
        self.wrap(d)
    }

    fn matmul(&self, a: View<'_, CudaStorage>, b: View<'_, CudaStorage>) -> CudaStorage {
        // Operands cuBLAS can't read in place are copied to contiguous first.
        let fix = |v: View<'_, CudaStorage>| -> Option<(CudaStorage, Layout)> {
            if Self::gemm_operand(v.layout).is_some() {
                None
            } else {
                Some((self.contiguous(v), Layout::contiguous(v.layout.shape())))
            }
        };
        let (ca, cb) = (fix(a), fix(b));
        let (sa, la) = match &ca {
            Some((s, l)) => (s, l),
            None => (a.storage, a.layout),
        };
        let (sb, lb) = match &cb {
            Some((s, l)) => (s, l),
            None => (b.storage, b.layout),
        };
        let d = match (&sa.data, &sb.data) {
            (CudaData::F32(x), CudaData::F32(y)) => CudaData::F32(self.matmul_t(x, la, y, lb)),
            (CudaData::F64(x), CudaData::F64(y)) => CudaData::F64(self.matmul_t(x, la, y, lb)),
            _ => panic!("matmul: unsupported dtypes on CUDA"),
        };
        self.wrap(d)
    }

    fn im2col(&self, x: View<'_, CudaStorage>, p: Im2ColParams) -> CudaStorage {
        let s = x.layout.shape();
        let cp = ConvP::new(p, (s[0], s[1], s[2], s[3]));
        let n = (cp.n * cp.c * cp.kh * cp.kw * cp.oh * cp.ow) as usize;
        let (xs, n64) = (Strides::of(x.layout), n as i64);
        let f = self.func(&format!("im2col_{}", sfx(x.dtype)));
        let d = typed!(self, &x.storage.data, n, |s, out| if n > 0 {
            unsafe { self.stream.launch_builder(f).arg(&mut out).arg(s).arg(&xs).arg(&cp).arg(&n64).launch(cfg(n)) }
                .expect("CUDA im2col failed");
        });
        self.wrap(d)
    }

    fn col2im(&self, cols: View<'_, CudaStorage>, p: Im2ColParams, (c, h, w): (usize, usize, usize)) -> CudaStorage {
        let nb = cols.layout.shape()[0];
        let cp = ConvP::new(p, (nb, c, h, w));
        let n = nb * c * h * w;
        let cont = self.contiguous(cols);
        let n64 = n as i64;
        let f = self.func(&format!("col2im_{}", sfx(cols.dtype)));
        let d = typed!(self, &cont.data, n, |s, out| if n > 0 {
            unsafe { self.stream.launch_builder(f).arg(&mut out).arg(s).arg(&cp).arg(&n64).launch(cfg(n)) }.expect("CUDA col2im failed");
        });
        self.wrap(d)
    }

    fn cast(&self, x: View<'_, CudaStorage>, dtype: DType) -> CudaStorage {
        let n = x.layout.numel();
        let (xs, n64) = (Strides::of(x.layout), n as i64);
        let f = self.func(&format!("cast_{}_{}", sfx(x.dtype), sfx(dtype)));
        let mut out = self.zeros_of(dtype, n);
        if n > 0 {
            let mut b = self.stream.launch_builder(f);
            match &mut out {
                CudaData::F32(o) => b.arg(o),
                CudaData::F64(o) => b.arg(o),
                CudaData::I64(o) => b.arg(o),
            };
            match &x.storage.data {
                CudaData::F32(s) => b.arg(s),
                CudaData::F64(s) => b.arg(s),
                CudaData::I64(s) => b.arg(s),
            };
            unsafe { b.arg(&xs).arg(&n64).launch(cfg(n)) }.expect("CUDA cast failed");
        }
        self.wrap(out)
    }
}
