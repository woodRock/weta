//! # weta-cpu
//!
//! The reference backend. Every other backend is tested against this one, so
//! it favours clarity: each kernel walks a strided [`Layout`] with a plain
//! loop, with fast paths for contiguous data and `rayon` threads for large
//! tensors. Matrix multiplication goes through the `gemm` crate, which is
//! within a small factor of MKL/OpenBLAS.

use weta_core::{Backend, BinaryOp, DType, Device, Element, Float, HostData, Im2ColParams, Layout, ReduceOp, UnaryOp, View};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Data on the CPU: a plain vector per dtype.
#[derive(Clone, Debug, PartialEq)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F64(_) => DType::F64,
            CpuStorage::I64(_) => DType::I64,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            CpuStorage::F32(v) => v.len(),
            CpuStorage::F64(v) => v.len(),
            CpuStorage::I64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

/// Below this many elements, threads cost more than they save.
#[cfg_attr(not(feature = "rayon"), allow(dead_code))]
const PAR_MIN: usize = 1 << 15;

// ---------------------------------------------------------------------------
// Strided traversal helpers. Each takes a slice plus a layout and produces a
// contiguous output in row-major order.
// ---------------------------------------------------------------------------

/// Applies `f` to every element of a strided view.
fn map<T: Copy + Send + Sync, U: Copy + Send + Sync + Default>(x: &[T], l: &Layout, f: impl Fn(T) -> U + Sync + Send) -> Vec<U> {
    let n = l.numel();
    if l.is_contiguous() {
        let s = &x[l.offset()..l.offset() + n];
        #[cfg(feature = "rayon")]
        if n >= PAR_MIN {
            return s.par_iter().map(|&v| f(v)).collect();
        }
        return s.iter().map(|&v| f(v)).collect();
    }
    let mut out = vec![U::default(); n];
    for_each_row(l, None, &mut out, |row_l, _, row_out| {
        for (o, i) in row_out.iter_mut().zip(row_l.offsets()) {
            *o = f(x[i]);
        }
    });
    out
}

/// Applies `f` element-wise to two strided views of the same shape.
fn zip<T: Copy + Send + Sync, U: Copy + Send + Sync + Default>(
    a: &[T],
    la: &Layout,
    b: &[T],
    lb: &Layout,
    f: impl Fn(T, T) -> U + Sync + Send,
) -> Vec<U> {
    let n = la.numel();
    if la.is_contiguous() && lb.is_contiguous() {
        let (sa, sb) = (&a[la.offset()..la.offset() + n], &b[lb.offset()..lb.offset() + n]);
        #[cfg(feature = "rayon")]
        if n >= PAR_MIN {
            return sa.par_iter().zip(sb.par_iter()).map(|(&x, &y)| f(x, y)).collect();
        }
        return sa.iter().zip(sb).map(|(&x, &y)| f(x, y)).collect();
    }
    let mut out = vec![U::default(); n];
    for_each_row(la, Some(lb), &mut out, |ra, rb, row_out| {
        let rb = rb.unwrap();
        for ((o, i), j) in row_out.iter_mut().zip(ra.offsets()).zip(rb.offsets()) {
            *o = f(a[i], b[j]);
        }
    });
    out
}

/// Splits the output by the first dimension and runs `f` on each slab,
/// in parallel when the tensor is large. `f` gets the slab's layout(s).
fn for_each_row<U: Send>(la: &Layout, lb: Option<&Layout>, out: &mut [U], f: impl Fn(&Layout, Option<&Layout>, &mut [U]) + Sync + Send) {
    let n = la.numel();
    if la.ndim() == 0 || n == 0 {
        f(la, lb, out);
        return;
    }
    let rows = la.shape()[0];
    let row_len = n / rows;
    let slab = |i: usize, chunk: &mut [U]| {
        let ra = la.narrow(0, i, 1).unwrap();
        let rb = lb.map(|l| l.narrow(0, i, 1).unwrap());
        f(&ra, rb.as_ref(), chunk);
    };
    #[cfg(feature = "rayon")]
    if n >= PAR_MIN && rows > 1 {
        out.par_chunks_mut(row_len).enumerate().for_each(|(i, c)| slab(i, c));
        return;
    }
    out.chunks_mut(row_len).enumerate().for_each(|(i, c)| slab(i, c));
}

/// A contiguous copy of a strided view.
fn gather<T: Copy + Send + Sync + Default>(x: &[T], l: &Layout) -> Vec<T> {
    map(x, l, |v| v)
}

// ---------------------------------------------------------------------------
// Kernels, generic over the element type.
// ---------------------------------------------------------------------------

fn unary_float<T: Float + Default>(op: UnaryOp, x: &[T], l: &Layout) -> Vec<T> {
    let c = T::from_f64;
    let (zero, one) = (T::zero(), T::one());
    match op {
        UnaryOp::Neg => map(x, l, |v| -v),
        UnaryOp::Exp => map(x, l, |v| v.exp()),
        UnaryOp::Log => map(x, l, |v| v.ln()),
        UnaryOp::Sqrt => map(x, l, |v| v.sqrt()),
        UnaryOp::Abs => map(x, l, |v| v.abs()),
        UnaryOp::Sign => map(x, l, move |v| {
            if v > zero {
                one
            } else if v < zero {
                -one
            } else {
                zero
            }
        }),
        UnaryOp::Relu => map(x, l, move |v| if v > zero { v } else { zero }),
        UnaryOp::Step => map(x, l, move |v| if v > zero { one } else { zero }),
        UnaryOp::Tanh => map(x, l, |v| v.tanh()),
        UnaryOp::Sigmoid => map(x, l, move |v| {
            // Two branches keep exp from overflowing for large |v|.
            if v >= zero {
                one / (one + (-v).exp())
            } else {
                let e = v.exp();
                e / (one + e)
            }
        }),
        UnaryOp::Erf => map(x, l, |v| v.erf()),
        UnaryOp::Sin => map(x, l, |v| v.sin()),
        UnaryOp::Cos => map(x, l, |v| v.cos()),
        UnaryOp::Recip => map(x, l, move |v| one / v),
        UnaryOp::Affine { mul, add } => {
            let (m, a) = (c(mul), c(add));
            map(x, l, move |v| v * m + a)
        }
        UnaryOp::Powf(e) => {
            let e = c(e);
            map(x, l, move |v| v.powf(e))
        }
        UnaryOp::GtScalar(s) => {
            let s = c(s);
            map(x, l, move |v| if v > s { one } else { zero })
        }
        UnaryOp::MaxScalar(s) => {
            let s = c(s);
            map(x, l, move |v| v.max(s))
        }
        UnaryOp::MinScalar(s) => {
            let s = c(s);
            map(x, l, move |v| v.min(s))
        }
    }
}

fn unary_int(op: UnaryOp, x: &[i64], l: &Layout) -> Vec<i64> {
    match op {
        UnaryOp::Neg => map(x, l, |v| -v),
        UnaryOp::Abs => map(x, l, |v| v.abs()),
        UnaryOp::Sign => map(x, l, |v| v.signum()),
        UnaryOp::Relu => map(x, l, |v| v.max(0)),
        UnaryOp::Step => map(x, l, |v| (v > 0) as i64),
        UnaryOp::Affine { mul, add } => map(x, l, move |v| (v as f64 * mul + add) as i64),
        UnaryOp::GtScalar(s) => map(x, l, move |v| ((v as f64) > s) as i64),
        UnaryOp::MaxScalar(s) => map(x, l, move |v| v.max(s as i64)),
        UnaryOp::MinScalar(s) => map(x, l, move |v| v.min(s as i64)),
        _ => panic!("{}: not supported for i64 tensors", op.name()),
    }
}

fn binary_float<T: Float + Default>(op: BinaryOp, a: &[T], la: &Layout, b: &[T], lb: &Layout) -> Vec<T> {
    let (zero, one) = (T::zero(), T::one());
    let m = move |c: bool| if c { one } else { zero };
    match op {
        BinaryOp::Add => zip(a, la, b, lb, |x, y| x + y),
        BinaryOp::Sub => zip(a, la, b, lb, |x, y| x - y),
        BinaryOp::Mul => zip(a, la, b, lb, |x, y| x * y),
        BinaryOp::Div => zip(a, la, b, lb, |x, y| x / y),
        BinaryOp::Pow => zip(a, la, b, lb, |x, y| x.powf(y)),
        BinaryOp::Maximum => zip(a, la, b, lb, |x, y| x.max(y)),
        BinaryOp::Minimum => zip(a, la, b, lb, |x, y| x.min(y)),
        BinaryOp::Eq => zip(a, la, b, lb, move |x, y| m(x == y)),
        BinaryOp::Ne => zip(a, la, b, lb, move |x, y| m(x != y)),
        BinaryOp::Lt => zip(a, la, b, lb, move |x, y| m(x < y)),
        BinaryOp::Le => zip(a, la, b, lb, move |x, y| m(x <= y)),
        BinaryOp::Gt => zip(a, la, b, lb, move |x, y| m(x > y)),
        BinaryOp::Ge => zip(a, la, b, lb, move |x, y| m(x >= y)),
    }
}

fn binary_int(op: BinaryOp, a: &[i64], la: &Layout, b: &[i64], lb: &Layout) -> Vec<i64> {
    match op {
        BinaryOp::Add => zip(a, la, b, lb, |x, y| x + y),
        BinaryOp::Sub => zip(a, la, b, lb, |x, y| x - y),
        BinaryOp::Mul => zip(a, la, b, lb, |x, y| x * y),
        BinaryOp::Div => zip(a, la, b, lb, |x, y| x.div_euclid(y)),
        BinaryOp::Pow => zip(a, la, b, lb, |x, y| x.pow(y as u32)),
        BinaryOp::Maximum => zip(a, la, b, lb, |x, y| x.max(y)),
        BinaryOp::Minimum => zip(a, la, b, lb, |x, y| x.min(y)),
        BinaryOp::Eq => zip(a, la, b, lb, |x, y| (x == y) as i64),
        BinaryOp::Ne => zip(a, la, b, lb, |x, y| (x != y) as i64),
        BinaryOp::Lt => zip(a, la, b, lb, |x, y| (x < y) as i64),
        BinaryOp::Le => zip(a, la, b, lb, |x, y| (x <= y) as i64),
        BinaryOp::Gt => zip(a, la, b, lb, |x, y| (x > y) as i64),
        BinaryOp::Ge => zip(a, la, b, lb, |x, y| (x >= y) as i64),
    }
}

/// Storage offsets of every output element of a reduction over `axis`,
/// plus the axis's length and stride.
fn reduce_plan(l: &Layout, axis: usize) -> (Vec<usize>, usize, usize) {
    let mut shape = l.shape().to_vec();
    let mut strides = l.strides().to_vec();
    let (n, stride) = (shape.remove(axis), strides.remove(axis));
    let outer = Layout::new(shape, strides, l.offset());
    (outer.offsets().collect(), n, stride)
}

fn reduce_generic<T: Element + Default, U: Copy + Send + Sync + Default>(
    x: &[T],
    l: &Layout,
    axis: usize,
    f: impl Fn(&mut dyn Iterator<Item = T>) -> U + Sync + Send,
) -> Vec<U> {
    let (bases, n, stride) = reduce_plan(l, axis);
    let one = |&base: &usize| f(&mut (0..n).map(|k| x[base + k * stride]));
    #[cfg(feature = "rayon")]
    if l.numel() >= PAR_MIN {
        return bases.par_iter().map(one).collect();
    }
    bases.iter().map(one).collect()
}

fn argext<T: Element>(it: &mut dyn Iterator<Item = T>, better: impl Fn(T, T) -> bool) -> i64 {
    let mut best: Option<(usize, T)> = None;
    for (i, v) in it.enumerate() {
        match best {
            Some((_, b)) if !better(v, b) => {}
            _ => best = Some((i, v)),
        }
    }
    best.map_or(0, |(i, _)| i as i64)
}

fn reduce_float<T: Float + Default>(op: ReduceOp, x: &[T], l: &Layout, axis: usize) -> CpuStorage
where
    Vec<T>: Into<CpuStorage>,
{
    match op {
        ReduceOp::Sum => reduce_generic(x, l, axis, |it| it.fold(T::zero(), |a, b| a + b)).into(),
        ReduceOp::Max => reduce_generic(x, l, axis, |it| it.fold(T::neg_infinity(), |a, b| a.max(b))).into(),
        ReduceOp::Min => reduce_generic(x, l, axis, |it| it.fold(T::infinity(), |a, b| a.min(b))).into(),
        ReduceOp::ArgMax => CpuStorage::I64(reduce_generic(x, l, axis, |it| argext(it, |v, b| v > b))),
        ReduceOp::ArgMin => CpuStorage::I64(reduce_generic(x, l, axis, |it| argext(it, |v, b| v < b))),
    }
}

fn reduce_int(op: ReduceOp, x: &[i64], l: &Layout, axis: usize) -> CpuStorage {
    CpuStorage::I64(match op {
        ReduceOp::Sum => reduce_generic(x, l, axis, |it| it.sum()),
        ReduceOp::Max => reduce_generic(x, l, axis, |it| it.max().unwrap_or(i64::MIN)),
        ReduceOp::Min => reduce_generic(x, l, axis, |it| it.min().unwrap_or(i64::MAX)),
        ReduceOp::ArgMax => reduce_generic(x, l, axis, |it| argext(it, |v, b| v > b)),
        ReduceOp::ArgMin => reduce_generic(x, l, axis, |it| argext(it, |v, b| v < b)),
    })
}

impl From<Vec<f32>> for CpuStorage {
    fn from(v: Vec<f32>) -> Self {
        CpuStorage::F32(v)
    }
}
impl From<Vec<f64>> for CpuStorage {
    fn from(v: Vec<f64>) -> Self {
        CpuStorage::F64(v)
    }
}
impl From<Vec<i64>> for CpuStorage {
    fn from(v: Vec<i64>) -> Self {
        CpuStorage::I64(v)
    }
}

/// Batched GEMM over 3-D views `[B, M, K] × [B, K, N]`, any strides.
fn matmul_float<T: Float + Default>(a: &[T], la: &Layout, b: &[T], lb: &Layout) -> Vec<T> {
    let (bs, m, k) = (la.shape()[0], la.shape()[1], la.shape()[2]);
    let n = lb.shape()[2];
    let (sa, sb) = (la.strides(), lb.strides());
    let mut out = vec![T::zero(); bs * m * n];
    if m * n == 0 {
        return out;
    }
    let one = |batch: usize, dst: &mut [T], par: gemm::Parallelism| unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr().add(la.offset() + batch * sa[0]),
            sa[2] as isize,
            sa[1] as isize,
            b.as_ptr().add(lb.offset() + batch * sb[0]),
            sb[2] as isize,
            sb[1] as isize,
            T::zero(),
            T::one(),
            false,
            false,
            false,
            par,
        )
    };
    #[cfg(feature = "rayon")]
    {
        // Many small matrices (e.g. a conv over a batch): one thread each.
        // Few large ones: let gemm split each matrix across threads.
        let threads = rayon::current_num_threads();
        if bs >= threads && bs > 1 {
            out.par_chunks_mut(m * n).enumerate().for_each(|(i, dst)| one(i, dst, gemm::Parallelism::None));
        } else {
            let par = if m * n * k >= 1 << 16 { gemm::Parallelism::Rayon(0) } else { gemm::Parallelism::None };
            out.chunks_mut(m * n).enumerate().for_each(|(i, dst)| one(i, dst, par));
        }
    }
    #[cfg(not(feature = "rayon"))]
    out.chunks_mut(m * n).enumerate().for_each(|(i, dst)| one(i, dst, gemm::Parallelism::None));
    out
}

fn im2col_generic<T: Element + Default>(x: &[T], l: &Layout, p: Im2ColParams) -> Vec<T> {
    let s = l.shape();
    let st = l.strides();
    let (nb, c, h, w) = (s[0], s[1], s[2], s[3]);
    let (kh, kw) = p.kernel;
    let (oh, ow) = p.out_size(h, w);
    let rows = c * kh * kw;
    let mut out = vec![T::zero(); nb * rows * oh * ow];
    // Each (n, c) pair fills kh·kw rows of the output independently.
    let fill = |nc: usize, chunk: &mut [T]| {
        let (n, ci) = (nc / c, nc % c);
        let base = l.offset() + n * st[0] + ci * st[1];
        for ki in 0..kh {
            for kj in 0..kw {
                let row = &mut chunk[(ki * kw + kj) * oh * ow..(ki * kw + kj + 1) * oh * ow];
                for y in 0..oh {
                    let iy = (y * p.stride.0 + ki * p.dilation.0) as isize - p.padding.0 as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for xx in 0..ow {
                        let ix = (xx * p.stride.1 + kj * p.dilation.1) as isize - p.padding.1 as isize;
                        if ix >= 0 && ix < w as isize {
                            row[y * ow + xx] = x[base + iy as usize * st[2] + ix as usize * st[3]];
                        }
                    }
                }
            }
        }
    };
    let chunk = kh * kw * oh * ow;
    if chunk == 0 {
        return out;
    }
    #[cfg(feature = "rayon")]
    if out.len() >= PAR_MIN {
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, ch)| fill(i, ch));
        return out;
    }
    out.chunks_mut(chunk).enumerate().for_each(|(i, ch)| fill(i, ch));
    out
}

fn col2im_float<T: Float + Default>(cols: &[T], p: Im2ColParams, nb: usize, (c, h, w): (usize, usize, usize)) -> Vec<T> {
    let (kh, kw) = p.kernel;
    let (oh, ow) = p.out_size(h, w);
    let mut out = vec![T::zero(); nb * c * h * w];
    let fill = |nc: usize, img: &mut [T]| {
        let src = &cols[nc * kh * kw * oh * ow..(nc + 1) * kh * kw * oh * ow];
        for ki in 0..kh {
            for kj in 0..kw {
                let row = &src[(ki * kw + kj) * oh * ow..];
                for y in 0..oh {
                    let iy = (y * p.stride.0 + ki * p.dilation.0) as isize - p.padding.0 as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for xx in 0..ow {
                        let ix = (xx * p.stride.1 + kj * p.dilation.1) as isize - p.padding.1 as isize;
                        if ix >= 0 && ix < w as isize {
                            let o = iy as usize * w + ix as usize;
                            img[o] = img[o] + row[y * ow + xx];
                        }
                    }
                }
            }
        }
    };
    let chunk = h * w;
    if chunk == 0 {
        return out;
    }
    #[cfg(feature = "rayon")]
    if out.len() >= PAR_MIN {
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, im)| fill(i, im));
        return out;
    }
    out.chunks_mut(chunk).enumerate().for_each(|(i, im)| fill(i, im));
    out
}

fn copy_into<T: Copy>(src: &[T], ls: &Layout, dst: &mut [T], ld: &Layout) {
    if ls.is_contiguous() && ld.is_contiguous() {
        let n = ls.numel();
        dst[ld.offset()..ld.offset() + n].copy_from_slice(&src[ls.offset()..ls.offset() + n]);
        return;
    }
    for (d, s) in ld.offsets().zip(ls.offsets()) {
        dst[d] = src[s];
    }
}

/// Runs `$body` with `$v` bound to the typed slice of a [`CpuStorage`].
macro_rules! with_float {
    ($s:expr, $v:ident => $body:expr, int $iv:ident => $ibody:expr) => {
        match $s {
            CpuStorage::F32($v) => CpuStorage::from($body),
            CpuStorage::F64($v) => CpuStorage::from($body),
            CpuStorage::I64($iv) => CpuStorage::from($ibody),
        }
    };
}

impl Backend for CpuBackend {
    type Storage = CpuStorage;

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn from_host(&self, data: HostData) -> CpuStorage {
        match data {
            HostData::F32(v) => CpuStorage::F32(v),
            HostData::F64(v) => CpuStorage::F64(v),
            HostData::I64(v) => CpuStorage::I64(v),
        }
    }

    fn to_host(&self, x: View<'_, CpuStorage>) -> HostData {
        match x.storage {
            CpuStorage::F32(v) => HostData::F32(gather(v, x.layout)),
            CpuStorage::F64(v) => HostData::F64(gather(v, x.layout)),
            CpuStorage::I64(v) => HostData::I64(gather(v, x.layout)),
        }
    }

    fn zeros(&self, len: usize, dtype: DType) -> CpuStorage {
        match dtype {
            DType::F32 => CpuStorage::F32(vec![0.0; len]),
            DType::F64 => CpuStorage::F64(vec![0.0; len]),
            DType::I64 => CpuStorage::I64(vec![0; len]),
        }
    }

    fn copy_strided(&self, src: View<'_, CpuStorage>, dst: &mut CpuStorage, dst_layout: &Layout) {
        match (src.storage, dst) {
            (CpuStorage::F32(s), CpuStorage::F32(d)) => copy_into(s, src.layout, d, dst_layout),
            (CpuStorage::F64(s), CpuStorage::F64(d)) => copy_into(s, src.layout, d, dst_layout),
            (CpuStorage::I64(s), CpuStorage::I64(d)) => copy_into(s, src.layout, d, dst_layout),
            (s, d) => panic!("copy: dtype mismatch {} → {}", s.dtype(), d.dtype()),
        }
    }

    fn unary(&self, op: UnaryOp, x: View<'_, CpuStorage>) -> CpuStorage {
        with_float!(x.storage, v => unary_float(op, v, x.layout), int v => unary_int(op, v, x.layout))
    }

    fn binary(&self, op: BinaryOp, a: View<'_, CpuStorage>, b: View<'_, CpuStorage>) -> CpuStorage {
        match (a.storage, b.storage) {
            (CpuStorage::F32(x), CpuStorage::F32(y)) => binary_float(op, x, a.layout, y, b.layout).into(),
            (CpuStorage::F64(x), CpuStorage::F64(y)) => binary_float(op, x, a.layout, y, b.layout).into(),
            (CpuStorage::I64(x), CpuStorage::I64(y)) => binary_int(op, x, a.layout, y, b.layout).into(),
            (x, y) => panic!("{}: dtype mismatch {} and {}", op.name(), x.dtype(), y.dtype()),
        }
    }

    fn reduce(&self, op: ReduceOp, x: View<'_, CpuStorage>, axis: usize) -> CpuStorage {
        match x.storage {
            CpuStorage::F32(v) => reduce_float(op, v, x.layout, axis),
            CpuStorage::F64(v) => reduce_float(op, v, x.layout, axis),
            CpuStorage::I64(v) => reduce_int(op, v, x.layout, axis),
        }
    }

    fn matmul(&self, a: View<'_, CpuStorage>, b: View<'_, CpuStorage>) -> CpuStorage {
        match (a.storage, b.storage) {
            (CpuStorage::F32(x), CpuStorage::F32(y)) => matmul_float(x, a.layout, y, b.layout).into(),
            (CpuStorage::F64(x), CpuStorage::F64(y)) => matmul_float(x, a.layout, y, b.layout).into(),
            (x, y) => panic!("matmul: unsupported dtypes {} and {}", x.dtype(), y.dtype()),
        }
    }

    fn im2col(&self, x: View<'_, CpuStorage>, p: Im2ColParams) -> CpuStorage {
        with_float!(x.storage, v => im2col_generic(v, x.layout, p), int v => im2col_generic(v, x.layout, p))
    }

    fn col2im(&self, cols: View<'_, CpuStorage>, p: Im2ColParams, chw: (usize, usize, usize)) -> CpuStorage {
        let nb = cols.layout.shape()[0];
        match cols.storage {
            CpuStorage::F32(v) => col2im_float(&gather(v, cols.layout), p, nb, chw).into(),
            CpuStorage::F64(v) => col2im_float(&gather(v, cols.layout), p, nb, chw).into(),
            CpuStorage::I64(_) => panic!("col2im: unsupported dtype i64"),
        }
    }

    fn cast(&self, x: View<'_, CpuStorage>, dtype: DType) -> CpuStorage {
        let l = x.layout;
        match (x.storage, dtype) {
            (CpuStorage::F32(v), DType::F32) => CpuStorage::F32(gather(v, l)),
            (CpuStorage::F32(v), DType::F64) => CpuStorage::F64(map(v, l, |a| a as f64)),
            (CpuStorage::F32(v), DType::I64) => CpuStorage::I64(map(v, l, |a| a as i64)),
            (CpuStorage::F64(v), DType::F32) => CpuStorage::F32(map(v, l, |a| a as f32)),
            (CpuStorage::F64(v), DType::F64) => CpuStorage::F64(gather(v, l)),
            (CpuStorage::F64(v), DType::I64) => CpuStorage::I64(map(v, l, |a| a as i64)),
            (CpuStorage::I64(v), DType::F32) => CpuStorage::F32(map(v, l, |a| a as f32)),
            (CpuStorage::I64(v), DType::F64) => CpuStorage::F64(map(v, l, |a| a as f64)),
            (CpuStorage::I64(v), DType::I64) => CpuStorage::I64(gather(v, l)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(s: &'a CpuStorage, l: &'a Layout) -> View<'a, CpuStorage> {
        View { storage: s, layout: l, dtype: s.dtype() }
    }

    #[test]
    fn matmul_with_transposed_input() {
        let a = CpuStorage::F32(vec![1., 2., 3., 4., 5., 6.]); // [1, 2, 3]
        let la = Layout::contiguous(&[1, 2, 3]);
        let b = CpuStorage::F32(vec![1., 0., 1., 0., 1., 1.]); // stored [1, 2, 3], used as [1, 3, 2]
        let lb = Layout::contiguous(&[1, 2, 3]).transpose(1, 2).unwrap();
        let out = CpuBackend.matmul(view(&a, &la), view(&b, &lb));
        // rows of b as columns: [1,0,1] and [0,1,1]
        assert_eq!(out, CpuStorage::F32(vec![4., 5., 10., 11.]));
    }

    #[test]
    fn reduce_over_middle_axis() {
        let x = CpuStorage::F64((0..24).map(|v| v as f64).collect());
        let l = Layout::contiguous(&[2, 3, 4]);
        let CpuStorage::F64(s) = CpuBackend.reduce(ReduceOp::Sum, view(&x, &l), 1) else { panic!() };
        assert_eq!(s, vec![12., 15., 18., 21., 48., 51., 54., 57.]);
        let CpuStorage::I64(am) = CpuBackend.reduce(ReduceOp::ArgMax, view(&x, &l), 2) else { panic!() };
        assert_eq!(am, vec![3; 6]);
    }

    #[test]
    fn im2col_then_col2im_counts_overlaps() {
        let x = CpuStorage::F32(vec![1.0; 9]);
        let l = Layout::contiguous(&[1, 1, 3, 3]);
        let p = Im2ColParams::new((2, 2));
        let cols = CpuBackend.im2col(view(&x, &l), p);
        assert_eq!(cols.len(), 4 * 4);
        let lc = Layout::contiguous(&[1, 4, 4]);
        let CpuStorage::F32(img) = CpuBackend.col2im(view(&cols, &lc), p, (1, 3, 3)) else { panic!() };
        assert_eq!(img, vec![1., 2., 1., 2., 4., 2., 1., 2., 1.]);
    }
}
