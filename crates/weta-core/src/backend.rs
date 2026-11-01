use crate::{BinaryOp, DType, Device, Im2ColParams, Layout, ReduceOp, UnaryOp};

/// A read-only look at a backend's storage through a layout.
pub struct View<'a, S> {
    pub storage: &'a S,
    pub layout: &'a Layout,
    pub dtype: DType,
}

impl<S> Clone for View<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S> Copy for View<'_, S> {}

/// Data on the host, as moved between devices or loaded from files.
#[derive(Clone, Debug, PartialEq)]
pub enum HostData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

impl HostData {
    pub fn dtype(&self) -> DType {
        match self {
            HostData::F32(_) => DType::F32,
            HostData::F64(_) => DType::F64,
            HostData::I64(_) => DType::I64,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            HostData::F32(v) => v.len(),
            HostData::F64(v) => v.len(),
            HostData::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self {
            HostData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            HostData::F64(v) => v.clone(),
            HostData::I64(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }
}

/// What a device has to provide: storage, and about a dozen primitive
/// kernels. Ops, autograd and layers are written once on top of this, so a
/// new backend only implements these methods.
///
/// Conventions:
/// - Every kernel returns fresh, contiguous storage.
/// - Inputs are [`View`]s and may be strided, including stride-0
///   (broadcast) dimensions, so transposes and broadcasts need no copy.
/// - Shapes have already been checked by `weta-tensor`; backends may assume
///   they are valid.
pub trait Backend: Send + Sync + 'static {
    type Storage: Send + Sync;

    fn device(&self) -> Device;

    #[allow(clippy::wrong_self_convention)]
    fn from_host(&self, data: HostData) -> Self::Storage;
    /// Copies the elements of `x` (in row-major order) to the host.
    fn to_host(&self, x: View<'_, Self::Storage>) -> HostData;
    fn zeros(&self, len: usize, dtype: DType) -> Self::Storage;

    /// Writes `src` into the elements of `dst` described by `dst_layout`.
    /// `src` and `dst_layout` have the same shape. This one kernel gives
    /// `contiguous()`, `cat`, slice assignment and in-place ops.
    fn copy_strided(&self, src: View<'_, Self::Storage>, dst: &mut Self::Storage, dst_layout: &Layout);

    fn unary(&self, op: UnaryOp, x: View<'_, Self::Storage>) -> Self::Storage;
    /// `a` and `b` have the same shape (already broadcast).
    fn binary(&self, op: BinaryOp, a: View<'_, Self::Storage>, b: View<'_, Self::Storage>) -> Self::Storage;
    /// Reduces `axis`; the output has the input's shape without that axis.
    fn reduce(&self, op: ReduceOp, x: View<'_, Self::Storage>, axis: usize) -> Self::Storage;
    /// Batched matrix multiply: `[B, M, K] × [B, K, N] → [B, M, N]`.
    fn matmul(&self, a: View<'_, Self::Storage>, b: View<'_, Self::Storage>) -> Self::Storage;
    /// `[N, C, H, W]` → `[N, C·kh·kw, OH·OW]`.
    fn im2col(&self, x: View<'_, Self::Storage>, p: Im2ColParams) -> Self::Storage;
    /// `[N, C·kh·kw, OH·OW]` → `[N, C, H, W]`, summing overlaps.
    fn col2im(&self, cols: View<'_, Self::Storage>, p: Im2ColParams, chw: (usize, usize, usize)) -> Self::Storage;
    fn cast(&self, x: View<'_, Self::Storage>, dtype: DType) -> Self::Storage;
}
