//! Device dispatch: [`Storage`] is one enum variant per backend, and each
//! method here matches on it and calls that backend's kernel. This is the
//! only place in weta that knows which backends exist.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use weta_core::{Backend, BinaryOp, DType, Device, HostData, Im2ColParams, Layout, ReduceOp, Result, UnaryOp, View};
use weta_cpu::{CpuBackend, CpuStorage};

pub enum Storage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda(weta_cuda::CudaStorage),
}

/// Storage shared by a tensor and all of its views, plus the version counter
/// that in-place ops bump (so autograd can detect a saved tensor changing).
pub(crate) struct StorageCell {
    data: RwLock<Storage>,
    version: AtomicU64,
}

impl StorageCell {
    pub fn new(s: Storage) -> Self {
        StorageCell { data: RwLock::new(s), version: AtomicU64::new(0) }
    }
    pub fn read(&self) -> RwLockReadGuard<'_, Storage> {
        self.data.read().unwrap_or_else(|e| e.into_inner())
    }
    pub fn write(&self) -> RwLockWriteGuard<'_, Storage> {
        self.version.fetch_add(1, Ordering::Relaxed);
        self.data.write().unwrap_or_else(|e| e.into_inner())
    }
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
}

/// True when the CUDA backend is compiled in and a GPU can be opened.
pub fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        weta_cuda::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// The CUDA backend's per-kernel timings when run with
/// `WETA_CUDA_PROFILE=1`, or `None`.
pub fn cuda_profile_report() -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        weta_cuda::profile_report()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        weta_cuda::device_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

macro_rules! v {
    ($s:expr, $l:expr, $dt:expr) => {
        View { storage: $s, layout: $l, dtype: $dt }
    };
}

/// Calls `$method` on the backend owning `$s`, wrapping the result back up.
macro_rules! dispatch1 {
    ($s:expr, $l:expr, $dt:expr, |$b:ident, $x:ident| $body:expr) => {
        match $s {
            Storage::Cpu(s) => {
                let $b = &CpuBackend;
                let $x = v!(s, $l, $dt);
                Storage::Cpu($body)
            }
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => {
                let $b = weta_cuda::backend(s.ordinal()).expect("CUDA device");
                let $x = v!(s, $l, $dt);
                Storage::Cuda($body)
            }
        }
    };
}

macro_rules! dispatch2 {
    ($sa:expr, $la:expr, $sb:expr, $lb:expr, $dt:expr, |$b:ident, $x:ident, $y:ident| $body:expr) => {
        match ($sa, $sb) {
            (Storage::Cpu(a), Storage::Cpu(c)) => {
                let $b = &CpuBackend;
                let ($x, $y) = (v!(a, $la, $dt), v!(c, $lb, $dt));
                Storage::Cpu($body)
            }
            #[cfg(feature = "cuda")]
            (Storage::Cuda(a), Storage::Cuda(c)) => {
                let $b = weta_cuda::backend(a.ordinal()).expect("CUDA device");
                let ($x, $y) = (v!(a, $la, $dt), v!(c, $lb, $dt));
                Storage::Cuda($body)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("device mismatch is checked before dispatch"),
        }
    };
}

impl Storage {
    pub fn from_host(data: HostData, device: Device) -> Result<Storage> {
        match device {
            Device::Cpu => Ok(Storage::Cpu(CpuBackend.from_host(data))),
            #[cfg(feature = "cuda")]
            Device::Cuda(i) => Ok(Storage::Cuda(weta_cuda::backend(i)?.from_host(data))),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(weta_core::Error::DeviceUnavailable("weta was built without the `cuda` feature".into())),
        }
    }

    pub fn zeros(len: usize, dtype: DType, device: Device) -> Result<Storage> {
        match device {
            Device::Cpu => Ok(Storage::Cpu(CpuBackend.zeros(len, dtype))),
            #[cfg(feature = "cuda")]
            Device::Cuda(i) => Ok(Storage::Cuda(weta_cuda::backend(i)?.zeros(len, dtype))),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(weta_core::Error::DeviceUnavailable("weta was built without the `cuda` feature".into())),
        }
    }

    pub fn to_host(&self, l: &Layout, dt: DType) -> HostData {
        match self {
            Storage::Cpu(s) => CpuBackend.to_host(v!(s, l, dt)),
            #[cfg(feature = "cuda")]
            Storage::Cuda(s) => weta_cuda::backend(s.ordinal()).expect("CUDA device").to_host(v!(s, l, dt)),
        }
    }

    pub fn unary(&self, op: UnaryOp, l: &Layout, dt: DType) -> Storage {
        dispatch1!(self, l, dt, |b, x| b.unary(op, x))
    }

    pub fn binary(&self, op: BinaryOp, la: &Layout, other: &Storage, lb: &Layout, dt: DType) -> Storage {
        dispatch2!(self, la, other, lb, dt, |b, x, y| b.binary(op, x, y))
    }

    pub fn reduce(&self, op: ReduceOp, l: &Layout, dt: DType, axis: usize) -> Storage {
        dispatch1!(self, l, dt, |b, x| b.reduce(op, x, axis))
    }

    pub fn matmul(&self, la: &Layout, other: &Storage, lb: &Layout, dt: DType) -> Storage {
        dispatch2!(self, la, other, lb, dt, |b, x, y| b.matmul(x, y))
    }

    pub fn im2col(&self, l: &Layout, dt: DType, p: Im2ColParams) -> Storage {
        dispatch1!(self, l, dt, |b, x| b.im2col(x, p))
    }

    pub fn col2im(&self, l: &Layout, dt: DType, p: Im2ColParams, chw: (usize, usize, usize)) -> Storage {
        dispatch1!(self, l, dt, |b, x| b.col2im(x, p, chw))
    }

    pub fn cast(&self, l: &Layout, dt: DType, to: DType) -> Storage {
        dispatch1!(self, l, dt, |b, x| b.cast(x, to))
    }

    /// Writes `src` (viewed through `ls`) into `self` at `ld`.
    pub fn copy_from(&mut self, ld: &Layout, src: &Storage, ls: &Layout, dt: DType) {
        match (self, src) {
            (Storage::Cpu(d), Storage::Cpu(s)) => CpuBackend.copy_strided(v!(s, ls, dt), d, ld),
            #[cfg(feature = "cuda")]
            (Storage::Cuda(d), Storage::Cuda(s)) => {
                weta_cuda::backend(d.ordinal()).expect("CUDA device").copy_strided(v!(s, ls, dt), d, ld)
            }
            #[allow(unreachable_patterns)]
            _ => unreachable!("device mismatch is checked before dispatch"),
        }
    }
}
