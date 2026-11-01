//! The global random number generator behind `rand`, `randn`, init schemes,
//! dropout and shuffling. [`manual_seed`] makes runs reproducible, as
//! `torch.manual_seed` does.
//!
//! Random *tensors* are made in two steps: the global generator draws one
//! 64-bit seed, then the tensor's backend fills every element from the
//! counter-based [`Distribution::sample`]. A GPU fills them in a kernel
//! without a host copy, and every backend gives the same values for the
//! same seed.

use std::sync::Mutex;

use weta_core::{DType, Device, Distribution};

use crate::storage::Storage;
use crate::tensor::ok;
use crate::Tensor;

/// PCG-XSH-RR (O'Neill, 2014): 64-bit state, 32-bit output, small and fast,
/// with good statistical quality. Plenty for initialisation and dropout.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng { state: 0, inc: (0xda3e_39cb_94b9_5bdb << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | self.next_u32() as u64
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        let hi = (self.next_u32() >> 5) as u64; // 27 bits
        let lo = (self.next_u32() >> 6) as u64; // 26 bits
        ((hi << 26) | lo) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via the Box–Muller transform.
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.uniform(); // (0, 1], so ln is finite
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: u64) -> u64 {
        (self.uniform() * n as f64) as u64 % n.max(1)
    }

    /// Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }
}

/// PyTorch's default seed, so an unseeded run is still deterministic.
const DEFAULT_SEED: u64 = 67280421310721;

static RNG: Mutex<Option<Rng>> = Mutex::new(None);

/// Runs `f` with the global generator.
pub fn with_rng<R>(f: impl FnOnce(&mut Rng) -> R) -> R {
    let mut g = RNG.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(|| Rng::new(DEFAULT_SEED)))
}

/// Seeds the global generator (`torch.manual_seed`).
pub fn manual_seed(seed: u64) {
    *RNG.lock().unwrap_or_else(|e| e.into_inner()) = Some(Rng::new(seed));
}

impl Tensor {
    /// A tensor of independent draws from `dist`, made on `device`.
    #[track_caller]
    pub fn random(shape: &[usize], dist: Distribution, dtype: DType, device: Device) -> Tensor {
        let seed = with_rng(|r| r.next_u64());
        Tensor::random_seeded(shape, dist, dtype, device, seed)
    }

    /// As [`random`](Self::random), with an explicit seed instead of one from
    /// the global generator: the same seed gives the same tensor on every
    /// backend, whatever else the program (or another thread) has drawn.
    #[track_caller]
    pub fn random_seeded(shape: &[usize], dist: Distribution, dtype: DType, device: Device, seed: u64) -> Tensor {
        let n = shape.iter().product();
        // Integer tensors are drawn as f64 and rounded down.
        let float = if dtype.is_float() { dtype } else { DType::F64 };
        let t = Tensor::from_storage(ok(Storage::random(n, float, device, dist, seed)), shape, float, device);
        t.to_dtype(dtype)
    }

    /// Uniform in `[0, 1)`.
    pub fn rand(shape: &[usize], dtype: DType, device: Device) -> Tensor {
        Tensor::uniform(shape, 0.0, 1.0, dtype, device)
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(shape: &[usize], lo: f64, hi: f64, dtype: DType, device: Device) -> Tensor {
        Tensor::random(shape, Distribution::Uniform { lo, hi }, dtype, device)
    }

    /// Standard normal.
    pub fn randn(shape: &[usize], dtype: DType, device: Device) -> Tensor {
        Tensor::normal(shape, 0.0, 1.0, dtype, device)
    }

    /// Normal with the given mean and standard deviation.
    pub fn normal(shape: &[usize], mean: f64, std: f64, dtype: DType, device: Device) -> Tensor {
        Tensor::random(shape, Distribution::Normal { mean, std }, dtype, device)
    }

    /// Integers uniform in `[low, high)`, as `I64`.
    pub fn randint(low: i64, high: i64, shape: &[usize], device: Device) -> Tensor {
        let n = shape.iter().product();
        let span = (high - low).max(1) as u64;
        let v: Vec<i64> = with_rng(|r| (0..n).map(|_| low + r.below(span) as i64).collect());
        Tensor::from_vec(v, shape).to(device)
    }
}

/// A random permutation of `0..n` (`torch.randperm`).
pub fn randperm(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    with_rng(|r| r.shuffle(&mut v));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_runs_repeat() {
        manual_seed(1);
        let a = Tensor::randn(&[5], DType::F32, Device::Cpu).to_vec::<f32>();
        manual_seed(1);
        let b = Tensor::randn(&[5], DType::F32, Device::Cpu).to_vec::<f32>();
        assert_eq!(a, b);
    }

    #[test]
    fn random_tensors_have_the_right_moments() {
        // Explicit seeds: tests run in parallel and share the global generator.
        let n = Tensor::random_seeded(&[40000], Distribution::Normal { mean: 2.0, std: 3.0 }, DType::F64, Device::Cpu, 5);
        let mean = n.mean().item();
        let std = n.var_dims(&[0], false, false).item().sqrt();
        assert!((mean - 2.0).abs() < 0.05 && (std - 3.0).abs() < 0.05, "mean {mean} std {std}");
        let u = Tensor::random_seeded(&[40000], Distribution::Uniform { lo: -1.0, hi: 1.0 }, DType::F32, Device::Cpu, 6).to_vec::<f32>();
        assert!(u.iter().all(|v| (-1.0..1.0).contains(v)));
        assert!((u.iter().sum::<f32>() / u.len() as f32).abs() < 0.02);
    }

    #[test]
    fn normal_has_unit_variance() {
        let mut r = Rng::new(3);
        let xs: Vec<f64> = (0..20000).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05, "mean {mean} var {var}");
    }
}
