//! # weta-tensor
//!
//! N-dimensional, strided tensors on any backend, with no autograd: the
//! NumPy layer of weta. A [`Tensor`] carries its [`DType`] and [`Device`] at
//! runtime, as in PyTorch, and every op checks shapes once with a clear
//! error. Short names (`matmul`) panic on bad shapes; `try_` versions
//! (`try_matmul`) return a [`Result`].
//!
//! Composite ops (softmax, conv2d, max pooling, variance) are written here
//! in terms of the backend primitives, so `weta-infer` can run models
//! without linking autograd.

mod display;
mod random;
mod storage;
mod tensor;

pub use random::{manual_seed, randperm, with_rng, Rng};
pub use storage::{cuda_device_count, cuda_is_available};
pub use tensor::{allclose, FromHost, IntoHost, Tensor};
pub use weta_core::{BinaryOp, DType, Device, Error, HostData, Im2ColParams, Layout, ReduceOp, Result, UnaryOp};

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape)
    }

    #[test]
    fn broadcasting_add() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = t(&[10., 20., 30.], &[3]);
        assert_eq!(a.add(&b).to_vec::<f32>(), vec![11., 22., 33., 14., 25., 36.]);
        let c = t(&[100., 200.], &[2, 1]);
        assert_eq!(a.add(&c).to_vec::<f32>(), vec![101., 102., 103., 204., 205., 206.]);
    }

    #[test]
    fn matmul_shapes_and_errors() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let b = t(&[1., 2., 3., 4., 5., 6.], &[3, 2]);
        assert_eq!(a.matmul(&b).to_vec::<f32>(), vec![22., 28., 49., 64.]);
        // A transposed view goes through gemm without a copy.
        assert_eq!(a.matmul(&a.t()).to_vec::<f32>(), vec![14., 32., 32., 77.]);
        let err = a.try_matmul(&a).unwrap_err().to_string();
        assert_eq!(err, "matmul: [2, 3] × [2, 3]: inner dims 3 ≠ 2");
        // Batched with broadcasting: [2, 2, 3] × [3, 2].
        let ab = Tensor::stack(&[&a, &a], 0);
        assert_eq!(ab.matmul(&b).shape(), &[2, 2, 2]);
        // Vector cases.
        let v = t(&[1., 1., 1.], &[3]);
        assert_eq!(a.matmul(&v).to_vec::<f32>(), vec![6., 15.]);
    }

    #[test]
    fn reductions_and_softmax() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        assert_eq!(a.sum().item(), 21.0);
        assert_eq!(a.sum_dims(&[0], false).to_vec::<f32>(), vec![5., 7., 9.]);
        assert_eq!(a.max_dims(&[1], true).shape(), &[2, 1]);
        assert_eq!(a.argmax(1, false).to_vec::<i64>(), vec![2, 2]);
        let s = a.softmax(1).sum_dims(&[1], false).to_vec::<f32>();
        assert!(s.iter().all(|v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn views_share_storage() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
        let row = a.narrow(0, 1, 1);
        row.fill_(0.0);
        assert_eq!(a.to_vec::<f32>(), vec![1., 2., 3., 0., 0., 0.]);
        assert_eq!(a.t().contiguous().to_vec::<f32>(), vec![1., 0., 2., 0., 3., 0.]);
        assert_eq!(a.select(1, 2).to_vec::<f32>(), vec![3., 0.]);
        let v0 = a.version();
        a.add_scalar_(1.0);
        assert!(a.version() > v0);
    }

    #[test]
    fn cat_and_conv() {
        let a = t(&[1., 2.], &[1, 2]);
        let b = t(&[3., 4., 5., 6.], &[2, 2]);
        assert_eq!(Tensor::cat(&[&a, &b], 0).to_vec::<f32>(), vec![1., 2., 3., 4., 5., 6.]);
        // A 2x2 sum filter over a 3x3 image of ones.
        let x = Tensor::ones(&[1, 1, 3, 3], DType::F32, Device::Cpu);
        let w = Tensor::ones(&[1, 1, 2, 2], DType::F32, Device::Cpu);
        assert_eq!(x.conv2d(&w, None, 1, 0).to_vec::<f32>(), vec![4.; 4]);
        assert_eq!(x.conv2d(&w, None, 1, 1).shape(), &[1, 1, 4, 4]);
        let img = t(&[1., 5., 2., 3., 0., 7., 4., 1., 9., 2., 6., 3., 8., 0., 1., 2.], &[1, 1, 4, 4]);
        assert_eq!(img.max_pool2d(2, 2).to_vec::<f32>(), vec![7., 4., 9., 6.]);
    }

    #[test]
    fn display_matches_pytorch() {
        let a = t(&[1., 2., 3., 4.], &[2, 2]);
        assert_eq!(a.to_string(), "tensor([[1.0000, 2.0000],\n        [3.0000, 4.0000]])");
        let l = Tensor::from_vec(vec![3i64, 1], &[2]);
        assert_eq!(l.to_string(), "tensor([3, 1], dtype=i64)");
    }

    #[test]
    fn missing_cuda_is_an_error() {
        if !cuda_is_available() {
            assert!(Tensor::ones(&[1], DType::F32, Device::Cpu).try_to(Device::Cuda(0)).is_err());
        }
    }
}
