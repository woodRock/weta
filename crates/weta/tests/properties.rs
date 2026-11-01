//! Property tests over random shapes.

use proptest::prelude::*;
use weta::raw::{allclose, DType, Device, Tensor};

fn shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..5, 1..5)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn reshape_round_trips(s in shape()) {
        let x = Tensor::randn(&s, DType::F32, Device::Cpu);
        let flat = x.reshape(&[-1]);
        prop_assert_eq!(flat.shape(), &[x.numel()][..]);
        prop_assert_eq!(flat.reshape_to(&s).to_vec::<f32>(), x.to_vec::<f32>());
    }

    #[test]
    fn double_transpose_is_identity(s in prop::collection::vec(1usize..5, 2..5), a in 0usize..4, b in 0usize..4) {
        let x = Tensor::randn(&s, DType::F32, Device::Cpu);
        let (a, b) = ((a % s.len()) as isize, (b % s.len()) as isize);
        let y = x.transpose(a, b).transpose(a, b);
        prop_assert_eq!(y.shape(), x.shape());
        prop_assert_eq!(y.to_vec::<f32>(), x.to_vec::<f32>());
    }

    #[test]
    fn broadcasting_matches_explicit_expansion(s in shape(), mask in prop::collection::vec(any::<bool>(), 4)) {
        // b has size 1 wherever mask is set.
        let bs: Vec<usize> = s.iter().zip(mask.iter().cycle()).map(|(&d, &m)| if m { 1 } else { d }).collect();
        let a = Tensor::randn(&s, DType::F64, Device::Cpu);
        let b = Tensor::randn(&bs, DType::F64, Device::Cpu);
        let expanded = b.broadcast_to(&s).contiguous();
        prop_assert!(allclose(&a.add(&b), &a.add(&expanded), 0.0, 0.0));
        prop_assert!(allclose(&a.mul(&b), &b.mul(&a), 0.0, 0.0)); // commutative
    }

    #[test]
    fn sum_over_all_dims_equals_total(s in shape()) {
        let x = Tensor::randn(&s, DType::F64, Device::Cpu);
        let dims: Vec<isize> = (0..s.len() as isize).collect();
        let total: f64 = x.to_vec::<f64>().iter().sum();
        prop_assert!((x.sum_dims(&dims, false).item() - total).abs() < 1e-9);
        prop_assert!((x.sum().item() - total).abs() < 1e-9);
    }

    #[test]
    fn matmul_of_transposes(m in 1usize..6, k in 1usize..6, n in 1usize..6) {
        // (AB)ᵀ = BᵀAᵀ, with the transposes as non-contiguous views.
        let a = Tensor::randn(&[m, k], DType::F64, Device::Cpu);
        let b = Tensor::randn(&[k, n], DType::F64, Device::Cpu);
        prop_assert!(allclose(&a.matmul(&b).t().contiguous(), &b.t().matmul(&a.t()), 1e-12, 1e-12));
    }
}
