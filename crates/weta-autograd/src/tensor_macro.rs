//! `tensor!([[1., 2.], [3., 4.]])`: Rust has no nested-list literals, so a
//! trait over nested arrays recovers the shape at compile time. Float
//! literals give an `F32` tensor and integer literals an `I64` one, as
//! `torch.tensor` does.

use crate::Tensor;

/// Scalars and (nested) arrays of them.
pub trait NestedData {
    const IS_FLOAT: bool;
    fn shape(&self, out: &mut Vec<usize>);
    fn flatten(&self, out: &mut Vec<f64>);
}

macro_rules! scalar_impl {
    ($($t:ty => $float:expr),*) => {$(
        impl NestedData for $t {
            const IS_FLOAT: bool = $float;
            fn shape(&self, _: &mut Vec<usize>) {}
            fn flatten(&self, out: &mut Vec<f64>) {
                out.push(*self as f64);
            }
        }
    )*};
}

scalar_impl!(f32 => true, f64 => true, i32 => false, i64 => false);

impl<T: NestedData, const N: usize> NestedData for [T; N] {
    const IS_FLOAT: bool = T::IS_FLOAT;
    fn shape(&self, out: &mut Vec<usize>) {
        out.push(N);
        if let Some(first) = self.first() {
            first.shape(out);
        }
    }
    fn flatten(&self, out: &mut Vec<f64>) {
        for x in self {
            x.flatten(out);
        }
    }
}

impl<T: NestedData> NestedData for Vec<T> {
    const IS_FLOAT: bool = T::IS_FLOAT;
    fn shape(&self, out: &mut Vec<usize>) {
        out.push(self.len());
        if let Some(first) = self.first() {
            first.shape(out);
        }
    }
    fn flatten(&self, out: &mut Vec<f64>) {
        for x in self {
            x.flatten(out);
        }
    }
}

pub(crate) fn from_nested<T: NestedData>(v: T) -> Tensor {
    let mut shape = Vec::new();
    v.shape(&mut shape);
    let mut data = Vec::new();
    v.flatten(&mut data);
    if T::IS_FLOAT {
        Tensor::from_vec(data.into_iter().map(|x| x as f32).collect::<Vec<f32>>(), &shape)
    } else {
        Tensor::from_vec(data.into_iter().map(|x| x as i64).collect::<Vec<i64>>(), &shape)
    }
}

/// Builds a tensor from a literal, like `torch.tensor`:
///
/// ```
/// use weta_autograd::tensor;
/// let x = tensor!([[1., 2.], [3., 4.]]);
/// assert_eq!(x.shape(), &[2, 2]);
/// let labels = tensor!([3, 1, 4]);
/// assert_eq!(labels.dtype(), weta_autograd::DType::I64);
/// ```
#[macro_export]
macro_rules! tensor {
    ($e:expr) => {
        $crate::Tensor::from_nested($e)
    };
}
