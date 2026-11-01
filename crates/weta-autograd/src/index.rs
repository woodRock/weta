//! `x.i((.., 0))`: PyTorch-style indexing. Rust's `Index` trait must return
//! a reference, so it can't return a new tensor view; like tch-rs, weta uses
//! a method instead. Each index is a view (narrow or select), so this is
//! differentiable and copies nothing.

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::Tensor;

/// One entry in an index tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorIndexer {
    /// A single position; removes the dimension.
    Select(usize),
    /// `start..end`; `end = None` means to the end.
    Narrow(usize, Option<usize>),
}

impl From<usize> for TensorIndexer {
    fn from(i: usize) -> Self {
        TensorIndexer::Select(i)
    }
}
impl From<Range<usize>> for TensorIndexer {
    fn from(r: Range<usize>) -> Self {
        TensorIndexer::Narrow(r.start, Some(r.end))
    }
}
impl From<RangeInclusive<usize>> for TensorIndexer {
    fn from(r: RangeInclusive<usize>) -> Self {
        TensorIndexer::Narrow(*r.start(), Some(*r.end() + 1))
    }
}
impl From<RangeFrom<usize>> for TensorIndexer {
    fn from(r: RangeFrom<usize>) -> Self {
        TensorIndexer::Narrow(r.start, None)
    }
}
impl From<RangeTo<usize>> for TensorIndexer {
    fn from(r: RangeTo<usize>) -> Self {
        TensorIndexer::Narrow(0, Some(r.end))
    }
}
impl From<RangeToInclusive<usize>> for TensorIndexer {
    fn from(r: RangeToInclusive<usize>) -> Self {
        TensorIndexer::Narrow(0, Some(r.end + 1))
    }
}
impl From<RangeFull> for TensorIndexer {
    fn from(_: RangeFull) -> Self {
        TensorIndexer::Narrow(0, None)
    }
}

/// Implemented for single indices and tuples of up to five.
pub trait IndexOp<T> {
    fn i(&self, index: T) -> Tensor;
}

impl Tensor {
    #[track_caller]
    fn index_with(&self, idx: &[TensorIndexer]) -> Tensor {
        assert!(idx.len() <= self.ndim(), "too many indices ({}) for tensor of shape {:?}", idx.len(), self.shape());
        let mut out = self.clone();
        let mut dim = 0isize;
        for ix in idx {
            match *ix {
                TensorIndexer::Select(i) => {
                    let n = out.size(dim);
                    assert!(i < n, "index {i} is out of bounds for dimension {dim} with size {n}");
                    out = out.select(dim, i);
                }
                TensorIndexer::Narrow(start, end) => {
                    let n = out.size(dim);
                    let end = end.unwrap_or(n).min(n);
                    if start != 0 || end != n {
                        out = out.narrow(dim, start, end.saturating_sub(start));
                    }
                    dim += 1;
                }
            }
        }
        out
    }
}

impl<A: Into<TensorIndexer>> IndexOp<A> for Tensor {
    #[track_caller]
    fn i(&self, a: A) -> Tensor {
        self.index_with(&[a.into()])
    }
}

macro_rules! tuple_index {
    ($($t:ident),+) => {
        impl<$($t: Into<TensorIndexer>),+> IndexOp<($($t,)+)> for Tensor {
            #[track_caller]
            #[allow(non_snake_case)]
            fn i(&self, ($($t,)+): ($($t,)+)) -> Tensor {
                self.index_with(&[$($t.into()),+])
            }
        }
    };
}

tuple_index!(A);
tuple_index!(A, B);
tuple_index!(A, B, C);
tuple_index!(A, B, C, D);
tuple_index!(A, B, C, D, E);
