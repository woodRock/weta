//! How a tensor's elements are found in its storage.
//!
//! Element `[i0, i1, ..., in]` lives at `offset + Σ ik * strides[k]`. A
//! contiguous (row-major) tensor of shape `[2, 3]` has strides `[3, 1]`.
//! Views change only the layout:
//!
//! - **transpose** swaps two strides: `[3, 1]` → `[1, 3]`.
//! - **narrow** (slicing) moves the offset and shrinks one dimension.
//! - **broadcast** gives a size-1 or missing dimension stride `0`, so every
//!   index along it reads the same element.
//!
//! None of these copy data, which is why `x.t()` costs nothing.

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

/// Row-major strides for a shape: the last dimension moves fastest.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1].max(1);
    }
    strides
}

impl Layout {
    pub fn contiguous(shape: &[usize]) -> Self {
        Layout { shape: shape.to_vec(), strides: contiguous_strides(shape), offset: 0 }
    }

    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        assert_eq!(shape.len(), strides.len());
        Layout { shape, strides, offset }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// True if the elements are laid out row-major with no gaps, starting at
    /// `offset`. Size-1 dimensions may have any stride.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        for (&d, &s) in self.shape.iter().zip(&self.strides).rev() {
            if d != 1 && s != expected {
                return false;
            }
            expected *= d;
        }
        true
    }

    /// The range of storage indices this layout can touch (for bounds checks).
    pub fn max_index(&self) -> usize {
        if self.numel() == 0 {
            return self.offset;
        }
        self.offset + self.shape.iter().zip(&self.strides).map(|(&d, &s)| (d - 1) * s).sum::<usize>()
    }

    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Layout> {
        let n = self.ndim();
        if d0 >= n || d1 >= n {
            return Err(Error::shape("transpose", format!("dims ({d0}, {d1}) out of range for {} dims", n)));
        }
        let mut l = self.clone();
        l.shape.swap(d0, d1);
        l.strides.swap(d0, d1);
        Ok(l)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Layout> {
        let n = self.ndim();
        let mut seen = vec![false; n];
        if dims.len() != n || dims.iter().any(|&d| d >= n || std::mem::replace(&mut seen[d], true)) {
            return Err(Error::shape("permute", format!("{dims:?} is not a permutation of {n} dims")));
        }
        Ok(Layout {
            shape: dims.iter().map(|&d| self.shape[d]).collect(),
            strides: dims.iter().map(|&d| self.strides[d]).collect(),
            offset: self.offset,
        })
    }

    /// `len` elements of dimension `dim`, starting at `start`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Layout> {
        if dim >= self.ndim() || start + len > self.shape[dim] {
            return Err(Error::shape("narrow", format!("can't take [{start}, {}) of dim {dim} in shape {:?}", start + len, self.shape)));
        }
        let mut l = self.clone();
        l.offset += start * self.strides[dim];
        l.shape[dim] = len;
        Ok(l)
    }

    /// Every `step`-th element of dimension `dim` (used by strided slicing).
    pub fn step(&self, dim: usize, step: usize) -> Layout {
        let mut l = self.clone();
        l.shape[dim] = self.shape[dim].div_ceil(step);
        l.strides[dim] *= step;
        l
    }

    pub fn unsqueeze(&self, dim: usize) -> Result<Layout> {
        if dim > self.ndim() {
            return Err(Error::shape("unsqueeze", format!("dim {dim} out of range for shape {:?}", self.shape)));
        }
        let mut l = self.clone();
        let stride = if dim < self.ndim() { self.strides[dim] * self.shape[dim] } else { 1 };
        l.shape.insert(dim, 1);
        l.strides.insert(dim, stride);
        Ok(l)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Layout> {
        if dim >= self.ndim() || self.shape[dim] != 1 {
            return Err(Error::shape("squeeze", format!("dim {dim} of shape {:?} is not size 1", self.shape)));
        }
        let mut l = self.clone();
        l.shape.remove(dim);
        l.strides.remove(dim);
        Ok(l)
    }

    /// A view with `shape`, following NumPy's broadcasting rules: dimensions
    /// are matched from the right, and a size-1 or missing dimension is
    /// stretched by giving it stride 0.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Layout> {
        if shape.len() < self.ndim() {
            return Err(Error::shape("broadcast_to", format!("can't broadcast {:?} to {shape:?}", self.shape)));
        }
        let extra = shape.len() - self.ndim();
        let mut strides = vec![0; shape.len()];
        for (i, &target) in shape.iter().enumerate().skip(extra) {
            let d = self.shape[i - extra];
            if d == target {
                strides[i] = self.strides[i - extra];
            } else if d != 1 {
                return Err(Error::shape("broadcast_to", format!("can't broadcast {:?} to {shape:?}", self.shape)));
            }
        }
        Ok(Layout { shape: shape.to_vec(), strides, offset: self.offset })
    }

    /// A view with a new shape over the same elements, if the layout allows
    /// it without copying (here: when contiguous). `None` means "copy first".
    pub fn reshape(&self, shape: &[usize]) -> Option<Layout> {
        if self.is_contiguous() {
            Some(Layout { shape: shape.to_vec(), strides: contiguous_strides(shape), offset: self.offset })
        } else {
            None
        }
    }

    /// Iterates over the storage index of every element, in row-major order.
    pub fn offsets(&self) -> StridedOffsets<'_> {
        StridedOffsets::new(self)
    }
}

/// Resolves `-1` in a requested shape, and checks the element count matches.
pub fn resolve_shape(op: &'static str, numel: usize, shape: &[isize]) -> Result<Vec<usize>> {
    let mut infer = None;
    let mut known = 1usize;
    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            if infer.replace(i).is_some() {
                return Err(Error::shape(op, "only one dimension can be -1"));
            }
        } else if d < 0 {
            return Err(Error::shape(op, format!("invalid dimension {d}")));
        } else {
            known *= d as usize;
        }
    }
    let mut out: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    if let Some(i) = infer {
        if known == 0 || !numel.is_multiple_of(known) {
            return Err(Error::shape(op, format!("can't infer -1 in {shape:?} for {numel} elements")));
        }
        out[i] = numel / known;
    } else if known != numel {
        return Err(Error::shape(op, format!("shape {shape:?} is invalid for {numel} elements")));
    }
    Ok(out)
}

/// The shape two tensors broadcast to, e.g. `[64, 1] and [10]` → `[64, 10]`.
pub fn broadcast_shapes(op: &'static str, a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(Error::shape(op, format!("shapes {a:?} and {b:?} can't be broadcast together")));
            }
        };
    }
    Ok(out)
}

/// An odometer over a strided layout, yielding storage indices.
pub struct StridedOffsets<'a> {
    layout: &'a Layout,
    index: Vec<usize>,
    next: Option<usize>,
}

impl<'a> StridedOffsets<'a> {
    fn new(layout: &'a Layout) -> Self {
        let next = if layout.numel() == 0 { None } else { Some(layout.offset) };
        StridedOffsets { layout, index: vec![0; layout.ndim()], next }
    }
}

impl Iterator for StridedOffsets<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let cur = self.next?;
        let mut off = cur;
        let mut d = self.layout.ndim();
        loop {
            if d == 0 {
                self.next = None;
                break;
            }
            d -= 1;
            self.index[d] += 1;
            off += self.layout.strides[d];
            if self.index[d] < self.layout.shape[d] {
                self.next = Some(off);
                break;
            }
            off -= self.layout.strides[d] * self.index[d];
            self.index[d] = 0;
        }
        Some(cur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_and_contiguity() {
        let l = Layout::contiguous(&[2, 3, 4]);
        assert_eq!(l.strides(), &[12, 4, 1]);
        assert!(l.is_contiguous());
        let t = l.transpose(1, 2).unwrap();
        assert_eq!(t.shape(), &[2, 4, 3]);
        assert!(!t.is_contiguous());
    }

    #[test]
    fn offsets_follow_strides() {
        let t = Layout::contiguous(&[2, 3]).transpose(0, 1).unwrap();
        assert_eq!(t.offsets().collect::<Vec<_>>(), vec![0, 3, 1, 4, 2, 5]);
        let n = Layout::contiguous(&[3, 4]).narrow(1, 1, 2).unwrap();
        assert_eq!(n.offsets().collect::<Vec<_>>(), vec![1, 2, 5, 6, 9, 10]);
    }

    #[test]
    fn broadcasting() {
        assert_eq!(broadcast_shapes("t", &[64, 1], &[10]).unwrap(), vec![64, 10]);
        assert_eq!(broadcast_shapes("t", &[2, 1, 3], &[4, 1]).unwrap(), vec![2, 4, 3]);
        assert!(broadcast_shapes("t", &[2, 3], &[4]).is_err());
        let b = Layout::contiguous(&[3]).broadcast_to(&[2, 3]).unwrap();
        assert_eq!(b.strides(), &[0, 1]);
        assert_eq!(b.offsets().collect::<Vec<_>>(), vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn resolve_infers_minus_one() {
        assert_eq!(resolve_shape("r", 12, &[-1, 4]).unwrap(), vec![3, 4]);
        assert!(resolve_shape("r", 12, &[5, -1]).is_err());
    }
}
