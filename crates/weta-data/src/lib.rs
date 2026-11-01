//! # weta-data
//!
//! Datasets and batching, as `torch.utils.data` and `torchvision.datasets`.
//!
//! ```ignore
//! let train = DataLoader::new(datasets::Mnist::train("data")?, 64).shuffle(true);
//! for (x, y) in &train {
//!     // x: [64, 1, 28, 28] f32 in [0, 1], y: [64] i64
//! }
//! ```
//!
//! Batching is done by the dataset (`get_batch`), so in-memory datasets like
//! MNIST build a whole batch with one copy instead of stacking 64 tensors.

pub mod datasets;

use std::sync::Arc;

use weta_autograd::Tensor;
use weta_tensor::Tensor as Raw;

/// A collection of `(input, target)` samples (`torch.utils.data.Dataset`).
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One sample, without a batch dimension.
    fn get(&self, index: usize) -> (Tensor, Tensor);

    /// Samples at `indices`, stacked along a new first dimension. Override
    /// this when a faster path exists.
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let (xs, ys): (Vec<Tensor>, Vec<Tensor>) = indices.iter().map(|&i| self.get(i)).unzip();
        (Tensor::stack(&xs.iter().collect::<Vec<_>>(), 0), Tensor::stack(&ys.iter().collect::<Vec<_>>(), 0))
    }
}

impl<D: Dataset + ?Sized> Dataset for Arc<D> {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn get(&self, index: usize) -> (Tensor, Tensor) {
        (**self).get(index)
    }
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        (**self).get_batch(indices)
    }
}

/// Rows `indices` of `t` (along dim 0), as one new tensor.
pub fn index_rows(t: &Raw, indices: &[usize]) -> Raw {
    let rows: Vec<Raw> = indices.iter().map(|&i| t.narrow(0, i, 1)).collect();
    Raw::cat(&rows.iter().collect::<Vec<_>>(), 0)
}

/// A dataset of in-memory tensors, indexed along their first dimension
/// (`TensorDataset`).
pub struct TensorDataset {
    pub inputs: Raw,
    pub targets: Raw,
}

impl TensorDataset {
    #[track_caller]
    pub fn new(inputs: &Tensor, targets: &Tensor) -> Self {
        assert_eq!(inputs.size(0), targets.size(0), "TensorDataset: inputs and targets have different lengths");
        TensorDataset { inputs: inputs.data().clone(), targets: targets.data().clone() }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.inputs.size(0)
    }
    fn get(&self, i: usize) -> (Tensor, Tensor) {
        (Tensor::from_raw(self.inputs.select(0, i)), Tensor::from_raw(self.targets.select(0, i)))
    }
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        (Tensor::from_raw(index_rows(&self.inputs, indices)), Tensor::from_raw(index_rows(&self.targets, indices)))
    }
}

/// A subset of another dataset (`torch.utils.data.Subset`).
pub struct Subset<D> {
    pub dataset: D,
    pub indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Subset { dataset, indices }
    }

    /// The first `n` samples.
    pub fn first(dataset: D, n: usize) -> Self {
        let n = n.min(dataset.len());
        Subset { dataset, indices: (0..n).collect() }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    fn len(&self) -> usize {
        self.indices.len()
    }
    fn get(&self, i: usize) -> (Tensor, Tensor) {
        self.dataset.get(self.indices[i])
    }
    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let mapped: Vec<usize> = indices.iter().map(|&i| self.indices[i]).collect();
        self.dataset.get_batch(&mapped)
    }
}

/// Splits a dataset into random, non-overlapping subsets of the given sizes.
pub fn random_split<D: Dataset + Clone>(dataset: D, lengths: &[usize]) -> Vec<Subset<D>> {
    assert_eq!(lengths.iter().sum::<usize>(), dataset.len(), "random_split: lengths don't sum to the dataset length");
    let perm = weta_tensor::randperm(dataset.len());
    let mut start = 0;
    lengths
        .iter()
        .map(|&n| {
            let s = Subset::new(dataset.clone(), perm[start..start + n].to_vec());
            start += n;
            s
        })
        .collect()
}

/// Iterates over a dataset in batches (`torch.utils.data.DataLoader`).
/// Iterate by reference, `for (x, y) in &loader`, to reuse it every epoch;
/// with `shuffle(true)` each pass uses a fresh permutation from the global
/// generator, so runs are reproducible under `manual_seed`.
pub struct DataLoader<D> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl<D: Dataset> DataLoader<D> {
    #[track_caller]
    pub fn new(dataset: D, batch_size: usize) -> Self {
        assert!(batch_size > 0, "DataLoader: batch_size should be a positive integer");
        DataLoader { dataset, batch_size, shuffle: false, drop_last: false }
    }
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub fn dataset(&self) -> &D {
        &self.dataset
    }
    /// The number of batches per pass.
    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> Batches<'_, D> {
        let order = if self.shuffle { weta_tensor::randperm(self.dataset.len()) } else { (0..self.dataset.len()).collect() };
        Batches { loader: self, order, pos: 0 }
    }
}

pub struct Batches<'a, D> {
    loader: &'a DataLoader<D>,
    order: Vec<usize>,
    pos: usize,
}

impl<D: Dataset> Iterator for Batches<'_, D> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<(Tensor, Tensor)> {
        let remaining = self.order.len() - self.pos;
        let bs = self.loader.batch_size;
        if remaining == 0 || (self.loader.drop_last && remaining < bs) {
            return None;
        }
        let end = self.pos + bs.min(remaining);
        let batch = self.loader.dataset.get_batch(&self.order[self.pos..end]);
        self.pos = end;
        Some(batch)
    }
}

impl<'a, D: Dataset> IntoIterator for &'a DataLoader<D> {
    type Item = (Tensor, Tensor);
    type IntoIter = Batches<'a, D>;
    fn into_iter(self) -> Batches<'a, D> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_cover_the_dataset() {
        let x = Tensor::arange(0.0, 10.0, 1.0).reshape(&[10, 1]);
        let y = Tensor::from_vec((0..10i64).collect(), &[10]);
        let loader = DataLoader::new(TensorDataset::new(&x, &y), 4).shuffle(true);
        assert_eq!(loader.len(), 3);
        let mut seen: Vec<i64> = Vec::new();
        for (bx, by) in &loader {
            assert_eq!(bx.size(0), by.size(0));
            seen.extend(by.to_vec::<i64>());
        }
        seen.sort();
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
        let loader = DataLoader::new(TensorDataset::new(&x, &y), 4).drop_last(true);
        assert_eq!(loader.iter().count(), 2);
    }
}
