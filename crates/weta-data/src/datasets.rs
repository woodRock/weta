//! Standard datasets, as in torchvision: downloaded on first use and cached
//! under the root directory you pass, so nothing large lives in the repo.

use std::io::Read;
use std::path::{Path, PathBuf};

use weta_autograd::Tensor;
use weta_core::{Error, Result};
use weta_tensor::Tensor as Raw;

use crate::Dataset;

/// Downloads `url` to `path` unless it already exists.
fn fetch(urls: &[String], path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    #[cfg(feature = "download")]
    {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut last_err = String::new();
        for url in urls {
            eprintln!("downloading {url}");
            match ureq::get(url).call() {
                Ok(resp) => {
                    let mut bytes = Vec::new();
                    resp.into_reader().read_to_end(&mut bytes)?;
                    // Write to a temporary name first so an interrupted
                    // download never leaves a truncated file behind.
                    let tmp = path.with_extension("part");
                    std::fs::write(&tmp, &bytes)?;
                    std::fs::rename(&tmp, path)?;
                    return Ok(());
                }
                Err(e) => last_err = format!("{url}: {e}"),
            }
        }
        Err(Error::Io(format!("download failed: {last_err}")))
    }
    #[cfg(not(feature = "download"))]
    {
        Err(Error::Io(format!("{} not found, and weta-data was built without the `download` feature ({urls:?})", path.display())))
    }
}

fn gunzip(path: &Path) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    if bytes.starts_with(&[0x1f, 0x8b]) {
        #[cfg(feature = "download")]
        {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(&bytes[..]).read_to_end(&mut out)?;
            return Ok(out);
        }
        #[cfg(not(feature = "download"))]
        return Err(Error::Io(format!("{}: gzip support needs the `download` feature", path.display())));
    }
    Ok(bytes)
}

/// Parses an IDX file (the MNIST format): a magic number whose last byte is
/// the number of dimensions, the dimensions as big-endian u32s, then bytes.
fn parse_idx(bytes: &[u8]) -> Result<(Vec<usize>, &[u8])> {
    if bytes.len() < 4 || bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0x08 {
        return Err(Error::Io("not an unsigned-byte IDX file".into()));
    }
    let nd = bytes[3] as usize;
    let mut dims = Vec::with_capacity(nd);
    for i in 0..nd {
        let b = &bytes[4 + 4 * i..8 + 4 * i];
        dims.push(u32::from_be_bytes(b.try_into().unwrap()) as usize);
    }
    let data = &bytes[4 + 4 * nd..];
    if data.len() != dims.iter().product::<usize>() {
        return Err(Error::Io("IDX file is truncated".into()));
    }
    Ok((dims, data))
}

/// Images stored as bytes, with integer labels: the shared shape of MNIST
/// and CIFAR. Batches are built straight from the bytes.
#[derive(Clone)]
pub struct ImageDataset {
    images: std::sync::Arc<Vec<u8>>,
    labels: std::sync::Arc<Vec<u8>>,
    /// Per-image shape, e.g. `[1, 28, 28]`.
    shape: Vec<usize>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl ImageDataset {
    fn new(images: Vec<u8>, labels: Vec<u8>, shape: Vec<usize>) -> Self {
        let c = shape[0];
        ImageDataset { images: images.into(), labels: labels.into(), shape, mean: vec![0.0; c], std: vec![1.0; c] }
    }

    /// Normalises each channel as `(x − mean) / std` after scaling to [0, 1]
    /// (torchvision's `Normalize`).
    pub fn normalize(mut self, mean: &[f32], std: &[f32]) -> Self {
        assert_eq!(mean.len(), self.shape[0]);
        self.mean = mean.to_vec();
        self.std = std.to_vec();
        self
    }

    pub fn image_shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn labels(&self) -> &[u8] {
        &self.labels
    }
}

impl Dataset for ImageDataset {
    fn len(&self) -> usize {
        self.labels.len()
    }

    fn get(&self, i: usize) -> (Tensor, Tensor) {
        let (x, y) = self.get_batch(&[i]);
        (x.squeeze(0), y.squeeze(0))
    }

    fn get_batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let per: usize = self.shape.iter().product();
        let plane = per / self.shape[0];
        let mut x = Vec::with_capacity(indices.len() * per);
        for &i in indices {
            let img = &self.images[i * per..(i + 1) * per];
            for (j, &b) in img.iter().enumerate() {
                let c = j / plane;
                x.push((b as f32 / 255.0 - self.mean[c]) / self.std[c]);
            }
        }
        let y: Vec<i64> = indices.iter().map(|&i| self.labels[i] as i64).collect();
        let mut shape = vec![indices.len()];
        shape.extend(&self.shape);
        (Tensor::from_raw(Raw::from_vec(x, &shape)), Tensor::from_raw(Raw::from_vec(y, &[indices.len()])))
    }
}

/// MNIST handwritten digits (LeCun et al., 1998): 60,000 training and
/// 10,000 test images, `[1, 28, 28]` in [0, 1], with labels 0–9.
pub struct Mnist;

const MNIST_MIRRORS: [&str; 2] = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"];

impl Mnist {
    /// The usual normalisation constants (mean and std of the training set).
    pub const MEAN: f32 = 0.1307;
    pub const STD: f32 = 0.3081;

    pub fn train(root: impl AsRef<Path>) -> Result<ImageDataset> {
        Self::load(root.as_ref(), "train")
    }

    pub fn test(root: impl AsRef<Path>) -> Result<ImageDataset> {
        Self::load(root.as_ref(), "t10k")
    }

    fn load(root: &Path, split: &str) -> Result<ImageDataset> {
        let dir: PathBuf = root.join("MNIST").join("raw");
        let read = |name: String| -> Result<Vec<u8>> {
            let path = dir.join(&name);
            let urls: Vec<String> = MNIST_MIRRORS.iter().map(|m| format!("{m}{name}")).collect();
            fetch(&urls, &path)?;
            gunzip(&path)
        };
        let img_bytes = read(format!("{split}-images-idx3-ubyte.gz"))?;
        let lbl_bytes = read(format!("{split}-labels-idx1-ubyte.gz"))?;
        let (idims, images) = parse_idx(&img_bytes)?;
        let (ldims, labels) = parse_idx(&lbl_bytes)?;
        if idims.len() != 3 || ldims.len() != 1 || idims[0] != ldims[0] {
            return Err(Error::Io(format!("unexpected MNIST shapes {idims:?} and {ldims:?}")));
        }
        Ok(ImageDataset::new(images.to_vec(), labels.to_vec(), vec![1, idims[1], idims[2]]))
    }
}

/// CIFAR-10 (Krizhevsky, 2009): 50,000 training and 10,000 test colour
/// images, `[3, 32, 32]`, in 10 classes.
pub struct Cifar10;

impl Cifar10 {
    pub const CLASSES: [&'static str; 10] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

    pub fn train(root: impl AsRef<Path>) -> Result<ImageDataset> {
        Self::load(root.as_ref(), &["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"])
    }

    pub fn test(root: impl AsRef<Path>) -> Result<ImageDataset> {
        Self::load(root.as_ref(), &["test_batch.bin"])
    }

    fn load(root: &Path, files: &[&str]) -> Result<ImageDataset> {
        let archive = root.join("cifar-10-binary.tar.gz");
        fetch(&["https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz".to_string()], &archive)?;
        let tar = gunzip(&archive)?;
        let (mut images, mut labels) = (Vec::new(), Vec::new());
        for f in files {
            let rec = tar_entry(&tar, f).ok_or_else(|| Error::Io(format!("{f} not found in {}", archive.display())))?;
            // Each record: 1 label byte, then 3072 pixel bytes (R, G, B planes).
            for r in rec.chunks_exact(3073) {
                labels.push(r[0]);
                images.extend_from_slice(&r[1..]);
            }
        }
        Ok(ImageDataset::new(images, labels, vec![3, 32, 32]))
    }
}

/// Finds a file in an uncompressed tar archive by the end of its name.
/// (Tar is 512-byte headers, each followed by the file's data padded to 512.)
fn tar_entry<'a>(tar: &'a [u8], name: &str) -> Option<&'a [u8]> {
    let mut pos = 0;
    while pos + 512 <= tar.len() {
        let h = &tar[pos..pos + 512];
        if h.iter().all(|&b| b == 0) {
            return None;
        }
        let fname = String::from_utf8_lossy(&h[..100]).trim_end_matches('\0').to_string();
        let size_s = String::from_utf8_lossy(&h[124..136]).trim_matches(|c: char| c == '\0' || c == ' ').to_string();
        let size = usize::from_str_radix(&size_s, 8).ok()?;
        let start = pos + 512;
        if fname.ends_with(name) {
            return tar.get(start..start + size);
        }
        pos = start + size.div_ceil(512) * 512;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx_parsing() {
        let mut b = vec![0, 0, 8, 3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2];
        b.extend([10, 20, 30, 40]);
        let (dims, data) = parse_idx(&b).unwrap();
        assert_eq!(dims, vec![2, 1, 2]);
        assert_eq!(data, &[10, 20, 30, 40]);
        let ds = ImageDataset::new(data.to_vec(), vec![3, 7], vec![1, 1, 2]);
        let (x, y) = ds.get_batch(&[1, 0]);
        assert_eq!(x.shape(), &[2, 1, 1, 2]);
        assert_eq!(y.to_vec::<i64>(), vec![7, 3]);
        assert!((x.to_vec::<f32>()[0] - 30.0 / 255.0).abs() < 1e-6);
    }
}
