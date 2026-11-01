//! Inference only: builds with `--no-default-features --features infer`, so
//! the binary has no autograd, optimisers or data loaders.
//!
//! ```text
//! cargo run --release --no-default-features --features infer --example infer_mnist -- \
//!     --model mnist_cnn.safetensors --data data
//! ```
//!
//! To ship one self-contained file, embed the weights instead of loading
//! them: `Model::from_bytes(include_bytes!("mnist_cnn.safetensors"))`.

mod common;

use std::time::Instant;

use weta::infer::Model;
use weta::raw::{DType, Device, Tensor};

/// Reads MNIST test images straight from the IDX files (gzip needs the
/// training crates, so this example expects the files `mnist_cnn` already
/// downloaded and unpacked, or raw IDX files).
fn load_test(root: &str) -> Option<(Vec<u8>, Vec<u8>)> {
    let dir = std::path::Path::new(root).join("MNIST").join("raw");
    let images = std::fs::read(dir.join("t10k-images-idx3-ubyte")).ok()?;
    let labels = std::fs::read(dir.join("t10k-labels-idx1-ubyte")).ok()?;
    Some((images[16..].to_vec(), labels[8..].to_vec()))
}

fn main() -> weta::Result<()> {
    let args = common::Args::parse();
    let model = Model::load(args.string("model", "mnist_cnn.safetensors"))?;
    println!("loaded a {}-layer model", model.layers.len());

    let Some((images, labels)) = load_test(&args.string("data", "data")) else {
        // No test set: classify a blank image to show the API.
        let x = Tensor::zeros(&[1, 1, 28, 28], DType::F32, Device::Cpu);
        println!("prediction for a blank image: {:?}", model.predict(&x));
        return Ok(());
    };
    let start = Instant::now();
    let mut correct = 0;
    for (chunk, lbl) in images.chunks(784 * 500).zip(labels.chunks(500)) {
        let x: Vec<f32> = chunk.iter().map(|&b| (b as f32 / 255.0 - 0.1307) / 0.3081).collect();
        let x = Tensor::from_vec(x, &[lbl.len(), 1, 28, 28]);
        correct += model.predict(&x).iter().zip(lbl).filter(|(p, &l)| **p == l as i64).count();
    }
    println!(
        "test accuracy {:.2}% on {} images in {:.2}s",
        100.0 * correct as f64 / labels.len() as f64,
        labels.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
