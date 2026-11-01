//! The spec's first target: an MLP on MNIST, at least 98% test accuracy.
//!
//! ```text
//! cargo run --release --example mnist_mlp -- --data data --epochs 10
//! ```

mod common;

use std::time::Instant;

use weta::nn::functional as F;
use weta::prelude::*;

#[derive(Module)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    drop: nn::Dropout,
}

impl Net {
    fn new() -> Self {
        Net { fc1: nn::Linear::new(784, 512), fc2: nn::Linear::new(512, 256), fc3: nn::Linear::new(256, 10), drop: nn::Dropout::new(0.2) }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.flatten(1, -1);
        let x = self.drop.forward(&F::relu(&self.fc1.forward(&x)));
        let x = self.drop.forward(&F::relu(&self.fc2.forward(&x)));
        self.fc3.forward(&x) // logits
    }
}

/// Fraction of correct predictions, without recording a graph.
fn accuracy(model: &Net, loader: &data::DataLoader<data::datasets::ImageDataset>, device: Device) -> f64 {
    let _guard = no_grad();
    let (mut correct, mut total) = (0usize, 0usize);
    for (x, y) in loader {
        let pred = model.forward(&x.to(device)).argmax(1, false).to(Device::Cpu);
        correct += pred.eq(&y).to_vec::<i64>().iter().filter(|&&v| v == 1).count();
        total += y.numel();
    }
    correct as f64 / total as f64
}

fn main() -> weta::Result<()> {
    let args = common::Args::parse();
    let root = args.string("data", "data");
    let epochs: usize = args.get("epochs", 10);
    let device = match args.string("device", "auto").as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        _ => Device::cuda_if_available(),
    };
    manual_seed(args.get("seed", 42));

    let (mean, std) = (data::datasets::Mnist::MEAN, data::datasets::Mnist::STD);
    let train = data::DataLoader::new(data::datasets::Mnist::train(&root)?.normalize(&[mean], &[std]), 128).shuffle(true);
    let test = data::DataLoader::new(data::datasets::Mnist::test(&root)?.normalize(&[mean], &[std]), 1000);

    let mut model = Net::new().to(device);
    println!("MLP with {} parameters on {device}", model.num_parameters());
    let mut opt = optim::Adam::new(model.parameters(), 1e-3);
    let mut sched = optim::lr_scheduler::CosineAnnealingLR::new(&opt, epochs, 1e-5);

    for epoch in 0..epochs {
        let start = Instant::now();
        model.train();
        let mut last = 0.0;
        for (x, y) in &train {
            let (x, y) = (x.to(device), y.to(device));
            let loss = F::cross_entropy(&model.forward(&x), &y);
            opt.zero_grad();
            loss.backward();
            opt.step();
            last = loss.item();
        }
        sched.step(&mut opt);
        model.eval();
        let acc = accuracy(&model, &test, device);
        println!("epoch {epoch}: last loss {last:.4}, test accuracy {:.2}% ({:.1}s)", acc * 100.0, start.elapsed().as_secs_f64());
    }
    model.state_dict().save("mnist_mlp.safetensors")?;
    println!("saved mnist_mlp.safetensors");
    Ok(())
}
