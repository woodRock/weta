//! A CNN on MNIST (target: at least 99% test accuracy), built from
//! `nn::Sequential` so it can be exported for `weta-infer`.
//!
//! ```text
//! cargo run --release --example mnist_cnn -- --data data --epochs 5
//! cargo run --release --example infer_mnist -- --model mnist_cnn.safetensors --data data
//! ```

mod common;

use std::time::Instant;

use weta::nn::functional as F;
use weta::prelude::*;

fn model() -> nn::Sequential {
    nn::Sequential::new()
        .add(nn::Conv2d::new(1, 32, 3).padding(1)) // [N, 32, 28, 28]
        .add(nn::BatchNorm2d::new(32))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2)) // [N, 32, 14, 14]
        .add(nn::Conv2d::new(32, 64, 3).padding(1)) // [N, 64, 14, 14]
        .add(nn::BatchNorm2d::new(64))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2)) // [N, 64, 7, 7]
        .add(nn::Flatten::default())
        .add(nn::Dropout::new(0.25))
        .add(nn::Linear::new(64 * 7 * 7, 128))
        .add(nn::ReLU)
        .add(nn::Dropout::new(0.25))
        .add(nn::Linear::new(128, 10))
}

fn main() -> weta::Result<()> {
    let args = common::Args::parse();
    let root = args.string("data", "data");
    let epochs: usize = args.get("epochs", 5);
    let limit: usize = args.get("limit", usize::MAX);
    let device = match args.string("device", "auto").as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        _ => Device::cuda_if_available(),
    };
    manual_seed(args.get("seed", 42));

    let (mean, std) = (data::datasets::Mnist::MEAN, data::datasets::Mnist::STD);
    let train_set = data::datasets::Mnist::train(&root)?.normalize(&[mean], &[std]);
    let n = limit.min(train_set.labels().len());
    let train = data::DataLoader::new(data::Subset::first(train_set, n), 128).shuffle(true);
    let test = data::DataLoader::new(data::datasets::Mnist::test(&root)?.normalize(&[mean], &[std]), 1000);

    let mut model = model().to(device);
    println!("CNN with {} parameters on {device}, {n} training images", model.num_parameters());
    let mut opt = optim::AdamW::new(model.parameters(), 2e-3).weight_decay(1e-4);
    let mut sched = optim::lr_scheduler::CosineAnnealingLR::new(&opt, epochs, 1e-5);

    for epoch in 0..epochs {
        let start = Instant::now();
        model.train();
        let (mut sum, mut batches) = (0.0, 0);
        let mut data_time = 0.0;
        let mut t = Instant::now();
        for (x, y) in &train {
            let (x, y) = (x.to(device), y.to(device));
            data_time += t.elapsed().as_secs_f64();
            let loss = F::cross_entropy(&model.forward(&x), &y);
            opt.zero_grad();
            loss.backward();
            opt.step();
            sum += loss.item();
            batches += 1;
            t = Instant::now();
        }
        sched.step(&mut opt);
        model.eval();
        let acc = {
            let _guard = no_grad();
            let (mut correct, mut total) = (0, 0);
            for (x, y) in &test {
                let pred = model.forward(&x.to(device)).argmax(1, false).to(Device::Cpu);
                correct += pred.eq(&y).to_vec::<i64>().iter().filter(|&&v| v == 1).count();
                total += y.numel();
            }
            correct as f64 / total as f64
        };
        println!(
            "epoch {epoch}: mean loss {:.4}, test accuracy {:.2}% ({:.1}s, {data_time:.1}s loading batches)",
            sum / batches as f64,
            acc * 100.0,
            start.elapsed().as_secs_f64()
        );
    }
    if let Some(r) = weta::cuda_profile_report() {
        println!("{r}");
    }
    let model = model.to(Device::Cpu);
    nn::utils::export(&model, "mnist_cnn.safetensors")?;
    println!("exported mnist_cnn.safetensors (weights + architecture) for weta-infer");
    Ok(())
}
