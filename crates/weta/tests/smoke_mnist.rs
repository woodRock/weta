//! Smoke training: an MLP reaches at least 90% on a 5,000-image MNIST
//! subset within 60 seconds. Downloads MNIST to `target/mnist-data` on first
//! run; if the download fails (offline), the test reports that and passes.

use std::time::Instant;

use weta::nn::functional as F;
use weta::prelude::*;

#[test]
fn mlp_reaches_90_percent_on_a_mnist_subset() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/mnist-data");
    let (train, test) = match (data::datasets::Mnist::train(root), data::datasets::Mnist::test(root)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("skipping: MNIST unavailable ({e})");
            return;
        }
    };
    manual_seed(0);
    let start = Instant::now();
    let (m, sd) = (data::datasets::Mnist::MEAN, data::datasets::Mnist::STD);
    let train = data::DataLoader::new(data::Subset::first(train.normalize(&[m], &[sd]), 5000), 64).shuffle(true);
    let test = data::DataLoader::new(data::Subset::first(test.normalize(&[m], &[sd]), 2000), 500);
    let model =
        nn::Sequential::new().add(nn::Flatten::default()).add(nn::Linear::new(784, 256)).add(nn::ReLU).add(nn::Linear::new(256, 10));
    let mut opt = optim::Adam::new(model.parameters(), 1e-3);
    for _ in 0..10 {
        for (x, y) in &train {
            let loss = F::cross_entropy(&model.forward(&x), &y);
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
    }
    let _g = no_grad();
    let (mut correct, mut total) = (0, 0);
    for (x, y) in &test {
        let pred = model.forward(&x).argmax(1, false);
        correct += pred.eq(&y).to_vec::<i64>().iter().sum::<i64>();
        total += y.numel() as i64;
    }
    let acc = correct as f64 / total as f64;
    let secs = start.elapsed().as_secs_f64();
    println!("accuracy {:.2}% in {secs:.1}s", acc * 100.0);
    assert!(acc >= 0.90, "accuracy {acc}");
    assert!(secs < 60.0, "took {secs}s");
}
