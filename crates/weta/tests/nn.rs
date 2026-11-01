//! Layers, losses, modules and serialisation, through the public API.

#![allow(clippy::cloned_ref_to_slice_refs)]

use weta::autograd::{gradcheck, GradcheckOptions, RawTensor};
use weta::nn::functional as F;
use weta::prelude::*;

fn leaf(shape: &[usize]) -> Tensor {
    Tensor::from_raw(RawTensor::randn(shape, DType::F64, Device::Cpu)).requires_grad_(true)
}

#[track_caller]
fn check(f: impl Fn(&[Tensor]) -> Tensor, inputs: &[Tensor]) {
    if let Err(e) = gradcheck(f, inputs, GradcheckOptions::default()) {
        panic!("{e}");
    }
}

#[test]
fn functional_gradchecks() {
    let (x, w, b) = (leaf(&[2, 3, 6, 6]), leaf(&[4, 3, 3, 3]), leaf(&[4]));
    check(|t| F::conv2d(&t[0], &t[1], Some(&t[2]), 1, 1), &[x.clone(), w.clone(), b.clone()]);
    check(|t| F::conv2d(&t[0], &t[1], None, 2, 0), &[x.clone(), w.clone()]);
    check(|t| F::conv2d_ext(&t[0], &t[1], None, (1, 1), (2, 2), (2, 2)), &[x.clone(), w]);
    check(|t| F::max_pool2d(&t[0], 2, 2), &[x.clone()]);
    check(|t| F::max_pool2d(&t[0], 3, 1), &[x.clone()]);
    check(|t| F::avg_pool2d(&t[0], 2, 2, 1), &[x.clone()]);
    let (g, bb) = (leaf(&[3]), leaf(&[3]));
    check(|t| F::batch_norm(&t[0], None, None, Some(&t[1]), Some(&t[2]), true, 0.1, 1e-5), &[x.clone(), g, bb]);
    let (lw, lb) = (leaf(&[6]), leaf(&[6]));
    check(|t| F::layer_norm(&t[0], 1, Some(&t[1]), Some(&t[2]), 1e-5), &[x.clone(), lw, lb]);
    let (a, y) = (leaf(&[5, 4]), Tensor::from_raw(RawTensor::rand(&[5, 4], DType::F64, Device::Cpu)));
    check(|t| F::mse_loss(&t[0], &y), &[a.clone()]);
    check(|t| F::binary_cross_entropy_with_logits(&t[0], &y), &[a.clone()]);
    check(|t| F::gelu(&t[0]), &[a.clone()]);
    check(|t| F::gelu_tanh(&t[0]), &[a.clone()]);
    check(|t| F::silu(&t[0]), &[a.clone()]);
    check(|t| F::leaky_relu(&t[0], 0.1), &[a.clone()]);
    let ids = Tensor::from_vec(vec![0i64, 3, 1, 2, 3], &[5]);
    check(|t| F::nll_loss(&t[0].log_softmax(1), &ids), &[a.clone()]);
    check(|t| F::linear(&t[0], &t[1], Some(&t[2])), &[a, leaf(&[3, 4]), leaf(&[3])]);
}

#[test]
fn bce_with_logits_is_stable_and_correct() {
    let x = tensor!([-100.0, 0.0, 100.0]);
    let y = tensor!([0.0, 1.0, 1.0]);
    let l = F::binary_cross_entropy_with_logits_with(&x, &y, nn::Reduction::None).to_vec::<f32>();
    assert!(l.iter().all(|v| v.is_finite()));
    assert!((l[1] - std::f32::consts::LN_2).abs() < 1e-6 && l[0] < 1e-6 && l[2] < 1e-6);
}

#[derive(Module)]
struct Net {
    features: nn::Sequential,
    head: nn::Linear,
    extra: Option<nn::Linear>,
    #[module(skip)]
    _note: String,
}

impl Net {
    fn new() -> Self {
        Net {
            features: nn::Sequential::new()
                .add(nn::Conv2d::new(1, 4, 3).padding(1))
                .add(nn::BatchNorm2d::new(4))
                .add(nn::ReLU)
                .add(nn::MaxPool2d::new(2)),
            head: nn::Linear::new(4 * 4 * 4, 10),
            extra: None,
            _note: "unused".into(),
        }
    }
    fn forward(&self, x: &Tensor) -> Tensor {
        self.head.forward(&self.features.forward(x).flatten(1, -1))
    }
}

#[test]
fn derive_names_parameters_like_pytorch() {
    let net = Net::new();
    let names: Vec<String> = net.named_parameters().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["features.0.weight", "features.0.bias", "features.1.weight", "features.1.bias", "head.weight", "head.bias"]);
    let sd = net.state_dict();
    assert!(sd.contains_key("features.1.running_mean") && sd.contains_key("features.1.num_batches_tracked"));
    assert_eq!(net.parameters().len(), 6);
    assert_eq!(net.num_parameters(), 4 * 9 + 4 + 4 + 4 + 64 * 10 + 10);
    assert!(net.extra.is_none());
    let y = net.forward(&Tensor::randn(&[2, 1, 8, 8]));
    assert_eq!(y.shape(), &[2, 10]);
}

#[test]
fn train_and_eval_modes() {
    let mut net = Net::new();
    let x = Tensor::randn(&[4, 1, 8, 8]);
    let before = net.features.state_dict().get("1.running_mean").unwrap().to_vec::<f32>();
    net.forward(&x);
    let after = net.features.state_dict().get("1.running_mean").unwrap().to_vec::<f32>();
    assert_ne!(before, after, "running stats update in train mode");
    net.eval();
    let a = net.forward(&x).to_vec::<f32>();
    let b = net.forward(&x).to_vec::<f32>();
    assert_eq!(a, b, "eval mode is deterministic");
    assert_eq!(net.features.state_dict().get("1.running_mean").unwrap().to_vec::<f32>(), after);

    let mut d = nn::Dropout::new(0.5);
    let ones = Tensor::ones(&[1000]);
    let y = d.forward(&ones).to_vec::<f32>();
    let zeros = y.iter().filter(|&&v| v == 0.0).count();
    assert!((400..600).contains(&zeros) && y.iter().all(|&v| v == 0.0 || v == 2.0));
    d.eval();
    assert_eq!(d.forward(&ones).to_vec::<f32>(), vec![1.0; 1000]);
}

#[test]
fn state_dict_round_trip_through_safetensors() {
    let dir = std::env::temp_dir().join(format!("weta-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("net.safetensors");
    let a = Net::new();
    a.forward(&Tensor::randn(&[2, 1, 8, 8])); // move BatchNorm's running stats
    a.state_dict().save(&path).unwrap();
    let mut b = Net::new();
    let opt_params = b.parameters();
    b.load_state_dict(&weta::io::StateDict::load(&path).unwrap(), true).unwrap();
    for ((n, p), (_, q)) in a.state_dict().iter().zip(b.state_dict().iter()) {
        assert_eq!(p.to_vec::<f64>(), q.to_vec::<f64>(), "{n}");
    }
    // Loading copies in place, so handles taken earlier see the new values.
    assert_eq!(opt_params[0].to_vec::<f32>(), a.parameters()[0].to_vec::<f32>());
    // Strict loading reports problems clearly.
    let mut sd = a.state_dict();
    sd.remove("head.bias");
    let err = b.load_state_dict(&sd, true).unwrap_err().to_string();
    assert!(err.contains("missing keys") && err.contains("head.bias"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn exported_model_runs_in_weta_infer() {
    manual_seed(0);
    let mut model = nn::Sequential::new()
        .add(nn::Conv2d::new(1, 4, 3).padding(1))
        .add(nn::BatchNorm2d::new(4))
        .add(nn::ReLU)
        .add(nn::MaxPool2d::new(2))
        .add(nn::Flatten::default())
        .add(nn::Dropout::new(0.2))
        .add(nn::Linear::new(4 * 4 * 4, 10));
    model.forward(&Tensor::randn(&[8, 1, 8, 8]));
    model.eval();
    let path = std::env::temp_dir().join(format!("weta-export-{}.safetensors", std::process::id()));
    nn::utils::export(&model, &path).unwrap();
    let deployed = weta::infer::Model::load(&path).unwrap();
    let x = Tensor::randn(&[3, 1, 8, 8]);
    let want = model.forward(&x).to_vec::<f32>();
    let got = deployed.forward(x.data()).to_vec::<f32>();
    for (a, b) in want.iter().zip(&got) {
        assert!((a - b).abs() < 1e-4, "{want:?} vs {got:?}");
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn mlp_learns_xor() {
    manual_seed(1);
    let x = tensor!([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]);
    let y = tensor!([0, 1, 1, 0]);
    let model = nn::Sequential::new().add(nn::Linear::new(2, 16)).add(nn::Tanh).add(nn::Linear::new(16, 2));
    let mut opt = optim::Adam::new(model.parameters(), 0.05);
    for _ in 0..300 {
        let loss = F::cross_entropy(&model.forward(&x), &y);
        opt.zero_grad();
        loss.backward();
        opt.step();
    }
    let pred = model.forward(&x).argmax(1, false).to_vec::<i64>();
    assert_eq!(pred, vec![0, 1, 1, 0]);
}

#[test]
fn shape_errors_name_the_op_and_shapes() {
    let a = Tensor::zeros(&[64, 784]);
    let b = Tensor::zeros(&[128, 784]);
    assert_eq!(a.try_matmul(&b).unwrap_err().to_string(), "matmul: [64, 784] × [128, 784]: inner dims 784 ≠ 128");
    let err = std::panic::catch_unwind(|| a.matmul(&b)).unwrap_err();
    assert!(err.downcast_ref::<String>().unwrap().contains("inner dims 784 ≠ 128"));
    assert!(a.try_add(&Tensor::zeros(&[3])).unwrap_err().to_string().contains("can't be broadcast"));
}

#[test]
fn indexing_matches_pytorch() {
    let x = Tensor::arange(0.0, 24.0, 1.0).reshape(&[2, 3, 4]);
    assert_eq!(x.i((.., 0)).shape(), &[2, 4]);
    assert_eq!(x.i((1, 2)).to_vec::<f32>(), vec![20., 21., 22., 23.]);
    assert_eq!(x.i((.., 1..3, 3)).to_vec::<f32>(), vec![7., 11., 19., 23.]);
    assert_eq!(x.i(1).shape(), &[3, 4]);
}

#[test]
fn clip_grad_norm_handles_shared_gradients() {
    // a + b gives both leaves the same gradient tensor.
    let a = Tensor::ones(&[4]).requires_grad_(true);
    let b = Tensor::ones(&[4]).requires_grad_(true);
    (&a + &b).sum().backward();
    let norm = nn::utils::clip_grad_norm_(&[a.clone(), b.clone()], 1.0);
    assert!((norm - 8f64.sqrt()).abs() < 1e-6);
    let after: f64 = [&a, &b].iter().map(|p| (p.grad().unwrap().square()).sum().item()).sum::<f64>().sqrt();
    assert!((after - 1.0).abs() < 1e-4, "norm after clipping {after}");
}
