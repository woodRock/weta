//! # weta-infer
//!
//! Inference without autograd. This crate depends only on `weta-tensor` and
//! `weta-io`, so a deployed binary contains no graph, optimisers or data
//! loaders.
//!
//! A model exported with `weta::nn::utils::export` is one safetensors file
//! holding the weights and, in its metadata, the architecture (one layer per
//! line, e.g. `conv2d stride=1 padding=1`). Embed it in the binary and run:
//!
//! ```ignore
//! static MODEL: &[u8] = include_bytes!("mnist_cnn.safetensors");
//! let model = weta_infer::Model::from_bytes(MODEL)?;
//! let logits = model.forward(&image); // [1, 1, 28, 28] → [1, 10]
//! ```

use std::collections::HashMap;
use std::path::Path;

use weta_core::{DType, Device, Error, Result};
use weta_io::StateDict;
use weta_tensor::Tensor;

/// One forward-only layer.
pub enum Layer {
    Linear {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Conv2d {
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    },
    MaxPool2d {
        kernel: usize,
        stride: usize,
    },
    AvgPool2d {
        kernel: usize,
        stride: usize,
        padding: usize,
    },
    /// Batch norm in eval mode, folded to `y = x · scale + shift` per channel.
    BatchNorm {
        scale: Tensor,
        shift: Tensor,
    },
    LayerNorm {
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        dims: usize,
        eps: f64,
    },
    Flatten {
        start: isize,
        end: isize,
    },
    Relu,
    Tanh,
    Sigmoid,
    Gelu,
    Silu,
    LeakyRelu(f64),
    Softmax(isize),
    Identity,
}

impl Layer {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Layer::Linear { weight, bias } => {
                let y = x.matmul(&weight.t());
                match bias {
                    Some(b) => y.add(b),
                    None => y,
                }
            }
            Layer::Conv2d { weight, bias, stride, padding, dilation } => {
                if *dilation == 1 {
                    x.conv2d(weight, bias.as_ref(), *stride, *padding)
                } else {
                    let (n, h, w) = (x.size(0), x.size(2), x.size(3));
                    let (o, kh, kw) = (weight.size(0), weight.size(2), weight.size(3));
                    let p = weta_core::Im2ColParams {
                        kernel: (kh, kw),
                        stride: (*stride, *stride),
                        padding: (*padding, *padding),
                        dilation: (*dilation, *dilation),
                    };
                    let (oh, ow) = p.out_size(h, w);
                    let mut y = weight.reshape(&[o as isize, -1]).matmul(&x.im2col(p));
                    if let Some(b) = bias {
                        y = y.add(&b.reshape(&[o as isize, 1]));
                    }
                    y.reshape_to(&[n, o, oh, ow])
                }
            }
            Layer::MaxPool2d { kernel, stride } => x.max_pool2d(*kernel, *stride),
            Layer::AvgPool2d { kernel, stride, padding } => {
                let (n, c, h, w) = (x.size(0), x.size(1), x.size(2), x.size(3));
                let p = weta_core::Im2ColParams {
                    kernel: (*kernel, *kernel),
                    stride: (*stride, *stride),
                    padding: (*padding, *padding),
                    dilation: (1, 1),
                };
                let (oh, ow) = p.out_size(h, w);
                x.reshape_to(&[n * c, 1, h, w]).im2col(p).mean_dims(&[1], false).reshape_to(&[n, c, oh, ow])
            }
            Layer::BatchNorm { scale, shift } => {
                let mut s = vec![1; x.ndim()];
                s[1] = scale.numel();
                x.mul(&scale.reshape_to(&s)).add(&shift.reshape_to(&s))
            }
            Layer::LayerNorm { weight, bias, dims, eps } => {
                let d: Vec<isize> = (x.ndim() - dims..x.ndim()).map(|d| d as isize).collect();
                let c = x.sub(&x.mean_dims(&d, true));
                let var = c.mul(&c).mean_dims(&d, true);
                let mut y = c.div(&var.add_scalar(*eps).sqrt());
                if let Some(w) = weight {
                    y = y.mul(w);
                }
                if let Some(b) = bias {
                    y = y.add(b);
                }
                y
            }
            Layer::Flatten { start, end } => x.flatten(*start, *end),
            Layer::Relu => x.relu(),
            Layer::Tanh => x.tanh(),
            Layer::Sigmoid => x.sigmoid(),
            Layer::Gelu => x.mul(&x.mul_scalar(std::f64::consts::FRAC_1_SQRT_2).erf().add_scalar(1.0)).mul_scalar(0.5),
            Layer::Silu => x.mul(&x.sigmoid()),
            Layer::LeakyRelu(a) => x.relu().sub(&x.neg().relu().mul_scalar(*a)),
            Layer::Softmax(d) => x.softmax(*d),
            Layer::Identity => x.clone(),
        }
    }

    fn tensors_mut(&mut self) -> Vec<&mut Tensor> {
        match self {
            Layer::Linear { weight, bias } | Layer::Conv2d { weight, bias, .. } => {
                let mut v = vec![weight];
                v.extend(bias.as_mut());
                v
            }
            Layer::BatchNorm { scale, shift } => vec![scale, shift],
            Layer::LayerNorm { weight, bias, .. } => weight.iter_mut().chain(bias.iter_mut()).collect(),
            _ => Vec::new(),
        }
    }
}

/// A chain of layers, built from an architecture description and weights.
pub struct Model {
    pub layers: Vec<Layer>,
}

/// `key=value` options after a layer name.
fn options(parts: &[&str]) -> Result<HashMap<String, String>> {
    parts
        .iter()
        .map(|p| {
            p.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .ok_or_else(|| Error::Msg(format!("weta-infer: expected key=value, got {p:?}")))
        })
        .collect()
}

fn opt<T: std::str::FromStr>(o: &HashMap<String, String>, key: &str, default: T) -> Result<T> {
    match o.get(key) {
        None => Ok(default),
        Some(v) => v.parse().map_err(|_| Error::Msg(format!("weta-infer: bad value {v:?} for {key}"))),
    }
}

impl Model {
    /// Builds a model from an architecture (one layer per line; layer `i`'s
    /// weights named `i.weight`, `i.bias`, ...) and a state dict.
    pub fn from_state_dict(arch: &str, sd: &StateDict) -> Result<Model> {
        let mut layers = Vec::new();
        for (i, line) in arch.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let o = options(&parts[1..])?;
            let get = |name: &str| {
                sd.get(&format!("{i}.{name}"))
                    .cloned()
                    .ok_or_else(|| Error::Msg(format!("weta-infer: layer {i} ({}) needs {i}.{name}", parts[0])))
            };
            let maybe = |name: &str| sd.get(&format!("{i}.{name}")).cloned();
            layers.push(match parts[0] {
                "linear" => Layer::Linear { weight: get("weight")?, bias: maybe("bias") },
                "conv2d" => Layer::Conv2d {
                    weight: get("weight")?,
                    bias: maybe("bias"),
                    stride: opt(&o, "stride", 1)?,
                    padding: opt(&o, "padding", 0)?,
                    dilation: opt(&o, "dilation", 1)?,
                },
                "maxpool2d" => {
                    let k = opt(&o, "kernel", 2)?;
                    Layer::MaxPool2d { kernel: k, stride: opt(&o, "stride", k)? }
                }
                "avgpool2d" => {
                    let k = opt(&o, "kernel", 2)?;
                    Layer::AvgPool2d { kernel: k, stride: opt(&o, "stride", k)?, padding: opt(&o, "padding", 0)? }
                }
                "batchnorm" => {
                    // Fold the running statistics into one scale and shift:
                    // γ (x − μ) / √(σ² + ε) + β = x·s + (β − μ·s), s = γ / √(σ² + ε).
                    let eps: f64 = opt(&o, "eps", 1e-5)?;
                    let (mean, var) = (get("running_mean")?, get("running_var")?);
                    let gamma = maybe("weight").unwrap_or_else(|| mean.ones_like());
                    let beta = maybe("bias").unwrap_or_else(|| mean.zeros_like());
                    let scale = gamma.div(&var.add_scalar(eps).sqrt());
                    let shift = beta.sub(&mean.mul(&scale));
                    Layer::BatchNorm { scale, shift }
                }
                "layernorm" => {
                    Layer::LayerNorm { weight: maybe("weight"), bias: maybe("bias"), dims: opt(&o, "dims", 1)?, eps: opt(&o, "eps", 1e-5)? }
                }
                "flatten" => Layer::Flatten { start: opt(&o, "start", 1)?, end: opt(&o, "end", -1)? },
                "relu" => Layer::Relu,
                "tanh" => Layer::Tanh,
                "sigmoid" => Layer::Sigmoid,
                "gelu" => Layer::Gelu,
                "silu" => Layer::Silu,
                "leakyrelu" => Layer::LeakyRelu(opt(&o, "slope", 0.01)?),
                "softmax" => Layer::Softmax(opt(&o, "dim", 1)?),
                "identity" | "dropout" => Layer::Identity,
                other => return Err(Error::Msg(format!("weta-infer: unknown layer {other:?}"))),
            });
        }
        Ok(Model { layers })
    }

    /// Loads an exported model from safetensors bytes (e.g. `include_bytes!`).
    pub fn from_bytes(bytes: &[u8]) -> Result<Model> {
        let sd = StateDict::from_bytes(bytes)?;
        let arch = sd
            .metadata
            .get("weta.arch")
            .ok_or_else(|| Error::Msg("weta-infer: no weta.arch in the file's metadata; was it saved with nn::utils::export?".into()))?
            .clone();
        Model::from_state_dict(&arch, &sd)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Model> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| Error::Io(format!("{}: {e}", path.as_ref().display())))?;
        Model::from_bytes(&bytes)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.layers.iter().fold(x.clone(), |h, l| l.forward(&h))
    }

    /// Moves the weights to `device`.
    pub fn to(mut self, device: Device) -> Model {
        for l in &mut self.layers {
            for t in l.tensors_mut() {
                *t = t.to(device);
            }
        }
        self
    }

    /// Class predictions for a batch of inputs (argmax of the outputs).
    pub fn predict(&self, x: &Tensor) -> Vec<i64> {
        self.forward(x).argmax(-1, false).to_vec::<i64>()
    }
}

/// A float input tensor from host data, for callers that don't otherwise
/// need `weta-tensor`.
pub fn input(data: Vec<f32>, shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape).to_dtype(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_a_small_cnn() {
        let mut sd = StateDict::new();
        sd.insert("0.weight", Tensor::ones(&[2, 1, 3, 3], DType::F32, Device::Cpu));
        sd.insert("0.bias", Tensor::zeros(&[2], DType::F32, Device::Cpu));
        sd.insert("1.running_mean", Tensor::zeros(&[2], DType::F32, Device::Cpu));
        sd.insert("1.running_var", Tensor::ones(&[2], DType::F32, Device::Cpu));
        sd.insert("5.weight", Tensor::ones(&[3, 2 * 2 * 2], DType::F32, Device::Cpu));
        let arch = "conv2d stride=1 padding=1\nbatchnorm eps=0\nrelu\nmaxpool2d kernel=2\nflatten\nlinear";
        let m = Model::from_state_dict(arch, &sd).unwrap();
        let y = m.forward(&Tensor::ones(&[1, 1, 4, 4], DType::F32, Device::Cpu));
        assert_eq!(y.shape(), &[1, 3]);
        // Each 2×2 pooled max is 9 (an interior 3×3 sum of ones); 8 of them.
        assert_eq!(y.to_vec::<f32>(), vec![72.0; 3]);
    }
}
