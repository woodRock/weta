//! PyTorch-style printing: `println!("{x}")` gives
//!
//! ```text
//! tensor([[1.0000, 2.0000],
//!         [3.0000, 4.0000]])
//! ```

use std::fmt;

use weta_core::{DType, Device, HostData};

use crate::Tensor;

/// Above this many elements, only the first and last few of each dim print.
const SUMMARIZE: usize = 1000;
const EDGE: usize = 3;

fn fmt_values(h: &HostData) -> Vec<String> {
    match h {
        HostData::I64(v) => v.iter().map(|x| x.to_string()).collect(),
        _ => {
            let v = h.to_f64_vec();
            let max = v.iter().fold(0.0f64, |m, x| if x.is_finite() { m.max(x.abs()) } else { m });
            let sci = max >= 1e8 || (max > 0.0 && max < 1e-4);
            v.iter().map(|x| if sci { format!("{x:.4e}") } else { format!("{x:.4}") }).collect()
        }
    }
}

fn write_dim(out: &mut String, vals: &[String], shape: &[usize], start: usize, depth: usize, width: usize, summarize: bool) {
    if shape.is_empty() {
        out.push_str(&format!("{:>width$}", vals[start]));
        return;
    }
    let n = shape[0];
    let inner: usize = shape[1..].iter().product();
    out.push('[');
    let indices: Vec<Option<usize>> = if summarize && n > 2 * EDGE {
        (0..EDGE).map(Some).chain([None]).chain((n - EDGE..n).map(Some)).collect()
    } else {
        (0..n).map(Some).collect()
    };
    for (k, idx) in indices.iter().enumerate() {
        if k > 0 {
            out.push(',');
            if shape.len() > 1 {
                out.push_str(&"\n".repeat(shape.len() - 1));
                out.push_str(&" ".repeat(depth + 8));
            } else {
                out.push(' ');
            }
        }
        match idx {
            Some(i) => write_dim(out, vals, &shape[1..], start + i * inner, depth + 1, width, summarize),
            None => out.push_str("..."),
        }
    }
    out.push(']');
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vals = fmt_values(&self.to_host());
        let width = vals.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut s = String::from("tensor(");
        write_dim(&mut s, &vals, self.shape(), 0, 0, width, self.numel() > SUMMARIZE);
        if self.device() != Device::Cpu {
            s.push_str(&format!(", device='{}'", self.device()));
        }
        if self.dtype() != DType::F32 {
            s.push_str(&format!(", dtype={}", self.dtype()));
        }
        s.push(')');
        f.write_str(&s)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor[{:?}, {}, {}]", self.shape(), self.dtype(), self.device())
    }
}
