//! # weta-io
//!
//! State dicts, saved as [safetensors](https://github.com/huggingface/safetensors)
//! instead of pickle: safe to load (no code runs), memory-layout friendly,
//! and readable from PyTorch with `safetensors.torch.load_file`. Names and
//! layouts match PyTorch's, so a `Linear` weight is `[out, in]` here too.
//!
//! The format is small enough to implement in a page:
//!
//! ```text
//! [8 bytes: header length N, u64 little-endian]
//! [N bytes: JSON {"name": {"dtype": "F32", "shape": [..], "data_offsets": [begin, end]}, ...}]
//! [raw little-endian tensor bytes]
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use weta_core::{DType, Device, Error, HostData, Result};
use weta_tensor::Tensor;

/// An ordered map from names (`fc1.weight`) to tensors, plus string metadata.
#[derive(Clone, Default)]
pub struct StateDict {
    entries: Vec<(String, Tensor)>,
    pub metadata: BTreeMap<String, String>,
}

impl StateDict {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces an entry, keeping insertion order.
    pub fn insert(&mut self, name: impl Into<String>, t: Tensor) {
        let name = name.into();
        match self.entries.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 = t,
            None => self.entries.push((name, t)),
        }
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.entries.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    pub fn remove(&mut self, name: &str) -> Option<Tensor> {
        let i = self.entries.iter().position(|(n, _)| n == name)?;
        Some(self.entries.remove(i).1)
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.entries.iter().map(|(n, t)| (n.as_str(), t))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries whose name starts with `prefix.`, with the prefix removed.
    pub fn sub(&self, prefix: &str) -> StateDict {
        let p = format!("{prefix}.");
        let entries = self.entries.iter().filter_map(|(n, t)| n.strip_prefix(&p).map(|rest| (rest.to_string(), t.clone()))).collect();
        StateDict { entries, metadata: BTreeMap::new() }
    }

    /// Serialises to safetensors bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        if !self.metadata.is_empty() {
            let m = self.metadata.iter().map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone()))).collect();
            header.insert("__metadata__".into(), serde_json::Value::Object(m));
        }
        let mut data = Vec::new();
        for (name, t) in &self.entries {
            let begin = data.len();
            match t.to_host() {
                HostData::F32(v) => v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes())),
                HostData::F64(v) => v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes())),
                HostData::I64(v) => v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes())),
            }
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": t.dtype().safetensors_name(),
                    "shape": t.shape(),
                    "data_offsets": [begin, data.len()],
                }),
            );
        }
        let mut h = serde_json::to_vec(&serde_json::Value::Object(header)).expect("JSON");
        // Pad the header so tensor data starts 8-byte aligned.
        while !h.len().is_multiple_of(8) {
            h.push(b' ');
        }
        let mut out = Vec::with_capacity(8 + h.len() + data.len());
        out.extend_from_slice(&(h.len() as u64).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        out
    }

    /// Parses safetensors bytes into CPU tensors. F16/BF16 aren't supported.
    pub fn from_bytes(bytes: &[u8]) -> Result<StateDict> {
        let bad = |m: &str| Error::Io(format!("invalid safetensors data: {m}"));
        if bytes.len() < 8 {
            return Err(bad("too short"));
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header_end = 8usize.checked_add(n).filter(|&e| e <= bytes.len()).ok_or_else(|| bad("header length"))?;
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end]).map_err(|e| bad(&e.to_string()))?;
        let header = header.as_object().ok_or_else(|| bad("header is not an object"))?;
        let data = &bytes[header_end..];
        let mut sd = StateDict::new();
        let mut items: Vec<(&String, &serde_json::Value)> = header.iter().collect();
        // Keep the file's data order, which is the order tensors were saved in.
        items.sort_by_key(|(_, v)| v["data_offsets"][0].as_u64().unwrap_or(0));
        for (name, info) in items {
            if name == "__metadata__" {
                if let Some(m) = info.as_object() {
                    for (k, v) in m {
                        sd.metadata.insert(k.clone(), v.as_str().unwrap_or_default().to_string());
                    }
                }
                continue;
            }
            let dtype_s = info["dtype"].as_str().ok_or_else(|| bad("missing dtype"))?;
            let dtype = DType::from_safetensors_name(dtype_s).ok_or_else(|| Error::Io(format!("{name}: unsupported dtype {dtype_s}")))?;
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .ok_or_else(|| bad("missing shape"))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| bad("shape")))
                .collect::<Result<_>>()?;
            let offs = info["data_offsets"].as_array().ok_or_else(|| bad("missing data_offsets"))?;
            let (b, e) = (offs[0].as_u64().unwrap_or(0) as usize, offs[1].as_u64().unwrap_or(0) as usize);
            let numel: usize = shape.iter().product();
            if b > e || e > data.len() || e - b != numel * dtype.size_in_bytes() {
                return Err(Error::Io(format!("{name}: data_offsets [{b}, {e}] don't match shape {shape:?} of {dtype}")));
            }
            let raw = &data[b..e];
            let host = match dtype {
                DType::F32 => HostData::F32(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
                DType::F64 => HostData::F64(raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()),
                DType::I64 => HostData::I64(raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()),
            };
            sd.insert(name.clone(), Tensor::try_from_host(host, &shape, Device::Cpu)?);
        }
        Ok(sd)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_bytes()).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<StateDict> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        StateDict::from_bytes(&bytes)
    }
}

impl std::fmt::Debug for StateDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut m = f.debug_map();
        for (n, t) in &self.entries {
            m.entry(n, &format_args!("{:?} {}", t.shape(), t.dtype()));
        }
        m.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut sd = StateDict::new();
        sd.insert("fc.weight", Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        sd.insert("bn.num_batches_tracked", Tensor::from_vec(vec![7i64], &[]));
        sd.insert("x", Tensor::from_vec(vec![0.5f64], &[1]).broadcast_to(&[3]));
        sd.metadata.insert("format".into(), "pt".into());
        let bytes = sd.to_bytes();
        assert_eq!(u64::from_le_bytes(bytes[..8].try_into().unwrap()) % 8, 0);
        let back = StateDict::from_bytes(&bytes).unwrap();
        assert_eq!(back.keys().collect::<Vec<_>>(), vec!["fc.weight", "bn.num_batches_tracked", "x"]);
        assert_eq!(back.get("fc.weight").unwrap().to_vec::<f32>(), vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(back.get("bn.num_batches_tracked").unwrap().shape(), &[] as &[usize]);
        assert_eq!(back.get("x").unwrap().to_vec::<f64>(), vec![0.5; 3]);
        assert_eq!(back.metadata["format"], "pt");
        assert_eq!(back.sub("fc").keys().collect::<Vec<_>>(), vec!["weight"]);
    }

    #[test]
    fn rejects_garbage() {
        assert!(StateDict::from_bytes(&[1, 2, 3]).is_err());
        assert!(StateDict::from_bytes(&[255; 16]).is_err());
    }
}
