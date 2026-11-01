use crate::{DType, Device};
use std::fmt;

/// Everything that can go wrong in weta. Short-named ops panic with this
/// error's message; their `try_` versions return it.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Shapes that don't fit an op, e.g. `matmul: [64, 784] × [128, 784]: inner dims 784 ≠ 128`.
    Shape { op: &'static str, msg: String },
    /// Mismatched or unsupported dtypes.
    DType { op: &'static str, msg: String },
    /// Tensors on two different devices used in one op.
    Device { op: &'static str, a: Device, b: Device },
    /// A device that isn't compiled in or can't be opened.
    DeviceUnavailable(String),
    /// File and format errors (safetensors, datasets).
    Io(String),
    /// Anything else.
    Msg(String),
}

impl Error {
    pub fn shape(op: &'static str, msg: impl Into<String>) -> Self {
        Error::Shape { op, msg: msg.into() }
    }
    pub fn dtype(op: &'static str, a: DType, b: DType) -> Self {
        Error::DType { op, msg: format!("expected matching dtypes, got {a} and {b}") }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Shape { op, msg } => write!(f, "{op}: {msg}"),
            Error::DType { op, msg } => write!(f, "{op}: {msg}"),
            Error::Device { op, a, b } => {
                write!(f, "{op}: expected all tensors on the same device, found {a} and {b}")
            }
            Error::DeviceUnavailable(m) => write!(f, "device unavailable: {m}"),
            Error::Io(m) => write!(f, "io: {m}"),
            Error::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
