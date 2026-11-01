use std::fmt;

/// Element type of a tensor, carried at runtime as in PyTorch.
///
/// `F32` is the default for data and parameters. `I64` holds class labels
/// and indices (what `F::cross_entropy` expects), and `F64` is used on the CPU
/// for gradient checks, where finite differences need the extra precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DType {
    #[default]
    F32,
    F64,
    I64,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 | DType::I64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    /// The name used in safetensors headers (and by `torch.float32` etc.).
    pub fn safetensors_name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::I64 => "I64",
        }
    }

    pub fn from_safetensors_name(s: &str) -> Option<DType> {
        match s {
            "F32" => Some(DType::F32),
            "F64" => Some(DType::F64),
            "I64" => Some(DType::I64),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I64 => "i64",
        };
        f.write_str(s)
    }
}

/// A Rust type that can be stored in a tensor.
pub trait Element: Copy + Send + Sync + PartialOrd + fmt::Debug + 'static {
    const DTYPE: DType;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn zero() -> Self;
    fn one() -> Self;
}

/// Floating-point elements, the ones every math kernel is written for.
pub trait Float:
    Element
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn erf(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, e: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn min(self, o: Self) -> Self;
    fn neg_infinity() -> Self;
    fn infinity() -> Self;
}

macro_rules! float_impl {
    ($t:ty, $dt:expr, $erf:path) => {
        impl Element for $t {
            const DTYPE: DType = $dt;
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn zero() -> Self {
                0.0
            }
            fn one() -> Self {
                1.0
            }
        }
        impl Float for $t {
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            fn erf(self) -> Self {
                $erf(self)
            }
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn powf(self, e: Self) -> Self {
                <$t>::powf(self, e)
            }
            fn max(self, o: Self) -> Self {
                if self >= o || self.is_nan() {
                    self
                } else {
                    o
                }
            }
            fn min(self, o: Self) -> Self {
                if self <= o || self.is_nan() {
                    self
                } else {
                    o
                }
            }
            fn neg_infinity() -> Self {
                <$t>::NEG_INFINITY
            }
            fn infinity() -> Self {
                <$t>::INFINITY
            }
        }
    };
}

// erf isn't in std; libm's port of the musl implementation is pure Rust and
// builds everywhere, including WASM.
fn erf32(x: f32) -> f32 {
    libm::erff(x)
}

fn erf64(x: f64) -> f64 {
    libm::erf(x)
}

float_impl!(f32, DType::F32, erf32);
float_impl!(f64, DType::F64, erf64);

impl Element for i64 {
    const DTYPE: DType = DType::I64;
    fn from_f64(v: f64) -> Self {
        v as i64
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.3, 0.328626759459127),
            (1.0, 0.842700792949715),
            (2.5, 0.999593047982555),
            (-1.0, -0.842700792949715),
            (5.0, 0.999999999998463),
        ] {
            assert!((erf64(x) - want).abs() < 1e-13, "erf({x}) = {} want {want}", erf64(x));
        }
    }
}
