//! Tiny argument parsing shared by the examples: `--key value` pairs.

#![allow(dead_code)]

use std::collections::HashMap;

pub struct Args(HashMap<String, String>);

impl Args {
    pub fn parse() -> Args {
        let mut m = HashMap::new();
        let mut it = std::env::args().skip(1);
        while let Some(k) = it.next() {
            if let Some(k) = k.strip_prefix("--") {
                m.insert(k.to_string(), it.next().unwrap_or_default());
            }
        }
        Args(m)
    }

    pub fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        self.0.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    pub fn string(&self, key: &str, default: &str) -> String {
        self.0.get(key).cloned().unwrap_or_else(|| default.to_string())
    }
}
