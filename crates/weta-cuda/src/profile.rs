//! A tiny opt-in profiler. With `WETA_CUDA_PROFILE=1`, every backend call
//! waits for the GPU to finish and adds its wall time to a per-label total,
//! so [`report`] shows where a training step goes. Waiting serialises the
//! CPU and GPU, so runs are slower while profiling; leave it off otherwise.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::CudaStream;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("WETA_CUDA_PROFILE").is_ok_and(|v| v != "0"))
}

thread_local! {
    // Only the outermost call is timed (e.g. not the cast inside to_host).
    static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

static TOTALS: Mutex<Option<HashMap<String, (u64, Duration)>>> = Mutex::new(None);

/// Times one backend call until dropped.
pub(crate) struct Timer<'a> {
    label: Option<(String, Instant, &'a Arc<CudaStream>)>,
    counted: bool,
}

/// `label` is only built when profiling is on.
pub(crate) fn timer<'a>(stream: &'a Arc<CudaStream>, label: impl FnOnce() -> String) -> Timer<'a> {
    if !enabled() {
        return Timer { label: None, counted: false };
    }
    if DEPTH.with(|d| d.replace(d.get() + 1)) > 0 {
        return Timer { label: None, counted: true };
    }
    stream.synchronize().ok();
    Timer { label: Some((label(), Instant::now(), stream)), counted: true }
}

impl Drop for Timer<'_> {
    fn drop(&mut self) {
        if self.counted {
            DEPTH.with(|d| d.set(d.get() - 1));
        }
        if let Some((label, start, stream)) = self.label.take() {
            stream.synchronize().ok();
            let mut t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
            let e = t.get_or_insert_with(HashMap::new).entry(label).or_default();
            e.0 += 1;
            e.1 += start.elapsed();
        }
    }
}

/// A table of time per label, largest first, or `None` if profiling is off.
pub fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let mut rows: Vec<(&String, &(u64, Duration))> = t.as_ref()?.iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1 .1));
    let total: Duration = rows.iter().map(|r| r.1 .1).sum();
    let mut s = format!("{:<44} {:>9} {:>10} {:>6}\n", "CUDA backend call", "calls", "ms", "%");
    for (label, (n, d)) in rows {
        let ms = d.as_secs_f64() * 1e3;
        s += &format!("{label:<44} {n:>9} {ms:>10.1} {:>5.1}%\n", 100.0 * d.as_secs_f64() / total.as_secs_f64());
    }
    s += &format!("{:<44} {:>9} {:>10.1}\n", "total", "", total.as_secs_f64() * 1e3);
    Some(s)
}
