//! CUDA C++ source for the element-wise kernels, compiled at runtime with
//! NVRTC. Every kernel takes its inputs as (pointer, [`Strides`]) so
//! transposed, sliced and broadcast views need no copy, mirroring the CPU
//! backend. One thread computes one output element.

/// Element types and their CUDA names.
pub const FLOATS: [(&str, &str); 2] = [("f32", "float"), ("f64", "double")];
pub const ALL: [(&str, &str); 3] = [("f32", "float"), ("f64", "double"), ("i64", "long long")];

const PRELUDE: &str = r#"
struct Strides { int ndim; int contig; long long shape[8]; long long stride[8]; long long offset; };
struct ConvP { long long n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow; };

// Storage index of row-major element i of a strided view. Contiguous views
// skip the per-dimension division, which costs far more than the memory
// access it computes the address of.
__device__ __forceinline__ long long at(const Strides& s, long long i) {
    if (s.contig) return s.offset + i;
    long long off = s.offset;
    for (int d = s.ndim - 1; d >= 0; --d) {
        long long n = s.shape[d];
        off += (i % n) * s.stride[d];
        i /= n;
    }
    return off;
}

// The same counter-based generator as weta_core::splitmix64.
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long seed, unsigned long long i) {
    unsigned long long z = seed + (i + 1ULL) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ double unit(unsigned long long seed, unsigned long long i) {
    return (double)(splitmix64(seed, i) >> 11) * (1.0 / 9007199254740992.0);
}

#define TID long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;

template <typename T> __device__ T unary_op(int op, T v, T p1, T p2) {
    switch (op) {
        case 0: return -v;
        case 1: return exp(v);
        case 2: return log(v);
        case 3: return sqrt(v);
        case 4: return fabs(v);
        case 5: return (T)((v > (T)0) - (v < (T)0));
        case 6: return v > (T)0 ? v : (T)0;
        case 7: return v > (T)0 ? (T)1 : (T)0;
        case 8: return tanh(v);
        case 9: if (v >= (T)0) { return (T)1 / ((T)1 + exp(-v)); } else { T e = exp(v); return e / ((T)1 + e); }
        case 10: return erf(v);
        case 11: return sin(v);
        case 12: return cos(v);
        case 13: return (T)1 / v;
        case 14: return v * p1 + p2;
        case 15: return pow(v, p1);
        case 16: return v > p1 ? (T)1 : (T)0;
        case 17: return (v >= p1 || v != v) ? v : p1;
        case 18: return (v <= p1 || v != v) ? v : p1;
    }
    return v;
}

__device__ long long unary_int(int op, long long v, double p1, double p2) {
    switch (op) {
        case 0: return -v;
        case 4: return v < 0 ? -v : v;
        case 5: return (v > 0) - (v < 0);
        case 6: return v > 0 ? v : 0;
        case 7: return v > 0;
        case 14: return (long long)((double)v * p1 + p2);
        case 16: return (double)v > p1;
        case 17: return v > (long long)p1 ? v : (long long)p1;
        case 18: return v < (long long)p1 ? v : (long long)p1;
    }
    return v;
}

template <typename T> __device__ T binary_op(int op, T a, T b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return pow(a, b);
        case 5: return (a >= b || a != a) ? a : b;
        case 6: return (a <= b || a != a) ? a : b;
        case 7: return a == b;
        case 8: return a != b;
        case 9: return a < b;
        case 10: return a <= b;
        case 11: return a > b;
        case 12: return a >= b;
    }
    return a;
}

__device__ long long binary_int(int op, long long a, long long b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: { long long q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1; return q; }
        case 4: { long long r = 1; for (long long k = 0; k < b; ++k) r *= a; return r; }
        case 5: return a > b ? a : b;
        case 6: return a < b ? a : b;
        case 7: return a == b;
        case 8: return a != b;
        case 9: return a < b;
        case 10: return a <= b;
        case 11: return a > b;
        case 12: return a >= b;
    }
    return a;
}
"#;

/// Generates the full kernel source.
pub fn source() -> String {
    let mut s = String::from(PRELUDE);
    for (n, t) in FLOATS {
        s += &format!(
            r#"
extern "C" __global__ void unary_{n}({t}* out, const {t}* x, Strides xs, long long n, int op, double p1, double p2) {{
    TID out[i] = unary_op<{t}>(op, x[at(xs, i)], ({t})p1, ({t})p2);
}}
extern "C" __global__ void binary_{n}({t}* out, const {t}* a, Strides as, const {t}* b, Strides bs, long long n, int op) {{
    TID out[i] = binary_op<{t}>(op, a[at(as, i)], b[at(bs, i)]);
}}
// dist 0: uniform in [a, b); dist 1: normal with mean a, std b (Box-Muller).
extern "C" __global__ void random_{n}({t}* out, long long n, unsigned long long seed, int dist, double a, double b) {{
    TID
    double v;
    if (dist == 0) {{
        v = a + (b - a) * unit(seed, i);
    }} else {{
        double u1 = 1.0 - unit(seed, 2 * i), u2 = unit(seed, 2 * i + 1);
        v = a + b * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    }}
    out[i] = ({t})v;
}}
extern "C" __global__ void col2im_{n}({t}* out, const {t}* cols, ConvP p, long long n) {{
    TID
    long long x = i % p.w, t = i / p.w;
    long long y = t % p.h; t /= p.h;
    long long c = t % p.c, b = t / p.c;
    {t} acc = 0;
    for (long long ki = 0; ki < p.kh; ++ki) {{
        long long ty = y + p.ph - ki * p.dh;
        if (ty < 0 || ty % p.sh != 0 || ty / p.sh >= p.oh) continue;
        for (long long kj = 0; kj < p.kw; ++kj) {{
            long long tx = x + p.pw - kj * p.dw;
            if (tx < 0 || tx % p.sw != 0 || tx / p.sw >= p.ow) continue;
            acc += cols[((b * p.c + c) * p.kh * p.kw + ki * p.kw + kj) * p.oh * p.ow + (ty / p.sh) * p.ow + tx / p.sw];
        }}
    }}
    out[i] = acc;
}}
"#
        );
    }
    s += r#"
extern "C" __global__ void unary_i64(long long* out, const long long* x, Strides xs, long long n, int op, double p1, double p2) {
    TID out[i] = unary_int(op, x[at(xs, i)], p1, p2);
}
extern "C" __global__ void binary_i64(long long* out, const long long* a, Strides as, const long long* b, Strides bs, long long n, int op) {
    TID out[i] = binary_int(op, a[at(as, i)], b[at(bs, i)]);
}
"#;
    for (n, t) in ALL {
        s += &format!(
            r#"
// op: 0 sum, 1 max, 2 min. One thread per output element walks the axis.
extern "C" __global__ void reduce_{n}({t}* out, const {t}* x, Strides outer, long long len, long long stride, long long n, int op) {{
    TID
    long long base = at(outer, i);
    {t} acc = x[base];
    for (long long k = 1; k < len; ++k) {{
        {t} v = x[base + k * stride];
        if (op == 0) acc += v;
        else if (op == 1) acc = (v > acc || v != v) ? v : acc;
        else acc = (v < acc || v != v) ? v : acc;
    }}
    if (len == 0) acc = 0;
    out[i] = acc;
}}
// The same reduction with one block per output: each thread folds a strided
// share of the axis, then the block combines the partial results in shared
// memory, halving the number of active threads each step (a tree reduction).
// Used for long axes, where one thread per output would leave most of the
// GPU idle.
extern "C" __global__ void reduce_block_{n}({t}* out, const {t}* x, Strides outer, long long len, long long stride, int op) {{
    extern __shared__ unsigned char smem[];
    {t}* part = ({t}*)smem;
    long long base = at(outer, blockIdx.x);
    // Sums start from 0. Max and min start from element 0, which every
    // thread may include, since repeating it doesn't change the result.
    {t} acc = op == 0 ? ({t})0 : x[base];
    for (long long k = threadIdx.x; k < len; k += blockDim.x) {{
        {t} v = x[base + k * stride];
        if (op == 0) acc += v;
        else if (op == 1) acc = (v > acc || v != v) ? v : acc;
        else acc = (v < acc || v != v) ? v : acc;
    }}
    part[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int w = blockDim.x / 2; w > 0; w /= 2) {{
        if (threadIdx.x < w) {{
            {t} v = part[threadIdx.x + w];
            if (op == 0) part[threadIdx.x] += v;
            else if (op == 1) part[threadIdx.x] = (v > part[threadIdx.x] || v != v) ? v : part[threadIdx.x];
            else part[threadIdx.x] = (v < part[threadIdx.x] || v != v) ? v : part[threadIdx.x];
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) out[blockIdx.x] = part[0];
}}
// op: 0 argmax, 1 argmin (first occurrence).
extern "C" __global__ void argreduce_{n}(long long* out, const {t}* x, Strides outer, long long len, long long stride, long long n, int op) {{
    TID
    long long base = at(outer, i), best = 0;
    {t} bv = x[base];
    for (long long k = 1; k < len; ++k) {{
        {t} v = x[base + k * stride];
        if (op == 0 ? v > bv : v < bv) {{ bv = v; best = k; }}
    }}
    out[i] = best;
}}
extern "C" __global__ void copy_{n}({t}* dst, Strides ds, const {t}* src, Strides ss, long long n) {{
    TID dst[at(ds, i)] = src[at(ss, i)];
}}
extern "C" __global__ void im2col_{n}({t}* out, const {t}* x, Strides xs, ConvP p, long long n) {{
    TID
    long long ox = i % p.ow, t = i / p.ow;
    long long oy = t % p.oh; t /= p.oh;
    long long kj = t % p.kw; t /= p.kw;
    long long ki = t % p.kh; t /= p.kh;
    long long c = t % p.c, b = t / p.c;
    long long iy = oy * p.sh - p.ph + ki * p.dh, ix = ox * p.sw - p.pw + kj * p.dw;
    out[i] = (iy >= 0 && iy < p.h && ix >= 0 && ix < p.w)
        ? x[xs.offset + b * xs.stride[0] + c * xs.stride[1] + iy * xs.stride[2] + ix * xs.stride[3]] : ({t})0;
}}
"#
        );
        for (m, u) in ALL {
            s += &format!(
                "extern \"C\" __global__ void cast_{n}_{m}({u}* out, const {t}* x, Strides xs, long long n) {{ TID out[i] = ({u})x[at(xs, i)]; }}\n"
            );
        }
    }
    s
}

/// Every kernel name in [`source`], to load after compiling.
pub fn names() -> Vec<String> {
    let mut v = Vec::new();
    for (n, _) in FLOATS {
        v.push(format!("col2im_{n}"));
        v.push(format!("random_{n}"));
    }
    for (n, _) in ALL {
        for k in ["unary", "binary", "reduce", "reduce_block", "argreduce", "copy", "im2col"] {
            v.push(format!("{k}_{n}"));
        }
        for (m, _) in ALL {
            v.push(format!("cast_{n}_{m}"));
        }
    }
    v
}
