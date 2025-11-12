use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use libc;
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use smallvec::SmallVec;
use std::fs::File;
use std::mem;

use ahash::AHashMap; // fast maps for hot paths
use rayon::prelude::*;
use std::collections::HashMap as StdHashMap; // std map for Snapshot.compat // parallel within a root

use std::sync::atomic::{AtomicU64, Ordering};

use crate::matching::types::{Bucket, Snapshot};

// expose the compat helper module you added at src/enumeration/compat.rs
pub mod compat;
use compat::{build_compat_map, debug_summary as compat_debug_summary};

// -------------------------------------------------------------------------------------
// Tunables & light-weight typedefs
// -------------------------------------------------------------------------------------

/// Pending-batch size that triggers an early flush (keeps peaks down).
/// Now runtime-tunable via `ENUM_PEND_FLUSH`; default 32_768.
fn pend_flush_codes() -> usize {
    std::env::var("ENUM_PEND_FLUSH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32_768)
}

/// Limit how many pre_jbt from the (0,0) root we enumerate.
/// Set via `ENUM_FIRST_LIMIT` (e.g., "500"); unset/empty -> no limit.
fn first_bucket_limit() -> Option<usize> {
    match std::env::var("ENUM_FIRST_LIMIT") {
        Ok(s) if !s.is_empty() => s.parse().ok(),
        _ => None,
    }
}

/// Enumeration-time weight type (integer counts). Cast to f64 at snapshot build.
type Weight = u32;

/// Count how many times we had to clamp Weight (u32) during reductions.
static SATURATED_WEIGHTS: AtomicU64 = AtomicU64::new(0);

// -------------------------------------------------------------------------------------
// Memory tracking helpers (HPC safety)
// -------------------------------------------------------------------------------------

const KB: u64 = 1024;
const MB: u64 = KB * 1024;
const GB: u64 = MB * 1024;

fn parse_budget_var(var: &str, multiplier: u64) -> Option<u64> {
    let raw = std::env::var(var).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    raw.trim()
        .parse::<u64>()
        .ok()
        .map(|v| v.saturating_mul(multiplier))
}

fn memory_budget_bytes() -> Option<u64> {
    parse_budget_var("ENUM_MAX_RSS_BYTES", 1)
        .or_else(|| parse_budget_var("ENUM_MAX_RSS_MB", MB))
        .or_else(|| parse_budget_var("ENUM_MAX_RSS_GB", GB))
}

fn current_rss_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/self/statm").ok()?;
    let mut parts = contents.split_whitespace();
    let _total = parts.next()?;
    let resident_pages: u64 = parts.next()?.parse().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }
    Some(resident_pages.saturating_mul(page_size as u64))
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / GB as f64
}

fn report_memory_after_vacate(root_idx: usize, budget: Option<u64>) -> Result<()> {
    if let Some(rss) = current_rss_bytes() {
        match budget {
            Some(limit) => {
                eprintln!(
                    "[mem] root={} rss={:.2} GiB (limit {:.2} GiB)",
                    root_idx,
                    bytes_to_gib(rss),
                    bytes_to_gib(limit)
                );
                if rss > limit {
                    bail!(
                        "RSS {:.2} GiB exceeded limit {:.2} GiB (set via ENUM_MAX_RSS_*)",
                        bytes_to_gib(rss),
                        bytes_to_gib(limit)
                    );
                }
            }
            None => {
                eprintln!("[mem] root={} rss={:.2} GiB", root_idx, bytes_to_gib(rss));
            }
        }
    }
    Ok(())
}

// --- NPZ compat loader (no `zip` crate needed) ---
fn try_load_compat_npz(
    path: &str,
) -> anyhow::Result<Option<std::collections::HashMap<i32, (Vec<i32>, Vec<i32>)>>> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut npz = match NpzReader::new(f) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    // If this key is missing, assume compat not provided
    let pops: Array1<i32> = match npz.by_name("meta_compat_pops.npy") {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };

    let mut compat: std::collections::HashMap<i32, (Vec<i32>, Vec<i32>)> =
        std::collections::HashMap::new();
    for &p in pops.iter() {
        let key1 = format!("compat_p{}_key1.npy", p);
        let key2 = format!("compat_p{}_key2.npy", p);

        let arr1: Array1<i32> = npz
            .by_name(&key1)
            .unwrap_or_else(|_| Array1::from_vec(vec![]));
        let arr2: Array1<i32> = npz
            .by_name(&key2)
            .unwrap_or_else(|_| Array1::from_vec(vec![]));

        compat.insert(p, (arr1.to_vec(), arr2.to_vec()));
    }
    Ok(Some(compat))
}

/// Ensure the solver will never panic:
/// - Have entries for every pop in 1..N-1
/// - Mirror-fill q=N-p if only p was provided
fn cover_and_symmetrize_compat(
    mut c: std::collections::HashMap<i32, (Vec<i32>, Vec<i32>)>,
    n_total: i32,
) -> std::collections::HashMap<i32, (Vec<i32>, Vec<i32>)> {
    // Mirror-fill: if p exists, create q=N-p with swapped lists if missing
    let base: Vec<i32> = c.keys().copied().collect();
    for p in base {
        let q = n_total - p;
        if let Some((k1, k2)) = c.get(&p).cloned() {
            c.entry(q).or_insert_with(|| (k2, k1));
        }
    }
    // Ensure coverage 1..N-1
    for p in 1..n_total {
        c.entry(p).or_insert_with(|| (Vec::new(), Vec::new()));
    }
    c
}

fn debug_pop_quickline(compat: &std::collections::HashMap<i32, (Vec<i32>, Vec<i32>)>, p: i32) {
    if let Some((k1, k2)) = compat.get(&p) {
        eprintln!("[compat] p={} -> (#k1={}, #k2={})", p, k1.len(), k2.len());
    } else {
        eprintln!("[compat] p={} MISSING", p);
    }
}

// -------------------------------------------------------------------------------------
// Packed row code (u128) utilities
// -------------------------------------------------------------------------------------

#[inline(always)]
fn bitwidth(m: usize) -> u32 {
    let m1 = m.saturating_sub(1) as u32;
    1u32.max(m1.next_power_of_two().trailing_zeros())
}

#[inline(always)]
fn code_len(code: u128) -> u32 {
    (code & 0xF) as u32
}

#[inline(always)]
fn code_len_u128(code: u128) -> usize {
    (code & 0xF) as usize
}

#[inline(always)]
fn code_get(code: u128, i: u32, b: u32) -> u32 {
    let shift = 4 + i * b;
    if shift < 64 {
        let rem = 64 - shift;
        if b <= rem {
            ((code as u64 >> shift) & ((1u64 << b) - 1)) as u32
        } else {
            let low = (code as u64 >> shift) & ((1u64 << rem) - 1);
            let hi = (code >> 64) as u64 & ((1u64 << (b - rem)) - 1);
            ((hi as u128) << rem | (low as u128)) as u32
        }
    } else {
        let s = shift - 64;
        (((code >> 64) as u64 >> s) & ((1u64 << b) - 1)) as u32
    }
}

#[inline(always)]
fn code_set(code: &mut u128, i: u32, b: u32, val: u32) {
    let shift = 4 + i * b;
    let v = (val as u128) & ((1u128 << b) - 1);
    if shift < 64 {
        let rem = 64 - shift;
        if b <= rem {
            let mask = !(((1u128 << b) - 1) << shift);
            *code = (*code & mask) | (v << shift);
        } else {
            // split across 64-bit boundary
            let low_bits = rem;
            let low_mask = ((1u128 << low_bits) - 1) << shift;
            let hi_bits = b - low_bits;
            let hi_mask = ((1u128 << hi_bits) - 1) << 64;

            let low_part = (v & ((1u128 << low_bits) - 1)) << shift;
            let hi_part = (v >> low_bits) << 64;

            *code = (*code & !low_mask) | low_part;
            *code = (*code & !hi_mask) | hi_part;
        }
    } else {
        let s = shift - 64;
        let mask = !(((1u128 << b) - 1) << (64 + s));
        *code = (*code & mask) | (v << (64 + s));
    }
}

#[inline(always)]
fn code_with_len(mut code: u128, k: u32) -> u128 {
    code = (code & !0xFu128) | (k as u128 & 0xF);
    code
}

/// Insert j into sorted set inside `code`. Returns (new_code, inserted).
#[inline(always)]
fn code_insert(code: u128, j: u32, b: u32) -> (u128, bool) {
    let mut k = code_len(code);
    let mut lo = 0i32;
    let mut hi = k as i32;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let v = code_get(code, mid as u32, b);
        if v < j {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo < k as i32 && code_get(code, lo as u32, b) == j {
        return (code, false);
    }
    if k >= 10 {
        return (code, false);
    }
    let mut out = code;
    let mut idx = k;
    while idx > lo as u32 {
        let prev = code_get(out, idx - 1, b);
        code_set(&mut out, idx, b, prev);
        idx -= 1;
    }
    code_set(&mut out, lo as u32, b, j);
    k += 1;
    out = code_with_len(out, k);
    (out, true)
}

/// Iterate j's in code (ascending).
#[inline(always)]
fn code_iter<'a>(code: u128, b: u32) -> impl Iterator<Item = u32> + 'a {
    let k = code_len(code);
    (0..k).map(move |i| code_get(code, i, b))
}

// -------------------------------------------------------------------------------------
// Bitboard helpers (N<=10, left half <= 50 bits)
// -------------------------------------------------------------------------------------

#[inline]
fn left_half_mask(n: u32) -> u64 {
    if n == 0 {
        0
    } else {
        ((1u128 << (n as u128 * (n / 2) as u128)) - 1) as u64
    }
}
#[inline]
fn col_mask(n: u32, x: u32) -> u64 {
    (((1u64 << n) - 1) as u64) << (x * n)
}
#[inline]
fn edge_masks(n: u32) -> (u64, u64) {
    let mut top = 0u64;
    let mut bot = 0u64;
    for x in 0..n {
        top |= 1u64 << (x * n + (n - 1));
        bot |= 1u64 << (x * n + 0);
    }
    (top, bot)
}

#[inline]
fn flood_fill(seed: u64, domain: u64, n: u32) -> u64 {
    if seed == 0 {
        return 0;
    }
    let (top_mask, bot_mask) = edge_masks(n);
    let mut comp = 0u64;
    let mut frontier = seed & domain;
    while frontier != 0 {
        comp |= frontier;
        let up = (frontier & !top_mask) << 1;
        let down = (frontier & !bot_mask) >> 1;
        let left = frontier >> n;
        let right = frontier << n;
        frontier = (up | down | left | right) & domain & !comp;
    }
    comp
}

#[inline]
fn detect_evil_pmask(partial_mask: u64, n: u32) -> bool {
    let half = left_half_mask(n);
    let escape = col_mask(n, n / 2 - 1);
    let mut complement = partial_mask ^ half;
    while complement != 0 {
        let seed = complement & complement.wrapping_neg();
        let comp = flood_fill(seed, complement, n);
        if (comp & escape) != 0 {
            complement ^= comp;
            continue;
        }
        if comp.count_ones() % n != 0 {
            return true;
        }
        complement ^= comp;
    }
    false
}

#[inline]
fn find_root(partial_mask: u64, n: u32) -> Option<(u32, u32)> {
    let left = left_half_mask(n);
    let complement = partial_mask ^ left;
    if complement == 0 {
        return None;
    }
    let lsb = complement & complement.wrapping_neg();
    let bit_pos = lsb.trailing_zeros();
    let (x, y) = (bit_pos / n, bit_pos % n);
    Some((x, y))
}

// -------------------------------------------------------------------------------------
// Input CSR for pre_jbt
// -------------------------------------------------------------------------------------

pub struct PreCsr {
    pub masks: Vec<u64>,     // len = nnz
    pub pops: Vec<u8>,       // len = nnz
    pub jidx: Vec<u32>,      // len = nnz
    pub offsets: Vec<usize>, // len = n_roots + 1
    pub n_roots: usize,
}

pub struct Inputs {
    pub n: u32,
    pub m: usize,
    pub pre: PreCsr,
    pub jbt_ref_pop: Vec<i32>,        // len = M
    pub jbt_ref_comps: Vec<[u16; 3]>, // len = M (or empty if not provided)
}

/// Load NPZ with:
/// - N, M
/// - pre_masks[u64], pre_pops[u8], pre_jidx[u32], pre_offsets[i64]
/// - jbt_ref_pop[i32], jbt_ref_comps[u16] (M x 3)
pub fn load_inputs_npz(path: &str) -> Result<Inputs> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let mut npz = NpzReader::new(f)?;
    let n_arr: Array1<i32> = npz.by_name("N.npy")?;
    let m_arr: Array1<i32> = npz.by_name("M.npy")?;

    let n = n_arr[0] as u32;
    let m = m_arr[0] as usize;

    let masks: Array1<u64> = npz.by_name("pre_masks.npy")?;
    let pops: Array1<u8> = npz.by_name("pre_pops.npy")?;
    let jidx: Array1<u32> = npz.by_name("pre_jidx.npy")?;
    let offs: Array1<i64> = npz.by_name("pre_offsets.npy")?;
    let jpop: Array1<i32> = npz.by_name("jbt_ref_pop.npy")?;

    // Optional comps — if missing, we proceed with empty compat (keys still filled)
    let jbt_ref_comps: Vec<[u16; 3]> = {
        let a2: Array2<u16> = match npz.by_name("jbt_ref_comps.npy") {
            Ok(arr) => arr,
            Err(_) => Array2::<u16>::zeros((0, 3)),
        };
        if a2.is_empty() {
            Vec::new()
        } else {
            let shape = a2.shape();
            if shape.len() != 2 || shape[1] != 3 {
                bail!(
                    "jbt_ref_comps.npy has wrong shape: {:?} (expected M x 3)",
                    shape
                );
            }
            a2.into_raw_vec()
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect()
        }
    };

    // --- sanity checks to catch stale/bad NPZs early ---
    let nnz = masks.len();
    if pops.len() != nnz || jidx.len() != nnz {
        bail!(
            "pre_* arrays have mismatched lengths: masks={}, pops={}, jidx={}",
            nnz,
            pops.len(),
            jidx.len()
        );
    }
    let off_last = *offs.last().context("pre_offsets is empty")? as usize;
    if off_last != nnz {
        bail!("pre_offsets last={} does not equal nnz={}", off_last, nnz);
    }
    if jpop.len() as usize != m {
        bail!("jbt_ref_pop has len {}, expected M={}", jpop.len(), m);
    }

    let offsets: Vec<usize> = offs.iter().map(|&x| x as usize).collect();
    let n_roots = offsets
        .len()
        .checked_sub(1)
        .context("offsets must be >= 1")?;
    Ok(Inputs {
        n,
        m,
        pre: PreCsr {
            masks: masks.to_vec(),
            pops: pops.to_vec(),
            jidx: jidx.to_vec(),
            offsets,
            n_roots,
        },
        jbt_ref_pop: jpop.to_vec(),
        jbt_ref_comps,
    })
}

// -------------------------------------------------------------------------------------
// Frontier + Out buckets
// -------------------------------------------------------------------------------------

#[derive(Default)]
struct AOBucket {
    // committed
    codes: Vec<u128>,
    weights: Vec<Weight>,
    // pending
    pend_codes: Vec<u128>,
    pend_w: Vec<Weight>,
}
impl AOBucket {
    fn append_batch(&mut self, codes: Vec<u128>, w: Vec<Weight>) {
        if codes.is_empty() {
            return;
        }
        self.pend_codes.extend(codes);
        self.pend_w.extend(w);
        if self.pend_codes.len() >= pend_flush_codes() {
            self.flush();
        }
    }
    fn flush(&mut self) {
        if self.pend_codes.is_empty() {
            return;
        }
        // concat committed + pending, then sort & reduce
        let mut all_codes = Vec::with_capacity(self.codes.len() + self.pend_codes.len());
        let mut all_w = Vec::with_capacity(self.weights.len() + self.pend_w.len());
        all_codes.extend_from_slice(&self.codes);
        all_codes.extend_from_slice(&self.pend_codes);
        all_w.extend_from_slice(&self.weights);
        all_w.extend_from_slice(&self.pend_w);

        let mut idx: Vec<usize> = (0..all_codes.len()).collect();
        idx.sort_unstable_by_key(|&i| all_codes[i]);

        let mut new_codes: Vec<u128> = Vec::with_capacity(all_codes.len());
        let mut new_w: Vec<Weight> = Vec::with_capacity(all_w.len());
        let mut i = 0usize;
        while i < idx.len() {
            let c = all_codes[idx[i]];
            let mut sum: u64 = all_w[idx[i]] as u64;
            i += 1;
            while i < idx.len() && all_codes[idx[i]] == c {
                sum = sum.saturating_add(all_w[idx[i]] as u64);
                i += 1;
            }
            new_codes.push(c);
            let packed = if sum > Weight::MAX as u64 {
                SATURATED_WEIGHTS.fetch_add(1, Ordering::Relaxed);
                Weight::MAX
            } else {
                sum as Weight
            };
            new_w.push(packed);
        }
        self.codes = new_codes;
        self.weights = new_w;
        self.pend_codes.clear();
        self.pend_w.clear();
    }
}

#[derive(Default)]
struct RootFrontier {
    masks: Vec<u64>,
    buckets: Vec<AOBucket>,
    index: AHashMap<u64, usize>,
}
impl RootFrontier {
    fn get_bucket_mut(&mut self, mask: u64) -> &mut AOBucket {
        if let Some(&pos) = self.index.get(&mask) {
            return &mut self.buckets[pos];
        }
        let pos = self.buckets.len();
        self.index.insert(mask, pos);
        self.masks.push(mask);
        self.buckets.push(AOBucket::default());
        &mut self.buckets[pos]
    }
    fn flush(&mut self) {
        for b in &mut self.buckets {
            b.flush();
        }
    }

    // (Removed the unused `clear` method to avoid a warning)
}

#[derive(Default)]
struct OutBuckets {
    by_key: AHashMap<u64, AOBucket>, // key = packed pop multiset; low nibble = k (fits u64 for N<=10)
}
impl OutBuckets {
    fn append_completed(&mut self, key: u64, codes: Vec<u128>, w: Vec<Weight>) {
        let b = self.by_key.entry(key).or_default();
        b.append_batch(codes, w);
    }
    fn flush_all(&mut self) {
        for b in self.by_key.values_mut() {
            b.flush();
        }
    }
}

fn pack_pop_key(mut pops: SmallVec<[u8; 10]>) -> u64 {
    pops.sort_unstable();
    let k = pops.len() as u64;
    let mut out = k & 0xF;
    let mut shift = 4u32;
    for p in pops {
        out |= ((p as u64) & 0xF) << shift;
        shift += 4;
    }
    out
}

fn code_pop_key(code: u128, b: u32, j_pop: &[i32]) -> u64 {
    let mut pops: SmallVec<[u8; 10]> = SmallVec::new();
    for j in code_iter(code, b) {
        pops.push(j_pop[j as usize] as u8);
    }
    pack_pop_key(pops)
}

// -------------------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------------------

pub fn enumerate_to_snapshot_from_npz(
    path_npz: &str,
) -> anyhow::Result<crate::matching::types::Snapshot> {
    let Inputs {
        n,
        m,
        pre,
        jbt_ref_pop,
        jbt_ref_comps,
    } = load_inputs_npz(path_npz)?;
    let mut snap = enumerate_to_snapshot(n, m, pre, &jbt_ref_pop)?;

    // Prefer Python-provided compat (authoritative); if not present, fall back to local build.
    if let Some(compat_npz) = try_load_compat_npz(path_npz)? {
        let compat_full = cover_and_symmetrize_compat(compat_npz, snap.n_total);
        snap.compat = compat_full;
        eprintln!("[compat] loaded from NPZ and symmetrized.");
    } else {
        // Fallback: local builder from comps (still creates all 1..N-1 keys).
        eprintln!("[compat] NPZ compat not found; building locally from comps.");
        snap.compat = build_compat_map(&snap.jbt_ref_pop, &jbt_ref_comps, snap.n_total);
    }

    // Quick sanity for p=4 (adjust p as you like)
    debug_pop_quickline(&snap.compat, 4);

    // Optional full summary (avoids “function never used” warning in compat.rs)
    if std::env::var("ENUM_COMPAT_DEBUG").ok().as_deref() == Some("1") {
        compat_debug_summary(&snap.compat, &snap.jbt_ref_pop, snap.n_total);
    }

    Ok(snap)
}

pub fn enumerate_to_snapshot(
    n: u32,
    m: usize,
    pre: PreCsr,
    jbt_ref_pop: &[i32],
) -> Result<Snapshot> {
    let b = bitwidth(m);
    let total_roots = ((n / 2) as usize) * n as usize;
    if pre.n_roots != total_roots {
        bail!(
            "pre.offsets len mismatch: got {}, expected {}",
            pre.n_roots,
            total_roots
        );
    }

    let mut all_frontiers: Vec<RootFrontier> =
        (0..total_roots).map(|_| RootFrontier::default()).collect();
    let mem_budget = memory_budget_bytes();

    // Seed (0,0) with one empty code (k=0) at mask 0 with weight 1.
    {
        let rf = &mut all_frontiers[0];
        let b0 = rf.get_bucket_mut(0);
        b0.append_batch(vec![0u128], vec![1 as Weight]);
    }

    let mut out = OutBuckets::default();

    let pb = ProgressBar::new(total_roots as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40} {pos}/{len} roots {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    // small loop hoist to avoid recomputing every survivor
    let evil_cut = total_roots - n as usize;

    for i in 0..total_roots {
        {
            let rf = &mut all_frontiers[i];
            rf.flush();
        }
        let (pmasks, buckets) = {
            let rf = &mut all_frontiers[i];
            let pmasks = mem::take(&mut rf.masks);
            let buckets = mem::take(&mut rf.buckets);
            rf.index.clear();
            (pmasks, buckets)
        };

        report_memory_after_vacate(i, mem_budget)?;

        let s = pre.offsets[i];
        let e = pre.offsets[i + 1];

        // Apply limit only to the (0,0) bucket = root index 0
        let e_eff = if i == 0 {
            first_bucket_limit()
                .map(|limit| s + (e - s).min(limit))
                .unwrap_or(e)
        } else {
            e
        };

        pb.set_message(format!(
            "root={} pre={} pmasks={}",
            i,
            e_eff - s,
            pmasks.len()
        ));

        if s == e || pmasks.is_empty() {
            pb.inc(1);
            continue;
        }

        // --- parallelized vacate of this root ---
        // Each worker returns: (frontier_map, completed_map), both thread-local.
        // frontier_map: key=(root_code, new_mask) -> (codes, weights)
        // completed_map: key=popkey -> (codes, weights)
        let jobs: Vec<(
            AHashMap<(i32, u64), (Vec<u128>, Vec<Weight>)>,
            AHashMap<u64, (Vec<u128>, Vec<Weight>)>,
        )> = (s..e_eff)
            .into_par_iter()
            .map(|k_pre| {
                let pmask_pre = pre.masks[k_pre];
                let pop_pre = pre.pops[k_pre] as u32;
                let jidx_pre = pre.jidx[k_pre];

                // find survivors
                let mut survivors = Vec::<usize>::new();
                survivors.reserve(pmasks.len());
                for (idx, &pm) in pmasks.iter().enumerate() {
                    if (pm & pmask_pre) == 0 {
                        survivors.push(idx);
                    }
                }
                if survivors.is_empty() {
                    return (AHashMap::default(), AHashMap::default());
                }

                // group by destination
                let mut group: AHashMap<(i32, u64), SmallVec<[usize; 8]>> = AHashMap::default();
                for &idx_pm in &survivors {
                    let new_mask = pmasks[idx_pm] | pmask_pre;
                    let do_evil = i < evil_cut; // skip last N roots
                    if do_evil && detect_evil_pmask(new_mask, n) {
                        continue;
                    }

                    let root_code: i32 = match find_root(new_mask, n) {
                        None => -1,
                        Some((u, v)) => (u as i32) * (n as i32) + v as i32,
                    };
                    group
                        .entry((root_code, new_mask))
                        .or_insert_with(|| SmallVec::new())
                        .push(idx_pm);
                }
                if group.is_empty() {
                    return (AHashMap::default(), AHashMap::default());
                }

                // local accumulators
                let mut frontier_map: AHashMap<(i32, u64), (Vec<u128>, Vec<Weight>)> =
                    AHashMap::default();
                let mut completed_map: AHashMap<u64, (Vec<u128>, Vec<Weight>)> =
                    AHashMap::default();

                if pop_pre == n {
                    // no signature update; codes unchanged
                    for ((root_code, new_mask), idx_list) in group.into_iter() {
                        let mut codes_cat = Vec::<u128>::new();
                        let mut w_cat = Vec::<Weight>::new();
                        for &idx_pm in &idx_list {
                            let bkt = &buckets[idx_pm];
                            if bkt.codes.is_empty() {
                                continue;
                            }
                            codes_cat.extend_from_slice(&bkt.codes);
                            w_cat.extend_from_slice(&bkt.weights);
                        }
                        if codes_cat.is_empty() {
                            continue;
                        }

                        if root_code == -1 {
                            // completed → group by pop-key locally
                            let mut by_key: AHashMap<u64, (Vec<u128>, Vec<Weight>)> =
                                AHashMap::default();
                            for (&c, &w) in codes_cat.iter().zip(w_cat.iter()) {
                                let key = code_pop_key(c, b, jbt_ref_pop);
                                let entry = by_key
                                    .entry(key)
                                    .or_insert_with(|| (Vec::new(), Vec::new()));
                                entry.0.push(c);
                                entry.1.push(w);
                            }
                            // merge into completed_map
                            for (key, (cc, ww)) in by_key {
                                let ent = completed_map
                                    .entry(key)
                                    .or_insert_with(|| (Vec::new(), Vec::new()));
                                ent.0.extend(cc);
                                ent.1.extend(ww);
                            }
                        } else {
                            // frontier destination
                            let ent = frontier_map
                                .entry((root_code, new_mask))
                                .or_insert_with(|| (Vec::new(), Vec::new()));
                            ent.0.extend(codes_cat);
                            ent.1.extend(w_cat);
                        }
                    }
                } else {
                    // signature update: insert jidx_pre once into each code
                    for ((root_code, new_mask), idx_list) in group.into_iter() {
                        if root_code == -1 {
                            // completed → compute codes2 then bucket per pop-key
                            let mut by_key: AHashMap<u64, (Vec<u128>, Vec<Weight>)> =
                                AHashMap::default();
                            for &idx_pm in &idx_list {
                                let bkt = &buckets[idx_pm];
                                if bkt.codes.is_empty() {
                                    continue;
                                }
                                for (&c, &w) in bkt.codes.iter().zip(bkt.weights.iter()) {
                                    let (c2, _ins) = code_insert(c, jidx_pre, b);
                                    let key = code_pop_key(c2, b, jbt_ref_pop);
                                    let entry = by_key
                                        .entry(key)
                                        .or_insert_with(|| (Vec::new(), Vec::new()));
                                    entry.0.push(c2);
                                    entry.1.push(w);
                                }
                            }
                            for (key, (cc, ww)) in by_key {
                                let ent = completed_map
                                    .entry(key)
                                    .or_insert_with(|| (Vec::new(), Vec::new()));
                                ent.0.extend(cc);
                                ent.1.extend(ww);
                            }
                        } else {
                            // frontier destination
                            let ent = frontier_map
                                .entry((root_code, new_mask))
                                .or_insert_with(|| (Vec::new(), Vec::new()));
                            for &idx_pm in &idx_list {
                                let bkt = &buckets[idx_pm];
                                if bkt.codes.is_empty() {
                                    continue;
                                }
                                for (&c, &w) in bkt.codes.iter().zip(bkt.weights.iter()) {
                                    let (c2, _ins) = code_insert(c, jidx_pre, b);
                                    ent.0.push(c2);
                                    ent.1.push(w);
                                }
                            }
                        }
                    }
                }

                (frontier_map, completed_map)
            })
            .collect();

        // Merge thread-local accumulators into global structures (sequential)
        for (frontier_map, completed_map) in jobs {
            for ((root_code, new_mask), (codes, w)) in frontier_map {
                if root_code == -1 {
                    // Shouldn't happen here, but guard anyway
                    let mut by_key: AHashMap<u64, (Vec<u128>, Vec<Weight>)> = AHashMap::default();
                    for (&c, &ww) in codes.iter().zip(w.iter()) {
                        let key = code_pop_key(c, b, jbt_ref_pop);
                        let entry = by_key
                            .entry(key)
                            .or_insert_with(|| (Vec::new(), Vec::new()));
                        entry.0.push(c);
                        entry.1.push(ww);
                    }
                    for (key, (cc, ww)) in by_key {
                        out.append_completed(key, cc, ww);
                    }
                } else {
                    let rf_dst = &mut all_frontiers[root_code as usize];
                    let bdst = rf_dst.get_bucket_mut(new_mask);
                    bdst.append_batch(codes, w);
                }
            }
            for (key, (codes, w)) in completed_map {
                out.append_completed(key, codes, w);
            }
        }

        pb.inc(1);
    }
    pb.finish_and_clear();

    out.flush_all();

    let sat = SATURATED_WEIGHTS.load(Ordering::Relaxed);
    if sat > 0 {
        eprintln!("[warn] weight saturations (u32->clamped): {}", sat);
    }

    build_snapshot_from_out(out, b, jbt_ref_pop, n as i32)
}

fn build_snapshot_from_out(
    mut out: OutBuckets,
    b: u32,
    jbt_ref_pop: &[i32],
    n_total: i32,
) -> Result<Snapshot> {
    let mut keys: Vec<u64> = out.by_key.keys().copied().collect();
    keys.sort_unstable();

    let mut buckets: Vec<Bucket> = Vec::with_capacity(keys.len());

    for key in keys {
        // take ownership of this bucket (move out, no clone)
        let bkt = out.by_key.remove(&key).unwrap();

        let n_rows = bkt.codes.len();

        // rows_data: Vec<i32>, indptr: Vec<i64>, weights: Vec<f64>, key: Vec<i32>
        let total_len: usize = bkt.codes.iter().map(|&c| code_len_u128(c)).sum();
        let mut rows_data: Vec<i32> = Vec::with_capacity(total_len);
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        indptr.push(0);

        for &c in &bkt.codes {
            let mut cnt = 0i64;
            for j in code_iter(c, b) {
                rows_data.push(j as i32);
                cnt += 1;
            }
            let last = *indptr.last().unwrap();
            indptr.push(last + cnt);
        }

        // Cast `u32` weights to `f64` only here:
        let weights: Vec<f64> = bkt.weights.iter().map(|&w| w as f64).collect();

        // decode pop-key back into Vec<i32>
        let mut key_vec: Vec<i32> = Vec::new();
        let k = (key & 0xF) as u32;
        let mut shift = 4u32;
        for _ in 0..k {
            let p = ((key >> shift) & 0xF) as i32;
            key_vec.push(p);
            shift += 4;
        }

        buckets.push(Bucket {
            rows_data,
            indptr,
            weights,
            key: key_vec,
        });
    }

    let jpop_vec: Vec<i32> = jbt_ref_pop.to_vec();
    Ok(Snapshot {
        buckets,
        jbt_ref_pop: jpop_vec,
        n_total,
        compat: StdHashMap::new(),
    })
}
