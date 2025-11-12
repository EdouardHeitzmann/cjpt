// src/enumeration/compat.rs

use std::collections::HashMap;

/// True iff the bitwise overlap of a and b is a single *contiguous* run of 1s.
/// Mirrors Python's:
///   overlap = a & b
///   if overlap == 0: return False
///   start = tz(overlap)
///   shifted = overlap >> start
///   return (shifted + 1) & shifted == 0
#[inline]
fn contiguous_overlap(a: u16, b: u16) -> bool {
    let o = (a & b) as u32;
    if o == 0 {
        return false;
    }
    let s = o >> o.trailing_zeros(); // align the run to LSB
    (s & (s + 1)) == 0 // s is 2^k - 1
}

/// Port of your `determine_compatibility` over 3×u16 component masks.
/// Zeros in comps are ignored.
#[inline]
fn determine_compatibility(c1: &[u16; 3], c2: &[u16; 3]) -> bool {
    // Compact non-zeros, preserve order
    let a0: Vec<u16> = c1.iter().copied().filter(|&x| x != 0).collect();
    let b0: Vec<u16> = c2.iter().copied().filter(|&x| x != 0).collect();

    // WLOG |a| <= |b|
    let (a, b) = if a0.len() <= b0.len() {
        (a0, b0)
    } else {
        (b0, a0)
    };

    match (a.len(), b.len()) {
        (1, 3) => {
            let x = a[0];
            contiguous_overlap(x, b[2])
                && contiguous_overlap(x, b[1])
                && contiguous_overlap(x, b[0])
        }
        (2, 2) => {
            // Follow your Python indexing: first_comp = [1], second_comp = [0]
            let (a1, a0) = (a[1], a[0]);
            let (b1, b0) = (b[1], b[0]);
            let co11 = contiguous_overlap(a1, b1);
            if !co11 {
                return false;
            }
            let co22 = contiguous_overlap(a0, b0);
            if !co22 {
                return false;
            }
            let co12 = contiguous_overlap(a1, b0);
            let co21 = contiguous_overlap(a0, b1);
            co12 ^ co21
        }
        (1, 2) => {
            let x = a[0];
            contiguous_overlap(x, b[1]) && contiguous_overlap(x, b[0])
        }
        (1, 1) => contiguous_overlap(a[0], b[0]),
        _ => false,
    }
}

/// Build the two key arrays for a single population pair `p` vs `q = N - p`.
/// Returns `(key1, key2)` where both are parallel arrays of j indices.
pub fn compat_for_pop_pair(
    jbt_ref_pop: &[i32],
    jbt_ref_comps: &[[u16; 3]],
    n_total: i32,
    p: i32,
) -> (Vec<i32>, Vec<i32>) {
    let q = n_total - p;
    if p <= 0 || q <= 0 || p >= n_total || q >= n_total {
        return (Vec::new(), Vec::new());
    }

    // Gather indices by pop
    let idxs_p: Vec<i32> = jbt_ref_pop
        .iter()
        .enumerate()
        .filter_map(|(j, &pp)| if pp == p { Some(j as i32) } else { None })
        .collect();

    let idxs_q: Vec<i32> = jbt_ref_pop
        .iter()
        .enumerate()
        .filter_map(|(j, &pp)| if pp == q { Some(j as i32) } else { None })
        .collect();

    if idxs_p.is_empty() || idxs_q.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut k1 = Vec::new();
    let mut k2 = Vec::new();
    for &i in &idxs_p {
        let c1 = &jbt_ref_comps[i as usize];
        for &j in &idxs_q {
            let c2 = &jbt_ref_comps[j as usize];
            if determine_compatibility(c1, c2) {
                k1.push(i);
                k2.push(j);
            }
        }
    }
    (k1, k2)
}

/// Build a full compat map covering *all* pops `1..N-1`.
/// For each p, the value is `(key1, key2)` for `p` vs `q=N-p`.
/// The map also contains entries for `q` with lists swapped to make lookups symmetric.
pub fn build_compat_map(
    jbt_ref_pop: &[i32],
    jbt_ref_comps: &[[u16; 3]],
    n_total: i32,
) -> HashMap<i32, (Vec<i32>, Vec<i32>)> {
    let mut out: HashMap<i32, (Vec<i32>, Vec<i32>)> = HashMap::new();

    // Compute only for 1..N/2, then mirror-fill q = N - p.
    for p in 1..=((n_total - 1) / 2) {
        let (k1, k2) = compat_for_pop_pair(jbt_ref_pop, jbt_ref_comps, n_total, p);
        let q = n_total - p;
        out.insert(p, (k1.clone(), k2.clone()));
        out.insert(q, (k2, k1));
    }

    // If N is even and p == q (midpoint), ensure it's present even if empty
    if n_total % 2 == 0 {
        let mid = n_total / 2;
        out.entry(mid).or_insert_with(|| (Vec::new(), Vec::new()));
    }

    // Ensure every pop in 1..N-1 exists (empty lists if no matches)
    for p in 1..n_total {
        out.entry(p).or_insert_with(|| (Vec::new(), Vec::new()));
    }

    out
}

/// Optional: quick summary for sanity checks.
pub fn debug_summary(
    compat: &HashMap<i32, (Vec<i32>, Vec<i32>)>,
    jbt_ref_pop: &[i32],
    n_total: i32,
) {
    let mut keys: Vec<i32> = compat.keys().copied().collect();
    keys.sort_unstable();
    let nonempty = compat
        .values()
        .filter(|(a, b)| !a.is_empty() && !b.is_empty())
        .count();
    eprintln!(
        "[compat] keys present: {} of {} (1..{}); nonempty: {}",
        keys.len(),
        n_total - 1,
        n_total - 1,
        nonempty
    );

    for &p in keys.iter().take(6) {
        let (k1, k2) = compat.get(&p).unwrap();
        // spot-check the first few entries
        eprintln!("[compat] p={} lens: k1={}, k2={}", p, k1.len(), k2.len());
        for i in 0..k1.len().min(3) {
            let j1 = k1[i] as usize;
            let j2 = k2[i] as usize;
            let p1 = jbt_ref_pop.get(j1).copied().unwrap_or(-1);
            let p2 = jbt_ref_pop.get(j2).copied().unwrap_or(-1);
            eprintln!("    pair[{i}] = ({j1},{j2})  pops=({}, {})", p1, p2);
        }
    }
}
