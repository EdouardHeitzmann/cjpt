use anyhow::{Context, Result};
use ndarray::Array1;
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use std::{env, fs::File};

// ----------------- Data structures -----------------

#[derive(Debug, Clone)]
struct Bucket {
    rows_data: Vec<i32>,
    indptr: Vec<i64>,
    weights: Vec<f64>,
    key: Vec<i32>, // pop tuple; empty [] represents the neutral bucket key ()
}
impl Bucket {
    #[inline]
    fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }
    #[inline]
    fn row_slice(&self, r: usize) -> &[i32] {
        let lo = self.indptr[r] as usize;
        let hi = self.indptr[r + 1] as usize;
        &self.rows_data[lo..hi]
    }
}

#[derive(Debug)]
struct Snapshot {
    buckets: Vec<Bucket>,
    jbt_ref_pop: Vec<i32>,
    n_total: i32,
    // compat: pop -> (key1, key2)
    compat: HashMap<i32, (Vec<i32>, Vec<i32>)>,
}

// ----------------- NPZ reading -----------------

fn read_array1_i64<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<Array1<i64>> {
    let arr: Array1<i64> = npz.by_name(name).with_context(|| format!("missing {}", name))?;
    Ok(arr)
}
fn read_array1_i32<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<Array1<i32>> {
    let arr: Array1<i32> = npz.by_name(name).with_context(|| format!("missing {}", name))?;
    Ok(arr)
}
fn read_array1_f64<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<Array1<f64>> {
    let arr: Array1<f64> = npz.by_name(name).with_context(|| format!("missing {}", name))?;
    Ok(arr)
}

fn load_snapshot(path: &str) -> Result<Snapshot> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let mut npz = NpzReader::new(f).context("read npz")?;

    let n_arr = read_array1_i32(&mut npz, "meta_N.npy")?;
    let n_total = n_arr[0];

    let jbt_ref_pop = read_array1_i32(&mut npz, "meta_jbt_ref_pop.npy")?.to_vec();

    let keys_indptr = read_array1_i64(&mut npz, "meta_bucket_keys_indptr.npy")?;
    let num_buckets = if keys_indptr.len() == 0 {
        0
    } else {
        keys_indptr.len() - 1
    };

    let mut buckets = Vec::with_capacity(num_buckets);
    for b in 0..num_buckets {
        let data = read_array1_i32(&mut npz, &format!("b{}_rows_data.npy", b))?.to_vec();
        let indptr = read_array1_i64(&mut npz, &format!("b{}_rows_indptr.npy", b))?.to_vec();
        let weights = read_array1_f64(&mut npz, &format!("b{}_weights.npy", b))?.to_vec();
        let key = read_array1_i32(&mut npz, &format!("b{}_key.npy", b))?.to_vec();
        buckets.push(Bucket {
            rows_data: data,
            indptr,
            weights,
            key,
        });
    }

    // compat tables
    let compat_pops = read_array1_i32(&mut npz, "meta_compat_pops.npy")?;
    let mut compat: HashMap<i32, (Vec<i32>, Vec<i32>)> = HashMap::new();
    for p in compat_pops.iter() {
        let k1 = read_array1_i32(&mut npz, &format!("compat_p{}_key1.npy", p))?.to_vec();
        let k2 = read_array1_i32(&mut npz, &format!("compat_p{}_key2.npy", p))?.to_vec();
        compat.insert(*p, (k1, k2));
    }

    Ok(Snapshot {
        buckets,
        jbt_ref_pop,
        n_total,
        compat,
    })
}

// ----------------- Pair utilities -----------------

#[inline]
fn key_sorted_vec(key: &[i32]) -> Vec<i32> {
    let mut v = key.to_vec();
    v.sort();
    v
}
#[inline]
fn compat_key_sorted(key: &[i32], n_total: i32) -> Vec<i32> {
    let mut v: Vec<i32> = key.iter().map(|&p| n_total - p).collect();
    v.sort();
    v
}

fn build_key_to_idx(buckets: &[Bucket]) -> HashMap<Vec<i32>, usize> {
    let mut map = HashMap::with_capacity(buckets.len());
    for (idx, b) in buckets.iter().enumerate() {
        map.insert(key_sorted_vec(&b.key), idx);
    }
    map
}

// rows_by_jbt: x -> sorted Vec<row_idx>
fn build_rows_by_jbt(bucket: &Bucket) -> HashMap<i32, Vec<usize>> {
    let mut m: HashMap<i32, Vec<usize>> = HashMap::new();
    for r in 0..bucket.n_rows() {
        for &v in bucket.row_slice(r) {
            m.entry(v).or_default().push(r);
        }
    }
    for rows in m.values_mut() {
        rows.sort_unstable();
    }
    m
}

// ----------------- Candidate precompute -----------------

// Precompute candidate xs for each j used in bucket1 (filtered to those present in bucket2)
fn precompute_candidates_for_bucket1(
    bucket1: &Bucket,
    rows_by_jbt: &HashMap<i32, Vec<usize>>,
    jbt_ref_pop: &[i32],
    n_total: i32,
    compat: &HashMap<i32, (Vec<i32>, Vec<i32>)>,
) -> HashMap<i32, Vec<i32>> {
    let mut all_j: HashSet<i32> = HashSet::new();
    for r in 0..bucket1.n_rows() {
        for &j in bucket1.row_slice(r) {
            if jbt_ref_pop[j as usize] != 0 {
                all_j.insert(j);
            }
        }
    }
    let mut out: HashMap<i32, Vec<i32>> = HashMap::with_capacity(all_j.len());
    for j in all_j {
        let pop = jbt_ref_pop[j as usize];
        let (k1_ref, k2_ref): (&Vec<i32>, &Vec<i32>) = if pop > n_total / 2 {
            let pair = compat
                .get(&(n_total - pop))
                .expect("compat missing pop");
            (&pair.1, &pair.0) // (key2, key1)
        } else {
            let pair = compat.get(&pop).expect("compat missing pop");
            (&pair.0, &pair.1)
        };
        let mut cands = Vec::<i32>::new();
        for (i, &v) in k1_ref.iter().enumerate() {
            if v == j {
                let x = k2_ref[i];
                if rows_by_jbt.contains_key(&x) {
                    cands.push(x);
                }
            }
        }
        cands.sort_unstable();
        cands.dedup();
        out.insert(j, cands);
    }
    out
}

// ----------------- Per-row solver (optimized) -----------------

fn subtotal_for_pair(
    bucket1: &Bucket,
    bucket2: &Bucket,
    jbt_ref_pop: &[i32],
    n_total: i32,
    compat: &HashMap<i32, (Vec<i32>, Vec<i32>)>,
    // extra precomputes:
    rows_by_jbt: &HashMap<i32, Vec<usize>>,
    cand_map: &HashMap<i32, Vec<i32>>,
) -> f64 {
    // neutral fast-path: empty key means every row compatible with every row
    if bucket1.key.is_empty() {
        let s1: f64 = bucket1.weights.iter().copied().sum();
        let s2: f64 = bucket2.weights.iter().copied().sum();
        return s1 * s2;
    }

    let n_rows2 = bucket2.n_rows();
    let mut subtotal = 0.0f64;

    // pop multiplicities from bucket1 key (hint for disjointness)
    let mut pop_mult: HashMap<i32, i32> = HashMap::new();
    for &p in &bucket1.key {
        *pop_mult.entry(p).or_insert(0) += 1;
    }

    // caches (per pair)
    let mut union_cache: HashMap<i32, Vec<bool>> = HashMap::new();
    let mut count_cache: HashMap<i32, Vec<i32>> = HashMap::new();

    // per-row solve
    'rowloop: for r1 in 0..bucket1.n_rows() {
        let row = bucket1.row_slice(r1);
        let w1 = bucket1.weights[r1] as f64;

        // classify positions
        let mut unique_positions: Vec<usize> = Vec::new();
        let mut colliding_positions: Vec<usize> = Vec::new();

        for (i, &j) in row.iter().enumerate() {
            let pop = jbt_ref_pop[j as usize];
            if pop == 0 {
                continue;
            }
            let cands = cand_map.get(&j).map(|v| v.as_slice()).unwrap_or(&[]);
            if cands.is_empty() {
                // impossible row
                continue 'rowloop;
            }
            let mult = *pop_mult.get(&pop).unwrap_or(&0);
            if mult <= 1 {
                unique_positions.push(i);
            } else {
                colliding_positions.push(i);
            }
        }

        // working mask (feasible rows in bucket2)
        let mut mask = vec![true; n_rows2];
        // effective weights per row (weights2 * per-position multiplicities)
        let mut eff = bucket2.weights.clone();

        // Unique-pop fast path
        for &i in &unique_positions {
            let j = row[i];
            if !union_cache.contains_key(&j) {
                let cands = cand_map.get(&j).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut union = vec![false; n_rows2];
                let mut counts = vec![0i32; n_rows2];
                for &x in cands {
                    if let Some(rows) = rows_by_jbt.get(&x) {
                        for &r in rows {
                            union[r] = true;
                            counts[r] += 1;
                        }
                    }
                }
                union_cache.insert(j, union);
                count_cache.insert(j, counts);
            }
            let union = union_cache.get(&j).unwrap();
            let counts = count_cache.get(&j).unwrap();
            let mut any = false;
            for r in 0..n_rows2 {
                mask[r] = mask[r] && union[r];
                if mask[r] {
                    eff[r] *= counts[r] as f64;
                    any = true;
                }
            }
            if !any {
                continue 'rowloop;
            }
        }

        // remaining positions (same-pop collisions only)
        let rem: Vec<i32> = colliding_positions.iter().map(|&i| row[i]).collect();
        if rem.is_empty() {
            // sum eff over mask
            let mut s = 0.0f64;
            for r in 0..n_rows2 {
                if mask[r] {
                    s += eff[r];
                }
            }
            subtotal += w1 * s;
            continue;
        }

        // Disjoint fast path: check for overlaps among remaining candidate sets
        let mut seen = HashSet::new();
        let mut overlap = false;
        let mut cand_lists: Vec<&[i32]> = Vec::with_capacity(rem.len());
        for &j in &rem {
            let cands = cand_map.get(&j).map(|v| v.as_slice()).unwrap_or(&[]);
            for &x in cands {
                if !seen.insert(x) {
                    overlap = true;
                    break;
                }
            }
            cand_lists.push(cands);
            if overlap {
                break;
            }
        }
        if !overlap {
            // vectorized multiply of counts per remaining position
            let mut s = 0.0f64;
            for r in 0..n_rows2 {
                if !mask[r] {
                    continue;
                }
                let mut mult = eff[r];
                for &cands in &cand_lists {
                    let mut cnt = 0i32;
                    for &x in cands {
                        if let Some(rows) = rows_by_jbt.get(&x) {
                            if rows.binary_search(&r).is_ok() {
                                cnt += 1;
                            }
                        }
                    }
                    mult *= cnt as f64;
                }
                s += mult;
            }
            subtotal += w1 * s;
            continue;
        }

        // Fallback: small branch-and-bound over overlapping candidates (with injectivity)
        fn intersect_in_place(dst: &mut [bool], rows: &[usize]) -> bool {
            let mut any = false;
            for (i, v) in dst.iter_mut().enumerate() {
                if *v {
                    *v = rows.binary_search(&i).is_ok();
                }
                if *v {
                    any = true;
                }
            }
            any
        }

        fn rec_overlap(
            idxs: &[i32], // remaining j's
            mask: &[bool],
            eff: &[f64],
            rows_by_jbt: &HashMap<i32, Vec<usize>>,
            cand_map: &HashMap<i32, Vec<i32>>,
            used_x: &mut HashSet<i32>, // enforce injectivity
        ) -> f64 {
            // feasibility: each remaining j has at least one unused candidate row under mask
            for &j in idxs {
                let cands = cand_map.get(&j).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut ok = false;
                'outer: for &x in cands {
                    if used_x.contains(&x) {
                        continue;
                    }
                    if let Some(rows) = rows_by_jbt.get(&x) {
                        for &r in rows {
                            if mask[r] {
                                ok = true;
                                break 'outer;
                            }
                        }
                    }
                }
                if !ok {
                    return 0.0;
                }
            }
            if idxs.is_empty() {
                let mut s = 0.0f64;
                for (r, &m) in mask.iter().enumerate() {
                    if m {
                        s += eff[r];
                    }
                }
                return s;
            }
            // pivot: j with fewest viable unused xs under current mask
            let mut best_j = idxs[0];
            let mut best_list: Vec<i32> = Vec::new();
            let mut best_cnt = usize::MAX;
            for &j in idxs {
                let cands = cand_map.get(&j).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut viable: Vec<i32> = Vec::new();
                for &x in cands {
                    if used_x.contains(&x) {
                        continue;
                    }
                    if let Some(rows) = rows_by_jbt.get(&x) {
                        if rows.iter().any(|&r| mask[r]) {
                            viable.push(x);
                        }
                    }
                }
                if viable.is_empty() {
                    return 0.0;
                }
                if viable.len() < best_cnt {
                    best_cnt = viable.len();
                    best_j = j;
                    best_list = viable;
                    if best_cnt == 1 {
                        break;
                    }
                }
            }
            // branch
            let mut total = 0.0f64;
            let rest: Vec<i32> = idxs.iter().copied().filter(|&x| x != best_j).collect();
            for x in best_list {
                if let Some(rows) = rows_by_jbt.get(&x) {
                    let mut new_mask = mask.to_vec();
                    if !intersect_in_place(&mut new_mask, rows) {
                        continue;
                    }
                    used_x.insert(x);
                    total += rec_overlap(&rest, &new_mask, eff, rows_by_jbt, cand_map, used_x);
                    used_x.remove(&x);
                }
            }
            total
        }

        let add = {
            let mut used = HashSet::<i32>::new();
            rec_overlap(&rem, &mask, &eff, rows_by_jbt, cand_map, &mut used)
        };
        subtotal += w1 * add;
    }

    subtotal
}

// ----------------- Pair profiling + parallel driver -----------------

#[derive(Debug)]
struct PairResult {
    key_left: Vec<i32>,
    key_right: Vec<i32>,
    rows1: usize,
    rows2: usize,
    subtotal: f64,
    t_index: f64,
    t_cands: f64,
    t_solve: f64,
    t_total: f64,
    factor: f64,
}

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .context("usage: matcher <buckets_snapshot.npz>")?;
    let t0 = Instant::now();
    let snap = load_snapshot(&path)?;
    println!(
        "Loaded: N={}, buckets={}, jbt_ref_pop={}",
        snap.n_total,
        snap.buckets.len(),
        snap.jbt_ref_pop.len()
    );

    let key_to_idx = build_key_to_idx(&snap.buckets);
    // Build unordered pairs via map lookup
    let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();
    let mut tasks: Vec<(usize, usize, usize, usize, f64)> = Vec::new(); // (left,right,i,j,factor)

    for (i, bi) in snap.buckets.iter().enumerate() {
        let compat_sorted = compat_key_sorted(&key_sorted_vec(&bi.key), snap.n_total);
        if let Some(&j) = key_to_idx.get(&compat_sorted) {
            let pair = if i <= j { (i, j) } else { (j, i) };
            if seen_pairs.insert(pair) {
                // pick smaller as left
                let (left, right) = if snap.buckets[pair.0].n_rows() <= snap.buckets[pair.1].n_rows() {
                    (pair.0, pair.1)
                } else {
                    (pair.1, pair.0)
                };
                let factor = if pair.0 != pair.1 { 2.0 } else { 1.0 };
                tasks.push((left, right, pair.0, pair.1, factor));
            }
        }
    }

    // Cost heuristic: biggest first
    tasks.sort_by_key(|&(left, right, _, _, _)| {
        // negative key for descending
        let cost = (snap.buckets[left].n_rows() as u64)
            * (snap.buckets[right].n_rows() as u64)
            * (std::cmp::max(1, snap.buckets[left].key.len()) as u64);
        std::cmp::Reverse(cost)
    });

    // Run tasks in parallel, collect results
    let results: Vec<PairResult> = tasks
        .par_iter()
        .map(|&(left, right, i, j, factor)| {
            let key_left = snap.buckets[left].key.clone();
            let key_right = snap.buckets[right].key.clone();

            let t_pair0 = Instant::now();

            // 1) Index bucket2
            let t_index0 = Instant::now();
            let rows_by_jbt = build_rows_by_jbt(&snap.buckets[right]);
            let t_index = t_index0.elapsed().as_secs_f64();

            // 2) Precompute candidates for bucket1 j's
            let t_cands0 = Instant::now();
            let cand_map = precompute_candidates_for_bucket1(
                &snap.buckets[left],
                &rows_by_jbt,
                &snap.jbt_ref_pop,
                snap.n_total,
                &snap.compat,
            );
            let t_cands = t_cands0.elapsed().as_secs_f64();

            // 3) Solve
            let t_solve0 = Instant::now();
            let mut subtotal = subtotal_for_pair(
                &snap.buckets[left],
                &snap.buckets[right],
                &snap.jbt_ref_pop,
                snap.n_total,
                &snap.compat,
                &rows_by_jbt,
                &cand_map,
            );
            subtotal *= factor;
            let t_solve = t_solve0.elapsed().as_secs_f64();

            let t_total = t_pair0.elapsed().as_secs_f64();

            PairResult {
                key_left,
                key_right,
                rows1: snap.buckets[left].n_rows(),
                rows2: snap.buckets[right].n_rows(),
                subtotal,
                t_index,
                t_cands,
                t_solve,
                t_total,
                factor,
            }
        })
        .collect();

    // Print per-pair lines (ordered by descending cost like tasks)
    let mut omega = 0.0f64;
    for r in &results {
        println!(
            "[pair {:?} vs {:?}{}] rows1={}, rows2={} | index={:.3}s, cands={:.3}s, solve={:.3}s → total={:.3}s | subtotal={:.6}",
            r.key_left,
            r.key_right,
            if r.factor == 2.0 { " x2" } else { "" },
            r.rows1,
            r.rows2,
            r.t_index,
            r.t_cands,
            r.t_solve,
            r.t_total,
            r.subtotal
        );
        omega += r.subtotal;
    }

    let wall = t0.elapsed().as_secs_f64();
    println!(
        "Omega total: {:.6} (pairs={}, wall={:.3}s, sum_pair_total={:.3}s, sum_pair_solve={:.3}s)",
        omega,
        results.len(),
        wall,
        results.iter().map(|r| r.t_total).sum::<f64>(),
        results.iter().map(|r| r.t_solve).sum::<f64>(),
    );

    Ok(())
}

