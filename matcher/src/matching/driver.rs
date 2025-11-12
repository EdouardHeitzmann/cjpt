use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::solve::{build_rows_by_jbt, precompute_candidates_for_bucket1, subtotal_for_pair};
use super::types::{Bucket, Snapshot, compat_key_sorted, key_sorted_vec};

#[derive(Debug)]
pub struct PairResult {
    pub key_left: Vec<i32>,
    pub key_right: Vec<i32>,
    pub rows1: usize,
    pub rows2: usize,
    pub subtotal: f64,
    pub t_index: f64,
    pub t_cands: f64,
    pub t_solve: f64,
    pub t_total: f64,
    pub factor: f64,
}

fn build_key_to_idx(buckets: &[Bucket]) -> HashMap<Vec<i32>, usize> {
    let mut map = HashMap::with_capacity(buckets.len());
    for (idx, b) in buckets.iter().enumerate() {
        map.insert(key_sorted_vec(&b.key), idx);
    }
    map
}

pub fn run_all_pairs_parallel(snap: &Snapshot, verbose: bool) -> (Vec<PairResult>, f64) {
    let t0 = Instant::now();

    // build unordered tasks
    let key_to_idx = build_key_to_idx(&snap.buckets);
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut tasks: Vec<(usize, usize, f64)> = Vec::new(); // (left,right,factor)

    for (i, bi) in snap.buckets.iter().enumerate() {
        let compat_sorted = compat_key_sorted(&key_sorted_vec(&bi.key), snap.n_total);
        if let Some(&j) = key_to_idx.get(&compat_sorted) {
            let pair = if i <= j { (i, j) } else { (j, i) };
            if seen.insert(pair) {
                let (left, right) =
                    if snap.buckets[pair.0].n_rows() <= snap.buckets[pair.1].n_rows() {
                        (pair.0, pair.1)
                    } else {
                        (pair.1, pair.0)
                    };
                let factor = if pair.0 != pair.1 { 2.0 } else { 1.0 };
                tasks.push((left, right, factor));
            }
        }
    }

    // cost sort heavy first
    tasks.sort_by_key(|&(l, r, _)| {
        use std::cmp::Reverse;
        let cost = (snap.buckets[l].n_rows() as u64)
            * (snap.buckets[r].n_rows() as u64)
            * (std::cmp::max(1, snap.buckets[l].key.len()) as u64);
        Reverse(cost)
    });

    // parallel run
    let results: Vec<PairResult> = tasks
        .par_iter()
        .map(|&(left, right, factor)| {
            let key_left = snap.buckets[left].key.clone();
            let key_right = snap.buckets[right].key.clone();

            let t_pair0 = Instant::now();

            let t_index0 = Instant::now();
            let rows_by_jbt = build_rows_by_jbt(&snap.buckets[right]);
            let t_index = t_index0.elapsed().as_secs_f64();

            let t_cands0 = Instant::now();
            let cand_map = precompute_candidates_for_bucket1(
                &snap.buckets[left],
                &rows_by_jbt,
                &snap.jbt_ref_pop,
                snap.n_total,
                &snap.compat,
            );
            let t_cands = t_cands0.elapsed().as_secs_f64();

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

    let wall = t0.elapsed().as_secs_f64();

    if verbose {
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
        }
        let omega: f64 = results.iter().map(|r| r.subtotal).sum();
        println!(
            "Omega total: {:.6} (pairs={}, wall={:.3}s, sum_pair_total={:.3}s, sum_pair_solve={:.3}s)",
            omega,
            results.len(),
            wall,
            results.iter().map(|r| r.t_total).sum::<f64>(),
            results.iter().map(|r| r.t_solve).sum::<f64>(),
        );
    }

    (results, wall)
}
