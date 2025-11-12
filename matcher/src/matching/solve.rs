use std::collections::{HashMap, HashSet};

use super::types::Bucket;

// x -> sorted Vec<row_idx>
pub fn build_rows_by_jbt(bucket: &Bucket) -> HashMap<i32, Vec<usize>> {
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

// candidates per j (filtered to x present in bucket2)
pub fn precompute_candidates_for_bucket1(
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
        let (k1, k2): (&Vec<i32>, &Vec<i32>) = if pop > n_total / 2 {
            let pair = compat.get(&(n_total - pop)).expect("compat missing pop");
            (&pair.1, &pair.0) // swapped
        } else {
            let pair = compat.get(&pop).expect("compat missing pop");
            (&pair.0, &pair.1)
        };
        let mut cands = Vec::<i32>::new();
        for (i, &v) in k1.iter().enumerate() {
            if v == j {
                let x = k2[i];
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

// per-pair subtotal (same logic you’re running now)
pub fn subtotal_for_pair(
    bucket1: &Bucket,
    bucket2: &Bucket,
    jbt_ref_pop: &[i32],
    _n_total: i32,
    _compat: &std::collections::HashMap<i32, (Vec<i32>, Vec<i32>)>,
    rows_by_jbt: &HashMap<i32, Vec<usize>>,
    cand_map: &HashMap<i32, Vec<i32>>,
) -> f64 {
    if bucket1.key.is_empty() {
        let s1: f64 = bucket1.weights.iter().copied().sum();
        let s2: f64 = bucket2.weights.iter().copied().sum();
        return s1 * s2;
    }

    let n_rows2 = bucket2.n_rows();
    let mut subtotal = 0.0f64;

    let mut pop_mult: HashMap<i32, i32> = HashMap::new();
    for &p in &bucket1.key {
        *pop_mult.entry(p).or_insert(0) += 1;
    }

    let mut union_cache: HashMap<i32, Vec<bool>> = HashMap::new();
    let mut count_cache: HashMap<i32, Vec<i32>> = HashMap::new();

    'rowloop: for r1 in 0..bucket1.n_rows() {
        let row = bucket1.row_slice(r1);
        let w1 = bucket1.weights[r1] as f64;

        let mut unique_positions = Vec::new();
        let mut colliding_positions = Vec::new();

        for (i, &j) in row.iter().enumerate() {
            let pop = jbt_ref_pop[j as usize];
            if pop == 0 {
                continue;
            }
            let cands = cand_map.get(&j).map(|v| v.as_slice()).unwrap_or(&[]);
            if cands.is_empty() {
                continue 'rowloop;
            }
            if *pop_mult.get(&pop).unwrap_or(&0) <= 1 {
                unique_positions.push(i);
            } else {
                colliding_positions.push(i);
            }
        }

        let mut mask = vec![true; n_rows2];
        let mut eff = bucket2.weights.clone();

        // unique-pop fast path
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

        let rem: Vec<i32> = colliding_positions.iter().map(|&i| row[i]).collect();
        if rem.is_empty() {
            let mut s = 0.0f64;
            for r in 0..n_rows2 {
                if mask[r] {
                    s += eff[r];
                }
            }
            subtotal += w1 * s;
            continue;
        }

        // disjoint fast path
        let mut seen = std::collections::HashSet::new();
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

        // fallback recursion with injectivity
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
        fn rec(
            idxs: &[i32],
            mask: &[bool],
            eff: &[f64],
            rows_by_jbt: &HashMap<i32, Vec<usize>>,
            cand_map: &HashMap<i32, Vec<i32>>,
            used_x: &mut std::collections::HashSet<i32>,
        ) -> f64 {
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
            // pivot
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
            let mut total = 0.0f64;
            let rest: Vec<i32> = idxs.iter().copied().filter(|&x| x != best_j).collect();
            for x in best_list {
                if let Some(rows) = rows_by_jbt.get(&x) {
                    let mut new_mask = mask.to_vec();
                    if !intersect_in_place(&mut new_mask, rows) {
                        continue;
                    }
                    used_x.insert(x);
                    total += rec(&rest, &new_mask, eff, rows_by_jbt, cand_map, used_x);
                    used_x.remove(&x);
                }
            }
            total
        }
        let add = {
            let mut used = HashSet::<i32>::new();
            rec(&rem, &mask, &eff, rows_by_jbt, cand_map, &mut used)
        };
        subtotal += w1 * add;
    }

    subtotal
}
