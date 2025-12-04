use anyhow::{Context, Result};
use rayon::prelude::*;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use matcher::matching::{build_rows_by_jbt, load_bucket_shard, load_schedule_meta, precompute_candidates_for_bucket1, subtotal_for_pair};
use matcher::runtime;

fn usage() -> ! {
    eprintln!("usage: snapshot_pair_run <schedule.json> <bucket_dir>");
    std::process::exit(1);
}

fn parse_args() -> (PathBuf, PathBuf) {
    let mut args = env::args().skip(1);
    let schedule = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let bucket_dir = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    if args.next().is_some() {
        usage();
    }
    (schedule, bucket_dir)
}

fn main() -> Result<()> {
    runtime::configure_thread_pool();

    let (schedule_path, bucket_dir) = parse_args();
    eprintln!("[pairs] loading schedule {}", schedule_path.display());
    let schedule = load_schedule_meta(&schedule_path)?;
    let bucket_meta = {
        let mut v = vec![None; schedule.buckets.len()];
        for m in &schedule.buckets {
            if m.id < v.len() {
                v[m.id] = Some(m.clone());
            }
        }
        Arc::new(v)
    };
    eprintln!(
        "[pairs] tasks={}, buckets={}, n_label={}, N={}",
        schedule.tasks.len(),
        schedule.buckets.len(),
        schedule.n_label,
        schedule.n_total
    );

    let bucket_path = |idx: usize| -> PathBuf {
        bucket_dir.join(format!("snapshot_bucket_n{}_{}.npz", schedule.n_label, idx))
    };

    let omega: f64 = schedule
        .tasks
        .par_iter()
        .map(|task| {
            let t_pair0 = Instant::now();
            let path_left = bucket_path(task.left);
            let path_right = bucket_path(task.right);

            let bucket1 = load_bucket_shard(&path_left)
                .with_context(|| format!("load bucket {}", path_left.display()))?;
            let bucket2 = load_bucket_shard(&path_right)
                .with_context(|| format!("load bucket {}", path_right.display()))?;
            let t_load = t_pair0.elapsed().as_secs_f64();

            let t_index0 = Instant::now();
            let rows_by_jbt = build_rows_by_jbt(&bucket2);
            let t_index = t_index0.elapsed().as_secs_f64();

            let t_cands0 = Instant::now();
            let cand_map = precompute_candidates_for_bucket1(
                &bucket1,
                &rows_by_jbt,
                &schedule.jbt_ref_pop,
                schedule.n_total,
                &schedule.compat,
            );
            let t_cands = t_cands0.elapsed().as_secs_f64();

            let t_solve0 = Instant::now();
            let subtotal = subtotal_for_pair(
                &bucket1,
                &bucket2,
                &schedule.jbt_ref_pop,
                schedule.n_total,
                &schedule.compat,
                &rows_by_jbt,
                &cand_map,
            ) * task.factor;
            let t_solve = t_solve0.elapsed().as_secs_f64();

            let t_total = t_pair0.elapsed().as_secs_f64();

            let meta_left = bucket_meta
                .get(task.left)
                .and_then(|m| m.as_ref())
                .map(|m| m.key.clone())
                .unwrap_or_else(|| bucket1.key.clone());
            let meta_right = bucket_meta
                .get(task.right)
                .and_then(|m| m.as_ref())
                .map(|m| m.key.clone())
                .unwrap_or_else(|| bucket2.key.clone());

            println!(
                "[pair {:5} vs {:5}{}] rows1={:8}, rows2={:8} | load={:.3}s, index={:.3}s, cands={:.3}s, solve={:.3}s → total={:.3}s | subtotal={:.6} | key_left={:?} key_right={:?}",
                task.left,
                task.right,
                if task.factor == 2.0 { " x2" } else { "" },
                bucket1.n_rows(),
                bucket2.n_rows(),
                t_load,
                t_index,
                t_cands,
                t_solve,
                t_total,
                subtotal,
                meta_left,
                meta_right
            );

            Ok::<f64, anyhow::Error>(subtotal)
        })
        .try_reduce(|| 0.0, |a, b| Ok(a + b))?;

    println!(
        "Omega total: {:.6} (tasks={}, n_label={}, N={})",
        omega,
        schedule.tasks.len(),
        schedule.n_label,
        schedule.n_total
    );

    Ok(())
}
