use anyhow::{Context, Result, bail};
use std::env;
use std::path::PathBuf;

use matcher::matching::{load_snapshot, snapshot_to_schedule, write_bucket_shard, write_schedule_meta, Bucket};
use matcher::runtime;

fn usage() -> ! {
    eprintln!("usage: snapshot_split <snapshot.npz> <output_dir> <n_label>");
    std::process::exit(1);
}

fn parse_args() -> Result<(PathBuf, PathBuf, i32)> {
    let mut args = env::args().skip(1);
    let snap = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let n_label_str = args.next().unwrap_or_else(|| usage());
    if args.next().is_some() {
        usage();
    }
    let n_label: i32 = n_label_str
        .parse()
        .with_context(|| format!("invalid n_label {n_label_str}"))?;
    if n_label != 8 && n_label != 10 {
        bail!("n_label must be 8 or 10");
    }
    Ok((snap, out_dir, n_label))
}

fn bucket_bytes(bucket: &Bucket) -> usize {
    bucket.rows_data.len() * std::mem::size_of::<i32>()
        + bucket.indptr.len() * std::mem::size_of::<i64>()
        + bucket.weights.len() * std::mem::size_of::<f64>()
        + bucket.key.len() * std::mem::size_of::<i32>()
}

fn main() -> Result<()> {
    runtime::configure_thread_pool();

    let (snap_path, out_dir, n_label) = parse_args()?;
    if !snap_path.exists() {
        bail!("snapshot {:?} does not exist", snap_path);
    }
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("create dir {}", out_dir.display()))?;

    eprintln!("[split] loading snapshot {}", snap_path.display());
    let snapshot = load_snapshot(&snap_path.to_string_lossy())?;
    eprintln!("[split] snapshot loaded (buckets={}, N={})", snapshot.buckets.len(), snapshot.n_total);

    let mut total_bytes = 0usize;
    for (idx, bucket) in snapshot.buckets.iter().enumerate() {
        let path = out_dir.join(format!("snapshot_bucket_n{}_{}.npz", n_label, idx));
        write_bucket_shard(&path, idx, bucket)?;
        let bytes = bucket_bytes(bucket);
        total_bytes += bytes;
        eprintln!(
            "[split] bucket {:5} | rows={:8} nnz={:10} key_len={:3} bytes≈{:.3} MB -> {}",
            idx,
            bucket.n_rows(),
            bucket.rows_data.len(),
            bucket.key.len(),
            (bytes as f64) / (1024.0 * 1024.0),
            path.display()
        );
    }
    eprintln!(
        "[split] buckets written to {} (count={}, est_mem={:.3} GB)",
        out_dir.display(),
        snapshot.buckets.len(),
        (total_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
    );

    let schedule = snapshot_to_schedule(&snapshot, n_label);
    let schedule_path = out_dir.join(format!("snapshot_schedule_n{}.json", n_label));
    write_schedule_meta(&schedule_path, &schedule)?;
    eprintln!(
        "[split] schedule saved to {} (tasks={}, buckets={})",
        schedule_path.display(),
        schedule.tasks.len(),
        schedule.buckets.len()
    );

    Ok(())
}
