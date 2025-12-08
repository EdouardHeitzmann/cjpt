use anyhow::{Context, Result};
use std::cmp::Reverse;
use std::env;
use std::path::PathBuf;

use matcher::matching::{load_schedule_meta, write_schedule_meta, ScheduleMeta};

fn usage() -> ! {
    eprintln!("usage: schedule_reorder <input_schedule.json> <output_schedule.json>");
    std::process::exit(1);
}

fn parse_args() -> (PathBuf, PathBuf) {
    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let output = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    if args.next().is_some() {
        usage();
    }
    (input, output)
}

fn task_cost(meta: &ScheduleMeta, left: usize, right: usize) -> u128 {
    let rows1 = meta
        .buckets
        .get(left)
        .map(|b| b.rows as u128)
        .unwrap_or(0);
    let rows2 = meta
        .buckets
        .get(right)
        .map(|b| b.rows as u128)
        .unwrap_or(0);
    rows1.saturating_mul(rows2)
}

fn main() -> Result<()> {
    let (input_path, output_path) = parse_args();
    let mut schedule = load_schedule_meta(&input_path)
        .with_context(|| format!("load schedule {}", input_path.display()))?;

    let meta_snapshot = schedule.clone(); // small (only metadata), avoids borrow issues during sort
    schedule.tasks.sort_by_key(|t| {
        let left = t.left.min(meta_snapshot.buckets.len().saturating_sub(1));
        let right = t.right.min(meta_snapshot.buckets.len().saturating_sub(1));
        Reverse(task_cost(&meta_snapshot, left, right))
    });

    write_schedule_meta(&output_path, &schedule)
        .with_context(|| format!("write schedule {}", output_path.display()))?;
    eprintln!(
        "[reorder] wrote {} tasks to {}",
        schedule.tasks.len(),
        output_path.display()
    );
    Ok(())
}
