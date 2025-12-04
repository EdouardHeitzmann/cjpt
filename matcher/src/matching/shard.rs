use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use super::driver::build_pair_tasks;
use super::io::read_bucket_from_npz;
use super::types::{Bucket, PairTask, Snapshot};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketMeta {
    pub id: usize,
    pub key: Vec<i32>,
    pub rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleMeta {
    pub n_label: i32,
    pub n_total: i32,
    pub buckets: Vec<BucketMeta>,
    pub tasks: Vec<PairTask>,
    pub jbt_ref_pop: Vec<i32>,
    pub compat: HashMap<i32, (Vec<i32>, Vec<i32>)>,
}

pub fn write_bucket_shard(path: &Path, bucket_id: usize, bucket: &Bucket) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut npz = ndarray_npy::NpzWriter::new(f);
    npz.add_array(
        "meta_bucket_id.npy",
        &ndarray::Array1::from_vec(vec![bucket_id as i32]),
    )?;
    npz.add_array(
        "rows_data.npy",
        &ndarray::Array1::from_vec(bucket.rows_data.clone()),
    )?;
    npz.add_array(
        "rows_indptr.npy",
        &ndarray::Array1::from_vec(bucket.indptr.clone()),
    )?;
    npz.add_array(
        "weights.npy",
        &ndarray::Array1::from_vec(bucket.weights.clone()),
    )?;
    npz.add_array("key.npy", &ndarray::Array1::from_vec(bucket.key.clone()))?;
    npz.finish()?;
    Ok(())
}

pub fn load_bucket_shard(path: &Path) -> Result<Bucket> {
    read_bucket_from_npz(path)
}

pub fn write_schedule_meta(path: &Path, meta: &ScheduleMeta) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut w = BufWriter::new(f);
    serde_json::to_writer_pretty(&mut w, meta).context("write schedule json")?;
    Ok(())
}

pub fn load_schedule_meta(path: &Path) -> Result<ScheduleMeta> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let r = BufReader::new(f);
    let meta: ScheduleMeta = serde_json::from_reader(r).context("parse schedule json")?;
    Ok(meta)
}

pub fn snapshot_to_schedule(snapshot: &Snapshot, n_label: i32) -> ScheduleMeta {
    let tasks = build_pair_tasks(snapshot);
    let buckets: Vec<BucketMeta> = snapshot
        .buckets
        .iter()
        .enumerate()
        .map(|(id, b)| BucketMeta {
            id,
            key: b.key.clone(),
            rows: b.n_rows(),
        })
        .collect();
    ScheduleMeta {
        n_label,
        n_total: snapshot.n_total,
        buckets,
        tasks,
        jbt_ref_pop: snapshot.jbt_ref_pop.clone(),
        compat: snapshot.compat.clone(),
    }
}
