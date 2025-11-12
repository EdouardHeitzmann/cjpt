use anyhow::{Context, Result};
use ndarray::Array1;
use ndarray_npy::{NpzReader, NpzWriter};
use std::fs::File;

use super::types::{Bucket, Snapshot};

fn read_i32<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<Array1<i32>> {
    let arr: Array1<i32> = npz
        .by_name(name)
        .with_context(|| format!("missing {}", name))?;
    Ok(arr)
}
fn read_i64<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<Array1<i64>> {
    let arr: Array1<i64> = npz
        .by_name(name)
        .with_context(|| format!("missing {}", name))?;
    Ok(arr)
}
fn read_f64<R: std::io::Read + std::io::Seek>(
    npz: &mut NpzReader<R>,
    name: &str,
) -> Result<Array1<f64>> {
    let arr: Array1<f64> = npz
        .by_name(name)
        .with_context(|| format!("missing {}", name))?;
    Ok(arr)
}

pub fn load_snapshot(path: &str) -> Result<Snapshot> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let mut npz = NpzReader::new(f).context("read npz")?;

    let n_total = read_i32(&mut npz, "meta_N.npy")?[0];
    let jbt_ref_pop = read_i32(&mut npz, "meta_jbt_ref_pop.npy")?.to_vec();

    let keys_indptr = read_i64(&mut npz, "meta_bucket_keys_indptr.npy")?;
    let num_buckets = if keys_indptr.len() == 0 {
        0
    } else {
        keys_indptr.len() - 1
    };

    let mut buckets = Vec::with_capacity(num_buckets);
    for b in 0..num_buckets {
        let rows_data = read_i32(&mut npz, &format!("b{}_rows_data.npy", b))?.to_vec();
        let indptr = read_i64(&mut npz, &format!("b{}_rows_indptr.npy", b))?.to_vec();
        let weights = read_f64(&mut npz, &format!("b{}_weights.npy", b))?.to_vec();
        let key = read_i32(&mut npz, &format!("b{}_key.npy", b))?.to_vec();
        buckets.push(Bucket {
            rows_data,
            indptr,
            weights,
            key,
        });
    }

    // compat tables (pop -> (key1, key2))
    let mut compat = std::collections::HashMap::new();
    let compat_pops = read_i32(&mut npz, "meta_compat_pops.npy")?;
    for p in compat_pops.iter() {
        let k1 = read_i32(&mut npz, &format!("compat_p{}_key1.npy", p))?.to_vec();
        let k2 = read_i32(&mut npz, &format!("compat_p{}_key2.npy", p))?.to_vec();
        compat.insert(*p, (k1, k2));
    }

    Ok(Snapshot {
        buckets,
        jbt_ref_pop,
        n_total,
        compat,
    })
}

pub fn save_snapshot(path: &str, snap: &Snapshot) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path))?;
    let mut npz = NpzWriter::new(f);

    npz.add_array("meta_N.npy", &Array1::from_vec(vec![snap.n_total]))?;
    npz.add_array(
        "meta_jbt_ref_pop.npy",
        &Array1::from_vec(snap.jbt_ref_pop.clone()),
    )?;

    let mut key_data: Vec<i32> = Vec::new();
    let mut key_indptr: Vec<i64> = Vec::with_capacity(snap.buckets.len() + 1);
    key_indptr.push(0);
    for bucket in &snap.buckets {
        key_data.extend(bucket.key.iter().copied());
        let last = *key_indptr.last().unwrap();
        key_indptr.push(last + bucket.key.len() as i64);
    }
    npz.add_array("meta_bucket_keys_data.npy", &Array1::from_vec(key_data))?;
    npz.add_array("meta_bucket_keys_indptr.npy", &Array1::from_vec(key_indptr))?;

    for (idx, bucket) in snap.buckets.iter().enumerate() {
        npz.add_array(
            &format!("b{}_rows_data.npy", idx),
            &Array1::from_vec(bucket.rows_data.clone()),
        )?;
        npz.add_array(
            &format!("b{}_rows_indptr.npy", idx),
            &Array1::from_vec(bucket.indptr.clone()),
        )?;
        npz.add_array(
            &format!("b{}_weights.npy", idx),
            &Array1::from_vec(bucket.weights.clone()),
        )?;
        npz.add_array(
            &format!("b{}_key.npy", idx),
            &Array1::from_vec(bucket.key.clone()),
        )?;
    }

    let mut compat_pops: Vec<i32> = snap.compat.keys().copied().collect();
    compat_pops.sort_unstable();
    npz.add_array(
        "meta_compat_pops.npy",
        &Array1::from_vec(compat_pops.clone()),
    )?;
    for p in compat_pops {
        if let Some((key1, key2)) = snap.compat.get(&p) {
            npz.add_array(
                &format!("compat_p{}_key1.npy", p),
                &Array1::from_vec(key1.clone()),
            )?;
            npz.add_array(
                &format!("compat_p{}_key2.npy", p),
                &Array1::from_vec(key2.clone()),
            )?;
        }
    }

    npz.finish()?;
    Ok(())
}
