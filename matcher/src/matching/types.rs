use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Bucket {
    pub rows_data: Vec<i32>,
    pub indptr: Vec<i64>, // ok to switch to u32 later if you like
    pub weights: Vec<f64>,
    pub key: Vec<i32>, // empty [] means neutral ()
}
impl Bucket {
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }
    #[inline]
    pub fn row_slice(&self, r: usize) -> &[i32] {
        let lo = self.indptr[r] as usize;
        let hi = self.indptr[r + 1] as usize;
        &self.rows_data[lo..hi]
    }
}

#[derive(Debug)]
pub struct Snapshot {
    pub buckets: Vec<Bucket>,
    pub jbt_ref_pop: Vec<i32>,
    pub n_total: i32,
    pub compat: HashMap<i32, (Vec<i32>, Vec<i32>)>, // pop -> (key1, key2)
}

#[inline]
pub fn key_sorted_vec(key: &[i32]) -> Vec<i32> {
    let mut v = key.to_vec();
    v.sort();
    v
}
#[inline]
pub fn compat_key_sorted(key: &[i32], n_total: i32) -> Vec<i32> {
    let mut v: Vec<i32> = key.iter().map(|&p| n_total - p).collect();
    v.sort();
    v
}
