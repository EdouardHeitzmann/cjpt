# cjpt
10x10 grid partition enumerator following a design inspired by Jamie Tucker-Foltz and Jeanne Clelland.

There are two main steps to this algorithm:
1) Enumerate all j-types for one half of the grid. The output of this step is a long list of vectors containing ID numbers of polyomino equivalence classes, as well as their corresponding weights, sorted into buckets according to their partial populations.
2) Match all compatible j-type buckets against each other, and accumulate the weight of compatible weights to produce a final number.

To determine which polyomino classes are compatible with which, as well as which polyomino placements are legal in step 1, we use some pre-computed data generated in python, which is stored in `/data`.

# Test run for n = 8:
- Clone this repo onto a directory in the hpc.
- Make sure you have a working cargo installation.
- Without setting any environment variables, execute `cargo run --release --bin matcher -- ../data/pre_ref_compat_inputs8.npz`
- Within a minute the script should print out the correct number (187,497,290,034).

# Running n = 10:
- Make sure you're ok with the os paths in the below commands (the first is the path where the output of step 1 will be saved).
```
export ENUM_MAX_RSS_GB=1028          # abort if RSS exceeds 1TB
export ENUM_SNAPSHOT_PATH=../data/cjpt10_snapshot.npz

cargo run --release --bin matcher -- ../data/pre_ref_compat_inputs10.npz
```
- If step 1 runs but step 2 times out (this would already be a huge win), we can resume step 2 from the cached results as follows:
`cargo run --release --bin matcher -- --resume ../data/cjpt10_snapshot.npz`

# Splitting a completed snapshot into shards:
Working in the root directory `matcher`, with a snapshot saved in `/data/pre_ref_compat_inputs10_snapshot.npz` (make sure the name of the snapshot is actually correct):
- Run `cargo run --release --bin snapshot_split -- ../data/pre_ref_compat_inputs10_snapshot.npz ../data/cjpt10_shards 10`; this will split the large saved file into many small buckets, and pre-compute a schedule of which buckets to match together.
- With this done, do `cargo run --release --bin snapshot_pair_run -- ../data/cjpt10_shards/snapshot_schedule_n10.json ../data/cjpt10_shards`; this will assign individual parallelized workers to load a bucket pair into RAM according to the schedule, perform the matching logic, accumulate the subtotal into the global total, and print out some profiling info.

I have no idea how many buckets there will be, or how large each will be! It's reasonable to expect that their footprint will be larger than the single-file 130GB snapshot, and that matching them will take longer than the original (>48h) matcher process (because each worker now has some loading/unloading to do).

# Running prebuilt executables (no Cargo)
If you have the following compiled binaries (in `matcher/target/release/`):
- `matcher` (enumerate + match, or resume from snapshot)
- `snapshot_split` (split a snapshot into per-bucket shards and a pairing schedule)
- `snapshot_pair_run` (match from shards using the saved schedule)

## To run N=10:
(adjust paths as needed)

- Fresh run N=10: `./matcher ../data/pre_ref_compat_inputs10.npz`
- Resume from full snapshot N=10: `./matcher -- --resume ../data/cjpt10_snapshot.npz`
- Split snapshot into shards N=10: `./snapshot_split ../data/cjpt10_snapshot.npz ../data/cjpt10_shards 10`
- Match from shards N=10: `./snapshot_pair_run ../data/cjpt10_shards/snapshot_schedule_n10.json ../data/cjpt10_shards`

## To test N=8:
- Fresh run N=8: `./matcher ../data/pre_ref_compat_inputs8.npz`
- Split an N=8 snapshot: `./snapshot_split ../data/cjpt8_snapshot.npz ../data/cjpt8_shards 8`
- Match shards N=8: `./snapshot_pair_run ../data/cjpt8_shards/snapshot_schedule_n8.json ../data/cjpt8_shards`

## Reordering a shard matching schedule
To create a new shard matching schedule:
- Build: `cargo build --release --bin schedule_reorder`
- Run: `./target/release/schedule_reorder <input_schedule.json> <output_schedule.json>`

The new schedule keeps the same buckets and metadata but sorts the tasks by descending pair cost = rows_left * rows_right.
