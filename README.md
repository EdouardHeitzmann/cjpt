# cjpt
10x10 grid partition enumerator following a design inspired by Jamie Tucker-Foltz and Jeanne Clelland.

There are two main steps to this algorithm:
1) Enumerate all j-types for one half of the grid. The output of this step is a long list of vectors containing ID numbers of polyomino equivalence classes, as well as their corresponding weights, sorted into buckets according to their partial populations.
2) Match all compatible j-type buckets against each other, and accumulate the weight of compatible weights to produce a final number.

To determine which polyomino classes are compatible with which, as well as which polyomino placements are legal in step 1, we use some pre-computed data generated in python, which is stored in `/data`.

# Test run for n = 8:
- Clone this repo onto a directory in the hpc.
- Make sure you have a working cargo installation.
- Without setting any environment variables, execute `cargo run --release -- ../data/pre_ref_compat_inputs8.npz`
- Within a minute the script should print out the correct number (187,497,290,034).

# Running n = 10:
- Make sure you're ok with the os paths in the below commands (the first is the path where the output of step 1 will be saved).
```
export ENUM_MAX_RSS_GB=1028          # abort if RSS exceeds 1TB
export ENUM_SNAPSHOT_PATH=../data/cjpt10_snapshot.npz

cargo run --release -- ../data/pre_ref_compat_inputs10.npz
```
- If step 1 runs but step 2 times out (this would already be a huge win), we can resume step 2 from the cached results as follows:
`cargo run --release -- --resume ../data/cjpt10_snapshot.npz`
