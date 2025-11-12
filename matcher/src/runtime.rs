use rayon::ThreadPoolBuilder;
use std::sync::Once;

struct ThreadConfig {
    count: usize,
    source: String,
}

fn parse_env_threads(keys: &[&str]) -> Option<ThreadConfig> {
    for &key in keys {
        if let Ok(v) = std::env::var(key) {
            if let Ok(val) = v.parse::<usize>() {
                if val > 0 {
                    return Some(ThreadConfig {
                        count: val,
                        source: key.to_string(),
                    });
                }
            }
        }
    }
    None
}

fn detect_thread_config() -> ThreadConfig {
    const ENV_HINTS: [&str; 6] = [
        "MATCHER_THREADS",
        "RAYON_NUM_THREADS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "PBS_NP",
        "OMP_NUM_THREADS",
    ];

    if let Some(cfg) = parse_env_threads(&ENV_HINTS) {
        return cfg;
    }

    let fallback = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1);

    ThreadConfig {
        count: fallback,
        source: "available_parallelism".to_string(),
    }
}

pub fn configure_thread_pool() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let cfg = detect_thread_config();
        match ThreadPoolBuilder::new()
            .num_threads(cfg.count)
            .thread_name(|i| format!("matcher-worker-{i}"))
            .build_global()
        {
            Ok(_) => {
                eprintln!("[threads] rayon pool = {} threads (hint: {})", cfg.count, cfg.source);
            }
            Err(err) => {
                eprintln!("[threads] warn: failed to configure rayon pool ({err}); continuing with default");
            }
        }
    });
}
