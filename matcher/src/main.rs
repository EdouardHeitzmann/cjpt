use anyhow::{Context, Result, bail};
use std::env;
use std::path::{Path, PathBuf};

mod enumeration;
mod matching;
mod runtime;

enum RunMode {
    Enumerate {
        input: PathBuf,
        snapshot_out: PathBuf,
    },
    Resume {
        snapshot: PathBuf,
    },
}

fn usage() -> ! {
    eprintln!(
        "usage: matcher <inputs.npz> [snapshot_out.npz]\n       matcher --resume <snapshot.npz>"
    );
    std::process::exit(1);
}

fn default_snapshot_path(input: &Path) -> PathBuf {
    let parent = input
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("snapshot");
    parent.join(format!("{stem}_snapshot.npz"))
}

fn parse_args() -> Result<RunMode> {
    let mut args = env::args().skip(1);
    let first = args.next().unwrap_or_else(|| usage());
    if first == "--resume" {
        let snap = args.next().unwrap_or_else(|| usage());
        return Ok(RunMode::Resume {
            snapshot: PathBuf::from(snap),
        });
    }

    let input = PathBuf::from(first);
    if !input.exists() {
        bail!("input {:?} does not exist", input);
    }

    let snapshot_out = if let Some(explicit) = args.next() {
        PathBuf::from(explicit)
    } else if let Ok(from_env) = env::var("ENUM_SNAPSHOT_PATH") {
        PathBuf::from(from_env)
    } else {
        default_snapshot_path(&input)
    };

    Ok(RunMode::Enumerate {
        input,
        snapshot_out,
    })
}

fn main() -> Result<()> {
    runtime::configure_thread_pool();

    let mode = parse_args()?;
    let snapshot = match &mode {
        RunMode::Resume { snapshot } => {
            eprintln!("[resume] loading snapshot from {}", snapshot.display());
            let snap_path = snapshot.to_string_lossy().into_owned();
            matching::load_snapshot(&snap_path)?
        }
        RunMode::Enumerate {
            input,
            snapshot_out,
        } => {
            eprintln!("[enumerate] reading inputs from {}", input.display());
            let input_path = input.to_string_lossy().into_owned();
            let snap = enumeration::enumerate_to_snapshot_from_npz(&input_path)?;
            if let Some(parent) = snapshot_out.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("create dir {}", parent.display()))?;
                }
            }
            let snapshot_path = snapshot_out.to_string_lossy().into_owned();
            matching::save_snapshot(&snapshot_path, &snap)?;
            eprintln!("[enumerate] snapshot cached at {}", snapshot_out.display());
            snap
        }
    };

    let _ = matching::run_all_pairs_parallel(&snapshot, true);
    Ok(())
}
