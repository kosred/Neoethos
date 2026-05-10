use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001B3;

pub fn write_json_atomic<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create artifact directory {}", parent.display()))?;
    let tmp_path = temporary_path(path);
    let json = serde_json::to_vec_pretty(value).context("serialize artifact")?;
    {
        let mut tmp = File::create(&tmp_path)
            .with_context(|| format!("create temp artifact {}", tmp_path.display()))?;
        tmp.write_all(&json)
            .with_context(|| format!("write temp artifact {}", tmp_path.display()))?;
        tmp.write_all(b"\n").context("terminate json artifact")?;
        tmp.sync_all()
            .with_context(|| format!("fsync temp artifact {}", tmp_path.display()))?;
    }
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "atomically rename {} to {}",
            tmp_path.display(),
            path.display()
        )
    })?;
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>, artifact_label: &str) -> Result<T> {
    let path = path.as_ref();
    let payload = fs::read(path)
        .with_context(|| format!("read {artifact_label} artifact {}", path.display()))?;
    serde_json::from_slice(&payload)
        .with_context(|| format!("parse {artifact_label} artifact {}", path.display()))
}

pub fn temporary_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact.json");
    path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id()))
}

pub fn stable_json_hash<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).context("serialize value for stable hash")?;
    Ok(format!("fnv64:{:016x}", fnv1a64(&bytes)))
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
