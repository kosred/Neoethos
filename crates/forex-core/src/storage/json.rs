use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonBackupWriteConfig {
    pub artifact_label: &'static str,
    pub temp_extension: &'static str,
    pub backup_extension: &'static str,
}

pub fn write_json_atomic<T: Serialize + ?Sized>(path: impl AsRef<Path>, value: &T) -> Result<()> {
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

pub fn write_json_with_backup<T: Serialize + ?Sized>(
    path: impl AsRef<Path>,
    value: &T,
    config: JsonBackupWriteConfig,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "create {} artifact directory {}",
                config.artifact_label,
                parent.display()
            )
        })?;
    }

    let temp_path = path.with_extension(config.temp_extension);
    let backup_path = path.with_extension(config.backup_extension);
    let payload = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize {}", config.artifact_label))?;
    if temp_path.exists() {
        fs::remove_file(&temp_path).with_context(|| {
            format!(
                "remove stale staged {} {}",
                config.artifact_label,
                temp_path.display()
            )
        })?;
    }
    if backup_path.exists() {
        fs::remove_file(&backup_path).with_context(|| {
            format!(
                "remove stale backup {} {}",
                config.artifact_label,
                backup_path.display()
            )
        })?;
    }
    {
        let mut temp = File::create(&temp_path).with_context(|| {
            format!(
                "create staged {} {}",
                config.artifact_label,
                temp_path.display()
            )
        })?;
        temp.write_all(&payload).with_context(|| {
            format!(
                "write staged {} to {}",
                config.artifact_label,
                temp_path.display()
            )
        })?;
        temp.sync_all().with_context(|| {
            format!(
                "fsync staged {} {}",
                config.artifact_label,
                temp_path.display()
            )
        })?;
    }
    if path.exists() {
        fs::rename(path, &backup_path).with_context(|| {
            format!(
                "backup current {} {}",
                config.artifact_label,
                path.display()
            )
        })?;
    }
    if let Err(error) = fs::rename(&temp_path, path) {
        if backup_path.exists() {
            let _ = fs::rename(&backup_path, path);
        } else if temp_path.exists() {
            let _ = fs::remove_file(&temp_path);
        }
        anyhow::bail!(
            "write {} to {} failed: {}",
            config.artifact_label,
            path.display(),
            error
        );
    }
    if backup_path.exists() {
        fs::remove_file(&backup_path).with_context(|| {
            format!(
                "remove backup {} {}",
                config.artifact_label,
                backup_path.display()
            )
        })?;
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
    Ok(format!("fnv64:{:016x}", crate::utils::fnv1a64(&bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct SampleArtifact {
        name: String,
        value: usize,
    }

    fn unique_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("current time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forex_core_json_io_{name}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    #[test]
    fn atomic_json_write_round_trips_and_uses_hidden_temp_path() {
        let dir = unique_test_dir("atomic");
        let path = dir.join("artifact.json");
        let artifact = SampleArtifact {
            name: "alpha".to_string(),
            value: 7,
        };

        write_json_atomic(&path, &artifact).expect("write atomic json");
        let reloaded: SampleArtifact = read_json(&path, "sample").expect("read atomic json");

        assert_eq!(reloaded, artifact);
        assert_eq!(
            temporary_path(&path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .starts_with(".artifact.json.tmp-"),
            true
        );
        std::fs::remove_dir_all(&dir).expect("cleanup atomic json dir");
    }

    #[test]
    fn backup_json_write_replaces_existing_file_and_removes_staging_files() {
        let dir = unique_test_dir("backup");
        let path = dir.join("artifact.json");
        let first = SampleArtifact {
            name: "first".to_string(),
            value: 1,
        };
        let second = SampleArtifact {
            name: "second".to_string(),
            value: 2,
        };
        let config = JsonBackupWriteConfig {
            artifact_label: "sample artifact",
            temp_extension: "tmp_sample",
            backup_extension: "bak_sample",
        };

        write_json_with_backup(&path, &first, config).expect("write first payload");
        write_json_with_backup(&path, &second, config).expect("replace payload");
        let reloaded: SampleArtifact = read_json(&path, "sample").expect("read replaced json");

        assert_eq!(reloaded, second);
        assert!(!path.with_extension("tmp_sample").exists());
        assert!(!path.with_extension("bak_sample").exists());
        std::fs::remove_dir_all(&dir).expect("cleanup backup json dir");
    }

    #[test]
    fn stable_json_hash_uses_canonical_fnv64_prefix() {
        let artifact = SampleArtifact {
            name: "alpha".to_string(),
            value: 7,
        };

        let first = stable_json_hash(&artifact).expect("hash first");
        let second = stable_json_hash(&artifact).expect("hash second");

        assert_eq!(first, second);
        assert!(first.starts_with("fnv64:"));
    }
}
