use std::path::Path;

pub(crate) fn vortex_feature_store_disk_mb(root: &Path) -> u64 {
    regular_file_bytes_without_following_symlinks(root) / (1 << 20)
}

fn regular_file_bytes_without_following_symlinks(root: &Path) -> u64 {
    let mut bytes = 0_u64;
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if metadata.file_type().is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                bytes = bytes.saturating_add(metadata.len());
            }
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{regular_file_bytes_without_following_symlinks, vortex_feature_store_disk_mb};

    fn scratch(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos-feature-status-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn recursively_counts_regular_vortex_run_files() {
        let root = scratch("nested");
        fs::create_dir_all(root.join("run-a/chunks")).expect("create nested fixture");
        fs::write(root.join("run-a/data.vortex"), [1_u8, 2, 3]).expect("write Vortex fixture");
        fs::write(root.join("run-a/chunks/manifest"), [4_u8; 5]).expect("write control fixture");

        assert_eq!(regular_file_bytes_without_following_symlinks(&root), 8);

        fs::remove_dir_all(root).expect("remove nested fixture");
    }

    #[test]
    fn missing_feature_run_root_reports_zero() {
        let root = scratch("missing");
        assert_eq!(regular_file_bytes_without_following_symlinks(&root), 0);
        assert_eq!(vortex_feature_store_disk_mb(&root), 0);
    }
}
