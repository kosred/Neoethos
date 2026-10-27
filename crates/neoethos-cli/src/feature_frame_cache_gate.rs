//! Receipt sidecar for the research-only feature-frame cache.
//!
//! A cached feature frame is correct only if rebuilding it from the same
//! inputs would produce the same bytes. The build key already covers every
//! input digest, but a stale entry (older binary, corrupted payload that still
//! parses) must fail loudly instead of silently feeding a different frame into
//! research. The canonical-research consumer therefore seals each stored entry
//! with the exact CanonicalSearchInputReceiptV2 computed from the freshly
//! built frame, and a cache hit must reproduce that receipt bit for bit.

use anyhow::{Context, Result, ensure};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::path::Path;

/// On a cache miss: seal the receipt next to the stored frame entry. On a
/// cache hit: load the sealed receipt and require it to equal the receipt
/// recomputed from the loaded frame. Any mismatch (or missing sidecar) is an
/// error telling the operator to clear the cache directory - never a silent
/// fallback to a different research input.
pub fn verify_or_seal_receipt_sidecar<T>(
    cache_dir: &Path,
    key: &str,
    cache_hit: bool,
    receipt: &T,
) -> Result<()>
where
    T: Serialize + DeserializeOwned + PartialEq,
{
    let sidecar = cache_dir.join(key).join("receipt.json");
    if cache_hit {
        let bytes = std::fs::read(&sidecar).with_context(|| {
            format!("cached feature frame `{key}` has no sealed receipt sidecar; clear the cache directory `{}` and rerun", cache_dir.display())
        })?;
        let stored: T = serde_json::from_slice(&bytes)
            .with_context(|| format!("cached feature frame `{key}` receipt sidecar is malformed"))?;
        ensure!(
            stored == *receipt,
            "cached feature frame `{key}` failed receipt verification: the rebuilt frame does not match the sealed receipt; clear the cache directory `{}` and rerun",
            cache_dir.display()
        );
    } else {
        let bytes = serde_json::to_vec_pretty(receipt)
            .context("serialize feature-frame cache receipt sidecar")?;
        let tmp = sidecar.with_file_name("receipt.json.tmp");
        std::fs::write(&tmp, &bytes).context("write feature-frame cache receipt sidecar")?;
        std::fs::rename(&tmp, &sidecar).context("seal feature-frame cache receipt sidecar")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct SealProbe {
        a: u32,
        b: String,
    }

    fn fresh_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "neoethos-cli-ffc-gate-{tag}-{}-{:?}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(dir.join("k1")).expect("entry dir");
        dir
    }

    #[test]
    fn miss_seals_then_hit_with_same_receipt_passes() {
        let dir = fresh_dir("seal");
        let probe = SealProbe { a: 7, b: "x".to_owned() };
        verify_or_seal_receipt_sidecar(&dir, "k1", false, &probe).expect("seal");
        assert!(dir.join("k1").join("receipt.json").is_file(), "sidecar written");
        verify_or_seal_receipt_sidecar(&dir, "k1", true, &probe).expect("verify");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hit_with_different_receipt_fails() {
        let dir = fresh_dir("mismatch");
        let sealed = SealProbe { a: 7, b: "x".to_owned() };
        let rebuilt = SealProbe { a: 8, b: "x".to_owned() };
        verify_or_seal_receipt_sidecar(&dir, "k1", false, &sealed).expect("seal");
        let error = verify_or_seal_receipt_sidecar(&dir, "k1", true, &rebuilt).unwrap_err();
        assert!(
            error.to_string().contains("receipt verification"),
            "error must name the verification failure: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hit_without_sidecar_fails() {
        let dir = fresh_dir("missing");
        let probe = SealProbe { a: 7, b: "x".to_owned() };
        let error = verify_or_seal_receipt_sidecar(&dir, "k1", true, &probe).unwrap_err();
        assert!(
            error.to_string().contains("no sealed receipt sidecar"),
            "error must name the missing sidecar: {error}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}