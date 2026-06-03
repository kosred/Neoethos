use anyhow::{Context, Result, bail};
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::dtype::NativePType;

// ─── Data-layer runtime overrides (config-consolidation S3-data) ───────────
// Config-driven replacement for the `NEOETHOS_BOT_NORMALIZE_FEATURES` /
// `NEOETHOS_BOT_REBUILD_STALE_HIGHER_TFS` env vars. The binary installs these
// from `Settings.models.data_runtime` once at startup (as plain bools, so this
// foundation crate keeps NOT depending on neoethos-core); the feature builder
// + resampler read the cached values instead of `std::env`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataRuntimeOverrides {
    pub normalize_features: bool,
    pub rebuild_stale_higher_tfs: bool,
}

impl Default for DataRuntimeOverrides {
    fn default() -> Self {
        Self {
            normalize_features: false,
            rebuild_stale_higher_tfs: false,
        }
    }
}

static DATA_RUNTIME_OVERRIDES: std::sync::OnceLock<DataRuntimeOverrides> =
    std::sync::OnceLock::new();

/// Install process-wide data-layer runtime overrides from config. The
/// binaries call this once at startup with `settings.models.data_runtime.*`.
/// Idempotent — the first install wins.
pub fn install_data_runtime_overrides(normalize_features: bool, rebuild_stale_higher_tfs: bool) {
    let _ = DATA_RUNTIME_OVERRIDES.set(DataRuntimeOverrides {
        normalize_features,
        rebuild_stale_higher_tfs,
    });
}

/// Current data-layer runtime overrides, or the deterministic defaults (both
/// OFF — matching the legacy env-unset behavior) when no install has happened.
pub fn current_data_runtime_overrides() -> DataRuntimeOverrides {
    DATA_RUNTIME_OVERRIDES.get().copied().unwrap_or_default()
}

#[cfg(test)]
mod data_runtime_overrides_tests {
    use super::*;

    #[test]
    fn data_runtime_overrides_default_is_off() {
        // Behavior-preservation: env-unset defaulted both knobs OFF.
        let d = DataRuntimeOverrides::default();
        assert!(!d.normalize_features);
        assert!(!d.rebuild_stale_higher_tfs);
    }
}

pub mod core;
pub mod test_fixtures;
// Re-export the canonical timeframe list so callers using neoethos-data
// can grab it without pulling in neoethos-core directly.
pub use crate::core::discover::{
    DataFileEntry, DataFormat, DatasetDiscovery, MAX_FILE_SIZE_BYTES, MAX_WALK_DEPTH, SkipReason,
    SkippedFile,
};
pub use crate::core::feature_registry::*;
pub use crate::core::features::*;
pub use crate::core::hpc_ta::*;
pub use crate::core::indicators::*;
pub use crate::core::loader::*;
pub use crate::core::parquet_migration::*;
pub use crate::core::quant_features::*;
pub use crate::core::regime_detection::*;
pub use crate::core::resample::*;
pub use crate::core::session_features::*;
pub use crate::core::slicing::slice_ohlcv;
pub use crate::core::smc::*;
pub use crate::core::timestamps::*;
pub use crate::core::vortex_io::*;
pub use neoethos_core::{CANONICAL_TIMEFRAMES, is_canonical_timeframe};

#[derive(Debug, Clone)]
pub struct Ohlcv {
    pub timestamp: Option<Vec<i64>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
}

impl Ohlcv {
    pub fn len(&self) -> usize {
        self.close.len()
    }
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct SymbolDataset {
    pub symbol: String,
    pub frames: HashMap<String, Ohlcv>,
}

impl SymbolDataset {
    pub fn timeframe(&self, tf: &str) -> Option<&Ohlcv> {
        self.frames.get(tf)
    }
    pub fn timeframes(&self) -> Vec<String> {
        let mut out: Vec<String> = self.frames.keys().cloned().collect();
        out.sort();
        out
    }
}

pub fn discover_symbols(root: impl AsRef<Path>) -> Result<Vec<String>> {
    let mut symbols = HashSet::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("symbol=") {
            symbols.insert(name.replace("symbol=", "").to_uppercase());
        }
    }
    let mut out: Vec<String> = symbols.into_iter().collect();
    out.sort();
    Ok(out)
}

/// Dedupe state for `discover_timeframes` warnings — fires the
/// "non-canonical timeframe folder" warning AT MOST ONCE per
/// `(symbol, timeframe)` per process lifetime.
///
/// Pre-fix: the warning ran inside the per-render scan loop, so a UI
/// frame rate of 60 Hz × 10 stale folders × N symbols produced
/// hundreds of identical log lines per second and drowned every other
/// trace. The dedupe set survives until the process exits — restart
/// the app to see the warning again (which is also what you'd want
/// since after a restart the stale folders may have been cleaned up).
static DISCOVER_TIMEFRAMES_WARNED: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(String, String)>>,
> = std::sync::OnceLock::new();

fn warned_once_for(symbol: &str, tf: &str) -> bool {
    let lock = DISCOVER_TIMEFRAMES_WARNED.get_or_init(|| std::sync::Mutex::new(HashSet::new()));
    let mut set = match lock.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    !set.insert((symbol.to_string(), tf.to_string()))
}

/// Cache TTL for `discover_timeframes`. Task #79 — the function was being
/// called once per render frame (60 Hz) from `market_chart_snapshot`, which
/// translated into 60 `read_dir` syscalls/sec per active symbol panel. Two
/// seconds is short enough that a new bootstrap shows up almost immediately
/// in the timeframe dropdown but long enough to fully eliminate the per-
/// frame syscall load (60 calls → ~0.5/sec).
const DISCOVER_TIMEFRAMES_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(2);

#[derive(Clone)]
struct DiscoverTimeframesCacheEntry {
    value: Vec<String>,
    captured_at: std::time::Instant,
}

static DISCOVER_TIMEFRAMES_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(PathBuf, String), DiscoverTimeframesCacheEntry>>,
> = std::sync::OnceLock::new();

fn discover_timeframes_cache_get(root: &Path, symbol: &str) -> Option<Vec<String>> {
    let lock = DISCOVER_TIMEFRAMES_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let guard = lock.lock().ok()?;
    let entry = guard.get(&(root.to_path_buf(), symbol.to_string()))?;
    if entry.captured_at.elapsed() < DISCOVER_TIMEFRAMES_CACHE_TTL {
        Some(entry.value.clone())
    } else {
        None
    }
}

fn discover_timeframes_cache_put(root: &Path, symbol: &str, value: Vec<String>) {
    let lock = DISCOVER_TIMEFRAMES_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(mut guard) = lock.lock() {
        guard.insert(
            (root.to_path_buf(), symbol.to_string()),
            DiscoverTimeframesCacheEntry {
                value,
                captured_at: std::time::Instant::now(),
            },
        );
    }
}

/// Integrity status of a single `(symbol, timeframe)` Vortex folder.
///
/// F-307 (2026-05-28): the loader used to accept any folder named
/// `timeframe=XX` that matched a canonical timeframe label, without
/// checking whether the data inside was actually usable. A half-finished
/// bootstrap leaves `data.vortex.partial` next to a stale `data.vortex`
/// from an earlier run, and the loader would happily feed that 12-month-
/// out-of-date blob into a 24-month sweep — producing NaN-laden features
/// and zero-trade GA candidates with no diagnostic.
///
/// State machine (all four states observed in production data dirs):
///   - `data.vortex` + `.complete` + no `.partial`  → `Complete`
///   - `data.vortex` + `.complete` + `.partial`     → `Complete` (new
///     bootstrap in progress; old data still usable) + WARN
///   - `data.vortex` + no `.complete` + `.partial`  → `StalePartial`  🚫
///   - `data.vortex` + no `.complete` + no `.partial` → `LegacyNoMarker`
///     (pre-marker era data; ACCEPT with one-time WARN so the operator
///     knows to re-bootstrap if they want full integrity)
///   - no `data.vortex`                            → `Missing` 🚫
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexIntegrity {
    /// Data is usable — `.complete` marker present.
    Complete,
    /// Data dir exists but `.partial` marker is there without `.complete`.
    /// A previous bootstrap aborted mid-write; the blob is stale and the
    /// loader MUST reject to avoid feeding half-data to backtests.
    StalePartial,
    /// `data.vortex` exists but neither `.complete` nor `.partial` is
    /// present. Pre-F-307 data files predate the marker convention.
    /// Accept-with-warn so existing operator data dirs keep working.
    LegacyNoMarker,
    /// No `data.vortex` in the folder — folder exists but is empty.
    Missing,
    /// `data.vortex` exists but is implausibly small
    /// (< [`VORTEX_MIN_PLAUSIBLE_BYTES`]) — a truncated or aborted write.
    /// Rejected even when a stale `.complete` marker is present: the
    /// 2026-06-01 corruption was an 84 KB EURUSD M1 / 9 KB EURUSD H1 that
    /// still carried `.complete` and was wrongly classified `Complete`.
    Truncated,
}

/// Minimum plausible byte size of a real `data.vortex` file. The
/// smallest legitimate frame seen in production (W1 / MN1, a few hundred
/// bars) is ~10 KB, so a 256-byte floor catches near-empty / garbage
/// stubs with a wide margin against false-positives. It does NOT catch a
/// file that is structurally valid but semantically short (too few bars
/// for its timeframe); that case is surfaced as a diagnostic warning in
/// [`load_symbol_timeframe`] via the `data.parquet` size ratio.
pub const VORTEX_MIN_PLAUSIBLE_BYTES: u64 = 256;

/// Inspect a `(symbol, timeframe)` directory and classify its load
/// readiness. See [`VortexIntegrity`] for the state machine.
pub fn vortex_integrity(dir: impl AsRef<Path>) -> VortexIntegrity {
    let dir = dir.as_ref();
    let vortex = dir.join("data.vortex");
    let Ok(meta) = std::fs::metadata(&vortex) else {
        return VortexIntegrity::Missing;
    };
    // Size-gate BEFORE trusting the markers: a truncated/aborted write can
    // leave a tiny `data.vortex` next to a stale `.complete` marker, which
    // the marker-only logic below would mis-classify as `Complete`.
    if meta.len() < VORTEX_MIN_PLAUSIBLE_BYTES {
        return VortexIntegrity::Truncated;
    }
    let complete = dir.join("data.vortex.complete").exists();
    let partial = dir.join("data.vortex.partial").exists();
    match (complete, partial) {
        (true, _) => VortexIntegrity::Complete,
        (false, true) => VortexIntegrity::StalePartial,
        (false, false) => VortexIntegrity::LegacyNoMarker,
    }
}

pub fn discover_timeframes(root: impl AsRef<Path>, symbol: &str) -> Result<Vec<String>> {
    let root_path = root.as_ref();
    // Task #79 — 60 Hz filesystem read elimination. The chart panel's
    // `market_chart_snapshot` calls us on every render frame; cache the
    // result for `DISCOVER_TIMEFRAMES_CACHE_TTL` so we don't hammer the
    // disk. The cache is keyed on (root, symbol) so different symbols /
    // different data dirs don't clobber each other.
    if let Some(cached) = discover_timeframes_cache_get(root_path, symbol) {
        return Ok(cached);
    }
    let path = PathBuf::from(root_path).join(format!("symbol={}", symbol));
    if !path.exists() {
        discover_timeframes_cache_put(root_path, symbol, Vec::new());
        return Ok(Vec::new());
    }
    let mut tfs = HashSet::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(raw) = name.strip_prefix("timeframe=") {
            let tf = raw.to_uppercase();
            // Note — gate against non-canonical timeframes
            // and path-traversal segments. Pre-fix, a stray
            // `timeframe=H2/` folder (cTrader has no H2 — see
            // `ctrader_api_reference.md` §4) was reported to the UI as
            // available, and a hostile/buggy `timeframe=../etc/passwd`
            // would have been accepted at this layer. We now compare
            // against `neoethos_core::CANONICAL_TIMEFRAMES`, which is the
            // single source of truth used by every other consumer (chart
            // panel, bootstrap, training).
            if neoethos_core::CANONICAL_TIMEFRAMES
                .iter()
                .any(|canonical| canonical.eq_ignore_ascii_case(&tf))
            {
                // F-307 (2026-05-28): integrity gate. The folder being
                // named correctly doesn't mean the data inside is
                // usable. `.partial` without `.complete` means a
                // half-finished bootstrap left a stale blob; the
                // discovery pipeline would feed that into the GA and
                // produce zero-trade candidates with no diagnostic
                // (root cause for the 4/4-candidate AUDUSD M15 funnel).
                match vortex_integrity(entry.path()) {
                    VortexIntegrity::Complete => {
                        tfs.insert(tf);
                    }
                    VortexIntegrity::LegacyNoMarker => {
                        // Accept (legacy data predates marker convention)
                        // but warn once so operators know to re-bootstrap
                        // for full integrity guarantees.
                        if !warned_once_for(symbol, &format!("{tf}/legacy")) {
                            tracing::warn!(
                                target: "neoethos_data::discover_timeframes",
                                symbol = symbol,
                                timeframe = %tf,
                                "legacy vortex file without .complete marker — accepted for backward-compat, re-bootstrap recommended"
                            );
                        }
                        tfs.insert(tf);
                    }
                    VortexIntegrity::StalePartial => {
                        // Reject loudly — the partial marker proves the
                        // existing data.vortex is from a previous
                        // never-completed bootstrap. Including this
                        // timeframe in the discovery pipeline would feed
                        // stale/half-data into the cost model.
                        if !warned_once_for(symbol, &format!("{tf}/stale")) {
                            tracing::warn!(
                                target: "neoethos_data::discover_timeframes",
                                symbol = symbol,
                                timeframe = %tf,
                                "REJECTED: data.vortex.partial present without .complete marker (half-finished bootstrap). Re-run --bootstrap-data for this timeframe."
                            );
                        }
                    }
                    VortexIntegrity::Truncated => {
                        // Truncated/aborted write (tiny data.vortex). Reject
                        // like StalePartial — the blob can't be trusted even
                        // if a stale .complete marker is present.
                        if !warned_once_for(symbol, &format!("{tf}/truncated")) {
                            tracing::warn!(
                                target: "neoethos_data::discover_timeframes",
                                symbol = symbol,
                                timeframe = %tf,
                                "REJECTED: data.vortex is implausibly small (truncated or aborted write). Re-run data bootstrap for this timeframe."
                            );
                        }
                    }
                    VortexIntegrity::Missing => {
                        // Folder exists but no data.vortex inside — silent
                        // skip; the bootstrap may be in its very first
                        // chunk and the file just isn't there yet.
                    }
                }
            } else if !warned_once_for(symbol, &tf) {
                // First time we see this (symbol, tf) since process start.
                // Subsequent calls with the same pair stay silent so the
                // UI's per-frame render loop doesn't flood the log.
                tracing::warn!(
                    target: "neoethos_data::discover_timeframes",
                    symbol = symbol,
                    timeframe = %tf,
                    "ignoring non-canonical timeframe folder; not in CANONICAL_TIMEFRAMES (warning is deduplicated; restart process to see again)"
                );
            }
        }
    }
    let mut out: Vec<String> = tfs.into_iter().collect();
    out.sort_by_key(|tf| parse_timeframe_to_minutes(tf).unwrap_or(999999));
    discover_timeframes_cache_put(root_path, symbol, out.clone());
    Ok(out)
}

pub fn load_symbol_timeframe(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
) -> Result<Ohlcv> {
    let path = symbol_timeframe_vortex_path(root, symbol, timeframe);
    if !path.exists() {
        bail!(
            "Vortex dataset not found for {} {} at {}. \
             Run Data Bootstrap (or `neoethos-cli import`) to download history first.",
            symbol, timeframe, path.display()
        );
    }
    // F-307 (2026-05-28): belt-and-braces integrity gate. Most callers
    // arrive here via `discover_timeframes` which already filters out
    // `StalePartial` folders, but direct callers (`load_symbol_dataset_with_timeframes`,
    // tail-readers, bridge endpoints) bypass that filter. Guard here too
    // so a half-finished bootstrap can't poison ANY load path.
    if let Some(parent) = path.parent() {
        match vortex_integrity(parent) {
            VortexIntegrity::Complete | VortexIntegrity::LegacyNoMarker => {
                // ok — proceed to load
            }
            VortexIntegrity::StalePartial => {
                bail!(
                    "vortex dataset {} {} REJECTED: data.vortex.partial present without \
                     .complete marker (half-finished bootstrap left stale data). \
                     Re-run data bootstrap for this timeframe.",
                    symbol,
                    timeframe
                );
            }
            VortexIntegrity::Truncated => {
                bail!(
                    "vortex dataset {} {} REJECTED: data.vortex is implausibly small \
                     (truncated or aborted write left a stale .complete marker). \
                     Re-run data bootstrap for this timeframe.",
                    symbol,
                    timeframe
                );
            }
            VortexIntegrity::Missing => {
                // Shouldn't happen — path.exists() already checked — but
                // guard against TOCTOU race where the file vanishes
                // between the .exists() call above and here.
                bail!("vortex dataset {} {} disappeared during load", symbol, timeframe);
            }
        }
    }

    // Diagnostic only (never rejects): a structurally-valid but
    // semantically-short data.vortex (too few bars for its timeframe) can
    // pass the integrity gate. When a data.parquet source sits beside it,
    // a vortex that is a tiny fraction of the parquet size is almost
    // certainly truncated — surface the diagnostic F-307 was meant to
    // provide, without deleting or rejecting anything.
    if let Some(parent) = path.parent() {
        let vortex_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let parquet_bytes = std::fs::metadata(parent.join("data.parquet"))
            .map(|m| m.len())
            .unwrap_or(0);
        if parquet_bytes > 0
            && vortex_bytes > 0
            && vortex_bytes.saturating_mul(50) < parquet_bytes
            && !warned_once_for(symbol, &format!("{timeframe}/ratio"))
        {
            tracing::warn!(
                target: "neoethos_data::load_symbol_timeframe",
                symbol = symbol,
                timeframe = timeframe,
                vortex_bytes,
                parquet_bytes,
                "data.vortex is <2% of its data.parquet source — likely a truncated/incomplete conversion. Re-run data bootstrap if candles look short."
            );
        }
    }

    load_vortex(path)
}

/// Load only the trailing `tail_n` rows for a symbol/timeframe.
///
/// #155: the full `load_symbol_timeframe` path materialises every row in
/// the Vortex file into an `Ohlcv` even when the caller only wants the
/// last 200 candles for a chart. For a multi-year M1 dataset that's
/// ~1 M rows × (5 × 8 bytes) ≈ 40 MB allocated per request, plus a
/// timestamp normalisation pass that walks every value.
///
/// This helper loads the same file but trims the in-memory `Ohlcv` down
/// to its last `tail_n` rows BEFORE returning it. The on-disk read still
/// materialises the whole stream — Vortex doesn't expose a cheap "skip
/// to row N" primitive at the layout level we use today — but the
/// caller-visible allocation and downstream iteration drops to
/// O(tail_n). When Vortex grows a true row-range scan API, this is the
/// function to upgrade; the surrounding contract stays the same.
pub fn load_symbol_timeframe_tail(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    tail_n: usize,
) -> Result<Ohlcv> {
    let mut ohlcv = load_symbol_timeframe(root, symbol, timeframe)?;
    let total = ohlcv.len();
    if tail_n >= total {
        return Ok(ohlcv);
    }
    let drop = total - tail_n;
    ohlcv.open.drain(..drop);
    ohlcv.high.drain(..drop);
    ohlcv.low.drain(..drop);
    ohlcv.close.drain(..drop);
    if let Some(ts) = ohlcv.timestamp.as_mut() {
        ts.drain(..drop);
    }
    if let Some(v) = ohlcv.volume.as_mut() {
        v.drain(..drop);
    }
    Ok(ohlcv)
}

pub fn load_symbol_dataset(root: impl AsRef<Path>, symbol: &str) -> Result<SymbolDataset> {
    let tfs = discover_timeframes(&root, symbol)?;
    let mut frames = HashMap::new();
    for tf in tfs {
        let ohlcv = load_symbol_timeframe(&root, symbol, &tf)
            .with_context(|| format!("failed to load dataset timeframe {} {}", symbol, tf))?;
        frames.insert(tf, ohlcv);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
    })
}

pub fn load_symbol_dataset_with_timeframes(
    root: impl AsRef<Path>,
    symbol: &str,
    target_tfs: &[&str],
) -> Result<SymbolDataset> {
    let mut frames = HashMap::new();
    for tf in target_tfs {
        let ohlcv = load_symbol_timeframe(&root, symbol, tf).with_context(|| {
            format!(
                "failed to load requested dataset timeframe {} {}",
                symbol, tf
            )
        })?;
        frames.insert(tf.to_string(), ohlcv);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
    })
}

pub fn symbol_timeframe_vortex_path(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
) -> PathBuf {
    PathBuf::from(root.as_ref())
        .join(format!("symbol={}", normalize_symbol_segment(symbol)))
        .join(format!(
            "timeframe={}",
            normalize_timeframe_segment(timeframe)
        ))
        .join("data.vortex")
}

pub fn write_symbol_timeframe_vortex(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    ohlcv: &Ohlcv,
) -> Result<PathBuf> {
    let path = symbol_timeframe_vortex_path(root, symbol, timeframe);
    write_ohlcv_vortex(&path, ohlcv)?;
    Ok(path)
}

pub fn write_ohlcv_vortex(path: impl AsRef<Path>, ohlcv: &Ohlcv) -> Result<()> {
    let normalized = normalize_ohlcv(ohlcv)?;
    let array = ohlcv_to_vortex_array(&normalized)?;
    write_vortex_array(path, array)
}

pub fn load_vortex(path: impl AsRef<Path>) -> Result<Ohlcv> {
    let array = read_vortex_array(path)?;
    vortex_array_to_ohlcv(array)
}

pub fn normalize_ohlcv(ohlcv: &Ohlcv) -> Result<Ohlcv> {
    let raw_timestamps = ohlcv
        .timestamp
        .as_ref()
        .context("OHLCV dataset has no timestamps")?;
    // Note — write-path timestamp unit normalisation.
    //
    // The READ path (`vortex_array_to_ohlcv`) calls
    // `normalize_timestamps_to_inferred_millis` to detect ns/μs/ms/s by
    // magnitude and convert to milliseconds. Until v0.4 the WRITE path
    // did NOT do this, so a caller passing nanoseconds (e.g. the
    // `BootstrapVortexWriter`, which writes `NormalizedBar.timestamp_ns`
    // directly) produced files that READ back as milliseconds — losing
    // the original unit and breaking every consumer that expected the
    // round-trip to be identity. Symmetric normalisation closes that
    // hole: ns / μs / s inputs are folded to ms here too, so the
    // canonical on-disk unit is always milliseconds. (See
    // `crates/neoethos-app/src/app_services/ctrader_bootstrap.rs` for the
    // coverage code that depends on this contract.)
    let normalized_ts = if raw_timestamps.is_empty() {
        raw_timestamps.clone()
    } else {
        crate::core::timestamps::normalize_timestamps_to_inferred_millis(raw_timestamps)
            .context("normalize OHLCV timestamps to milliseconds on write")?
    };
    let timestamps = &normalized_ts;
    let volume = ohlcv.volume.as_ref();
    let expected_len = timestamps.len();

    if ohlcv.open.len() != expected_len
        || ohlcv.high.len() != expected_len
        || ohlcv.low.len() != expected_len
        || ohlcv.close.len() != expected_len
        || volume.is_some_and(|values| values.len() != expected_len)
    {
        bail!(
            "OHLCV column length mismatch: timestamps={} open={} high={} low={} close={} — \
             the file may be corrupted; re-import it.",
            expected_len,
            ohlcv.open.len(),
            ohlcv.high.len(),
            ohlcv.low.len(),
            ohlcv.close.len()
        );
    }

    let mut rows = Vec::with_capacity(expected_len);
    for (idx, &timestamp) in timestamps.iter().enumerate().take(expected_len) {
        let volume_value = volume.and_then(|values| values.get(idx).copied());
        let row = OhlcvRow {
            timestamp,
            open: ohlcv.open[idx],
            high: ohlcv.high[idx],
            low: ohlcv.low[idx],
            close: ohlcv.close[idx],
            volume: volume_value,
        };
        validate_ohlcv_row(&row)?;
        rows.push(row);
    }

    rows.sort_by_key(|row| row.timestamp);
    rows.dedup_by_key(|row| row.timestamp);

    let has_volume = rows.iter().any(|row| row.volume.is_some());
    let mut out_timestamps = Vec::with_capacity(rows.len());
    let mut out_open = Vec::with_capacity(rows.len());
    let mut out_high = Vec::with_capacity(rows.len());
    let mut out_low = Vec::with_capacity(rows.len());
    let mut out_close = Vec::with_capacity(rows.len());
    let mut out_volume = has_volume.then(|| Vec::with_capacity(rows.len()));

    for row in rows {
        out_timestamps.push(row.timestamp);
        out_open.push(row.open);
        out_high.push(row.high);
        out_low.push(row.low);
        out_close.push(row.close);
        if let Some(values) = out_volume.as_mut() {
            values.push(row.volume.unwrap_or_default());
        }
    }

    Ok(Ohlcv {
        timestamp: Some(out_timestamps),
        open: out_open,
        high: out_high,
        low: out_low,
        close: out_close,
        volume: out_volume,
    })
}

fn ohlcv_to_vortex_array(ohlcv: &Ohlcv) -> Result<vortex_array::ArrayRef> {
    let timestamps = ohlcv
        .timestamp
        .as_ref()
        .context("OHLCV dataset has no timestamps")?;
    let mut fields = vec![
        (
            "timestamp",
            PrimitiveArray::from_iter(timestamps.iter().copied()).into_array(),
        ),
        (
            "open",
            PrimitiveArray::from_iter(ohlcv.open.iter().copied()).into_array(),
        ),
        (
            "high",
            PrimitiveArray::from_iter(ohlcv.high.iter().copied()).into_array(),
        ),
        (
            "low",
            PrimitiveArray::from_iter(ohlcv.low.iter().copied()).into_array(),
        ),
        (
            "close",
            PrimitiveArray::from_iter(ohlcv.close.iter().copied()).into_array(),
        ),
    ];

    if let Some(volume) = &ohlcv.volume {
        fields.push((
            "volume",
            PrimitiveArray::from_iter(volume.iter().copied()).into_array(),
        ));
    }

    Ok(StructArray::from_fields(&fields)
        .context("failed to build OHLCV vortex struct array")?
        .into_array())
}

fn vortex_array_to_ohlcv(array: vortex_array::ArrayRef) -> Result<Ohlcv> {
    let struct_array = array.to_struct();

    let raw_ts = extract_non_null_primitive_vec::<i64>(
        struct_array
            .unmasked_field_by_name("timestamp")
            .context("timestamp field missing")?,
        "timestamp",
    )?;
    // Normalize timestamps to milliseconds at the load boundary. Older
    // vortex files store nanoseconds (parquet/arrow default), while the
    // entire downstream pipeline (discovery prop-firm gate, eval day_key
    // aggregation, quality screen, regime labels) assumes milliseconds.
    // Without this conversion, every "day_key" comes out wrong and the
    // prop-firm window-pass gate degenerates to 5-second windows.
    let timestamp = if raw_ts.is_empty() {
        Some(raw_ts)
    } else {
        Some(
            crate::core::timestamps::normalize_timestamps_to_inferred_millis(&raw_ts)
                .context("normalize timestamps to milliseconds")?,
        )
    };

    let get_col = |names: &[&str]| -> Result<Vec<f64>> {
        for name in names {
            if let Some(field) = struct_array.unmasked_field_by_name_opt(name) {
                return extract_non_null_primitive_vec::<f64>(field, name);
            }
        }
        bail!(
            "Missing OHLCV column(s) {:?} — re-import the source; the file may use an older schema.",
            names
        )
    };

    let open = get_col(&["open", "o"])?;
    let high = get_col(&["high", "h"])?;
    let low = get_col(&["low", "l"])?;
    let close = get_col(&["close", "c"])?;
    let volume = get_col(&["volume", "vol", "v"]).ok();

    // Read-path structural check: a corrupt/truncated file can decode into
    // columns of different lengths, after which positional indexing
    // downstream (`ohlcv.open[i]`, chart rendering, feature extraction)
    // would panic out of bounds. The write path validates every row in
    // `normalize_ohlcv`; mirror a cheap length check here so a bad file
    // fails with a clear error instead of a later panic.
    let n = close.len();
    let ts_len = timestamp.as_ref().map_or(n, |t| t.len());
    if open.len() != n
        || high.len() != n
        || low.len() != n
        || ts_len != n
        || volume.as_ref().is_some_and(|v| v.len() != n)
    {
        bail!(
            "Vortex column length mismatch (timestamp={ts_len} open={} high={} low={} close={n}) \
             — the file is corrupt or truncated; re-import it.",
            open.len(),
            high.len(),
            low.len(),
        );
    }

    Ok(Ohlcv {
        timestamp,
        open,
        high,
        low,
        close,
        volume,
    })
}

fn extract_non_null_primitive_vec<T: NativePType>(
    array: &vortex_array::ArrayRef,
    label: &str,
) -> Result<Vec<T>> {
    if !array
        .all_valid()
        .with_context(|| format!("failed to inspect {label} validity"))?
    {
        bail!(
            "Column '{label}' has null values — the source data has gaps; \
             re-import after filling/trimming them."
        );
    }

    Ok(array.to_primitive().as_slice::<T>().to_vec())
}

fn validate_ohlcv_row(row: &OhlcvRow) -> Result<()> {
    if !row.open.is_finite()
        || !row.high.is_finite()
        || !row.low.is_finite()
        || !row.close.is_finite()
        || row.volume.is_some_and(|value| !value.is_finite())
    {
        bail!(
            "NaN/Inf in OHLCV at timestamp {} (open={} high={} low={} close={}) — \
             re-import and verify the price data is clean.",
            row.timestamp, row.open, row.high, row.low, row.close
        );
    }
    if row.high < row.low
        || row.open < row.low
        || row.open > row.high
        || row.close < row.low
        || row.close > row.high
    {
        bail!(
            "Invalid OHLC row at timestamp {} (open={} high={} low={} close={}) — \
             source has bad candles; re-import or trim them.",
            row.timestamp, row.open, row.high, row.low, row.close
        );
    }
    if row.volume.is_some_and(|value| value < 0.0) {
        bail!("negative volume detected");
    }
    Ok(())
}

fn normalize_symbol_segment(raw: &str) -> String {
    raw.trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

fn normalize_timeframe_segment(raw: &str) -> String {
    raw.trim().to_ascii_uppercase()
}

#[derive(Debug, Clone, Copy)]
struct OhlcvRow {
    timestamp: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: Option<f64>,
}

pub fn compute_hpc_features(ohlcv: &Ohlcv) -> Result<FeatureFrame> {
    compute_hpc_feature_frame(ohlcv, FeatureProfile::Standard)
}

pub fn compute_hpc_feature_frame(ohlcv: &Ohlcv, _profile: FeatureProfile) -> Result<FeatureFrame> {
    let mut names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();

    for (name, col) in compute_smc_feature_columns(ohlcv) {
        names.push(name);
        columns.push(col);
    }

    for (name, col) in compute_classic_ta_columns(ohlcv) {
        names.push(name);
        columns.push(col);
    }

    for (name, col) in compute_quant_feature_columns(ohlcv) {
        names.push(name);
        columns.push(col);
    }

    for (name, col) in compute_session_feature_columns(ohlcv) {
        names.push(name);
        columns.push(col);
    }

    for (name, col) in compute_regime_feature_columns(ohlcv) {
        names.push(name);
        columns.push(col);
    }

    let n_rows = ohlcv.len();
    let n_cols = columns.len();
    let mut data = Array2::zeros((n_rows, n_cols));
    // Note — explicit f64 → f32 narrowing at the feature-cube
    // boundary. The downstream consumers (neoethos-models tree models, the
    // genetic backtest kernel via cubecl) all expect f32 to halve memory
    // (~5 GB of features for a 5-year EURUSD M1 cube doubles to 10 GB if
    // stored as f64) and to map cleanly onto GPU tensor cores. The loss
    // is bounded by the source feature engineering: indicators sit in
    // [-1, 1] after normalisation, and absolute values rarely exceed
    // 1e6 (price-scaled). f32's mantissa (24 bits) preserves ~7
    // significant decimals, well above the noise floor of any FX
    // feature we emit. Acceptable trade-off — DO NOT promote to f64
    // without auditing the cube memory budget and the cubecl kernel
    // signatures.
    //
    // (We do NOT warn on truncation here because the trade-off was made
    // intentionally — flooding logs with "f64 → f32 narrowing at
    // n_rows*n_cols cells" would drown out real diagnostics. A unit
    // test that asserts feature magnitudes stay below f32::MAX would be
    // the right regression guard; tracked under follow-up audit.)
    for (c, col) in columns.iter().enumerate() {
        for (r, &val) in col.iter().enumerate() {
            data[(r, c)] = val as f32;
        }
    }

    // HARD FAIL: a FeatureFrame without timestamps cannot be joined with
    // labels, so a silent empty-Vec fallback masks an upstream loader bug.
    let timestamps = ohlcv
        .timestamp
        .clone()
        .ok_or_else(|| anyhow::anyhow!("compute_hpc_feature_frame: OHLCV is missing timestamps"))?;
    Ok(FeatureFrame {
        timestamps,
        names,
        data,
    })
}

pub fn prepare_multitimeframe_features(
    ds: &SymbolDataset,
    base_tf: &str,
    higher_tfs: &[&str],
    cache: Option<&FeatureCache>,
) -> Result<FeatureFrame> {
    let opts = FeatureBuildOptions {
        higher_tfs: higher_tfs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    prepare_multitimeframe_features_with_options(ds, base_tf, &opts, cache)
}

pub fn prepare_multitimeframe_features_with_options(
    ds: &SymbolDataset,
    base_tf: &str,
    opts: &FeatureBuildOptions,
    _cache: Option<&FeatureCache>,
) -> Result<FeatureFrame> {
    let base_ohlcv = ds
        .frames
        .get(base_tf)
        .ok_or_else(|| anyhow::anyhow!(
            "Base timeframe '{}' is missing from dataset '{}' — resample it first.",
            base_tf, ds.symbol
        ))?;
    let base_ns = base_ohlcv
        .timestamp
        .as_ref()
        .context("base has no timestamps")?;

    let mut all_names = Vec::new();
    let mut all_data_parts: Vec<Array2<f32>> = Vec::new();

    let base_feats = compute_hpc_feature_frame(base_ohlcv, opts.profile)?;
    all_names.extend(base_feats.names.iter().map(|n| {
        if opts.prefix_base_features {
            format!("{}_{}", base_tf, n)
        } else {
            n.clone()
        }
    }));
    all_data_parts.push(base_feats.data);

    for h_tf in &opts.higher_tfs {
        if h_tf == base_tf {
            continue;
        }
        if let Some(h_ohlcv) = ds.frames.get(h_tf) {
            let h_feats = compute_hpc_feature_frame(h_ohlcv, opts.profile)?;
            let h_ns = h_ohlcv
                .timestamp
                .as_ref()
                .context("higher tf has no timestamps")?;
            // F-308 fix (2026-05-28): cap forward-fill at 2× the
            // higher-TF period. Stale higher-TF data (last bar weeks
            // before base last bar) would otherwise propagate as a
            // frozen-constant column for the entire held-out window
            // — indicators on constants produce constants → GA sees
            // zero or look-alike signals → `ranked=N, post_passes_filter=0`
            // funnel with no diagnostic. Beyond `max_age`, rows
            // become NaN and the downstream feature-cube summary's
            // NaN counter flags them.
            //
            // 2× period is the lenient bound (one missed bar is
            // normal; two missed bars means the higher-TF source
            // has stalled). Failure to parse the TF label leaves
            // max_age None (legacy unbounded behaviour) — safer
            // default for hand-rolled non-canonical TF strings.
            let max_age_ns = parse_timeframe_to_minutes(h_tf)
                .ok()
                .filter(|m| *m > 0)
                .map(|m| (m as i64).saturating_mul(60).saturating_mul(1_000_000_000).saturating_mul(2));
            // Surface stale-higher-TF condition before the align call
            // so operators see a clear log line even when the data
            // looks fine downstream.
            let base_last = base_ns.last().copied().unwrap_or(0);
            let h_last = h_ns.last().copied().unwrap_or(0);
            if base_last > 0 && h_last > 0 && base_last > h_last {
                if let Some(max_age) = max_age_ns {
                    let lag_ns = base_last - h_last;
                    if lag_ns > max_age {
                        tracing::warn!(
                            target: "neoethos_data::prepare_multitimeframe_features",
                            base_tf = base_tf,
                            higher_tf = h_tf,
                            lag_seconds = lag_ns / 1_000_000_000,
                            max_age_seconds = max_age / 1_000_000_000,
                            "higher-TF last bar is older than 2× period — feature columns past max_age will be NaN. Re-run --bootstrap-data for this (symbol, timeframe) to refresh."
                        );
                    }
                }
            }
            let aligned = align_features_by_ns(base_ns, h_ns, &h_feats.data, true, max_age_ns);
            all_names.extend(h_feats.names.iter().map(|n| format!("{}_{}", h_tf, n)));
            all_data_parts.push(aligned);
        }
    }

    let total_cols = all_data_parts.iter().map(|p| p.ncols()).sum();
    let mut merged = Array2::zeros((base_ns.len(), total_cols));
    let mut curr_col = 0;
    for part in all_data_parts {
        let ncols = part.ncols();
        merged
            .slice_mut(ndarray::s![.., curr_col..curr_col + ncols])
            .assign(&part);
        curr_col += ncols;
    }

    // Per-column robust z-score (median + MAD * 1.4826) with NaN→0
    // and clip to ±10. Opt-in via `NEOETHOS_BOT_NORMALIZE_FEATURES=1`.
    //
    // Why opt-in: normalization is correct architecture (puts every
    // column on the same scale, kills NaN propagation, fixes the
    // EURJPY/XAUUSD empty-portfolio bug at the root) but the GA's
    // `random_coarse_threshold = [0.15..0.55]` is calibrated for the
    // un-normalized magnitude regime. Enabling normalization without
    // re-calibrating thresholds breaks discovery for symbols that
    // currently work (EURUSD/GBPUSD/AUDUSD). Threshold re-calibration
    // is a follow-up; until then operators opt in per-symbol when
    // they want to attack the JPY/XAU portfolio gap.
    if current_data_runtime_overrides().normalize_features {
        let _norm_stats = crate::core::normalization::normalize_feature_matrix(&mut merged);
    }

    Ok(FeatureFrame {
        timestamps: base_ns.clone(),
        names: all_names,
        data: merged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_root(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_data_{}_{}_{}",
            test_name,
            std::process::id(),
            nonce
        ))
    }

    fn write_valid_ohlcv_vortex(path: &Path) -> Result<()> {
        // 64 rows so the resulting file is comfortably above
        // VORTEX_MIN_PLAUSIBLE_BYTES. The integrity tests assert marker
        // classification, not size, but the size gate runs first — keep
        // their fixtures clear of it.
        let n = 64usize;
        let base_ms = 1_700_000_000_000_i64;
        let mut timestamp = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        for i in 0..n {
            let base = 1.0 + (i as f64) * 0.001;
            timestamp.push(base_ms + (i as i64) * 60_000);
            open.push(base);
            high.push(base + 0.002);
            low.push(base - 0.002);
            close.push(base + 0.001);
        }
        write_ohlcv_vortex(
            path,
            &Ohlcv {
                timestamp: Some(timestamp),
                open,
                high,
                low,
                close,
                volume: None,
            },
        )
    }

    #[test]
    fn normalize_ohlcv_sorts_deduplicates_and_validates_rows() -> Result<()> {
        // Note: `normalize_ohlcv` now folds timestamps to the
        // canonical milliseconds unit at the write boundary (mirrors the
        // read-side `vortex_array_to_ohlcv` call). Tiny integers like
        // `[3, 1, 1, 2]` are inferred as Seconds by magnitude and multiplied
        // by 1000 → `[1000, 2000, 3000]` ms after sort+dedup.
        let normalized = normalize_ohlcv(&Ohlcv {
            timestamp: Some(vec![3, 1, 1, 2]),
            open: vec![1.3, 1.1, 9.9, 1.2],
            high: vec![1.4, 1.2, 9.9, 1.3],
            low: vec![1.2, 1.0, 9.9, 1.1],
            close: vec![1.35, 1.15, 9.9, 1.25],
            volume: Some(vec![2.0, 1.0, 9.9, 1.5]),
        })?;

        assert_eq!(normalized.timestamp, Some(vec![1000, 2000, 3000]));
        assert_eq!(normalized.open, vec![1.1, 1.2, 1.3]);
        assert_eq!(normalized.volume, Some(vec![1.0, 1.5, 2.0]));
        Ok(())
    }

    #[test]
    fn load_symbol_dataset_rejects_unreadable_discovered_timeframe() -> Result<()> {
        let root = unique_temp_root("unreadable_discovered_timeframe");
        let m1_dir = root.join("symbol=EURUSD").join("timeframe=M1");
        let m5_dir = root.join("symbol=EURUSD").join("timeframe=M5");
        fs::create_dir_all(&m1_dir)?;
        fs::create_dir_all(&m5_dir)?;

        write_valid_ohlcv_vortex(&m1_dir.join("data.vortex"))?;
        // >= VORTEX_MIN_PLAUSIBLE_BYTES + a .complete marker so the
        // integrity gate passes and the failure comes from the byte-level
        // Vortex parser rejecting non-Vortex content (the test's intent),
        // not from the size/marker gate.
        fs::write(m5_dir.join("data.vortex"), vec![0u8; 1024])?;
        fs::write(m5_dir.join("data.vortex.complete"), b"")?;

        let err = load_symbol_dataset(&root, "EURUSD")
            .expect_err("discovered unreadable timeframe must fail the dataset load");
        assert!(
            err.to_string().contains("M5") || err.to_string().contains("vortex"),
            "unexpected error: {err}"
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn load_symbol_dataset_with_timeframes_rejects_requested_timeframe_failure() -> Result<()> {
        let root = unique_temp_root("requested_timeframe_failure");
        let m1_dir = root.join("symbol=EURUSD").join("timeframe=M1");
        let m5_dir = root.join("symbol=EURUSD").join("timeframe=M5");
        fs::create_dir_all(&m1_dir)?;
        fs::create_dir_all(&m5_dir)?;

        write_valid_ohlcv_vortex(&m1_dir.join("data.vortex"))?;
        // >= VORTEX_MIN_PLAUSIBLE_BYTES + a .complete marker so the
        // integrity gate passes and the failure comes from the byte-level
        // Vortex parser rejecting non-Vortex content (the test's intent),
        // not from the size/marker gate.
        fs::write(m5_dir.join("data.vortex"), vec![0u8; 1024])?;
        fs::write(m5_dir.join("data.vortex.complete"), b"")?;

        let err = load_symbol_dataset_with_timeframes(&root, "EURUSD", &["M1", "M5"])
            .expect_err("requested unreadable timeframe must fail the dataset load");
        assert!(
            err.to_string().contains("M5") || err.to_string().contains("vortex"),
            "unexpected error: {err}"
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    // ─── F-307 vortex integrity gate (2026-05-28) ─────────────────────

    fn make_tf_dir(root: &Path, symbol: &str, tf: &str) -> Result<PathBuf> {
        let dir = root.join(format!("symbol={symbol}")).join(format!("timeframe={tf}"));
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    #[test]
    fn vortex_integrity_missing_when_no_data_vortex() -> Result<()> {
        let root = unique_temp_root("integ_missing");
        let dir = make_tf_dir(&root, "TEST", "M1")?;
        assert_eq!(vortex_integrity(&dir), VortexIntegrity::Missing);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn vortex_integrity_complete_when_marker_present() -> Result<()> {
        let root = unique_temp_root("integ_complete");
        let dir = make_tf_dir(&root, "TEST", "M1")?;
        write_valid_ohlcv_vortex(&dir.join("data.vortex"))?;
        fs::write(dir.join("data.vortex.complete"), b"")?;
        assert_eq!(vortex_integrity(&dir), VortexIntegrity::Complete);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn vortex_integrity_complete_dominates_partial() -> Result<()> {
        // New bootstrap in progress next to old-but-complete data —
        // operator should still see Complete (existing data is usable).
        let root = unique_temp_root("integ_complete_with_partial");
        let dir = make_tf_dir(&root, "TEST", "M1")?;
        write_valid_ohlcv_vortex(&dir.join("data.vortex"))?;
        fs::write(dir.join("data.vortex.complete"), b"")?;
        fs::write(dir.join("data.vortex.partial"), b"")?;
        assert_eq!(vortex_integrity(&dir), VortexIntegrity::Complete);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn vortex_integrity_stale_partial_without_complete() -> Result<()> {
        // The AUDUSD M15 production failure mode — partial marker exists
        // but no complete marker. Loader MUST reject.
        let root = unique_temp_root("integ_stale_partial");
        let dir = make_tf_dir(&root, "TEST", "M1")?;
        write_valid_ohlcv_vortex(&dir.join("data.vortex"))?;
        fs::write(dir.join("data.vortex.partial"), b"")?;
        assert_eq!(vortex_integrity(&dir), VortexIntegrity::StalePartial);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn vortex_integrity_legacy_no_marker() -> Result<()> {
        // Pre-marker era data: vortex file but no markers. Accept-with-warn.
        let root = unique_temp_root("integ_legacy");
        let dir = make_tf_dir(&root, "TEST", "M1")?;
        write_valid_ohlcv_vortex(&dir.join("data.vortex"))?;
        assert_eq!(vortex_integrity(&dir), VortexIntegrity::LegacyNoMarker);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn vortex_integrity_truncated_when_below_size_floor() -> Result<()> {
        // A truncated/aborted write leaves a tiny data.vortex; even WITH a
        // stale .complete marker it must classify as Truncated, not
        // Complete (the 2026-06-01 corruption mode).
        let root = unique_temp_root("integ_truncated");
        let dir = make_tf_dir(&root, "TEST", "M1")?;
        fs::write(dir.join("data.vortex"), vec![0u8; 64])?; // < VORTEX_MIN_PLAUSIBLE_BYTES
        fs::write(dir.join("data.vortex.complete"), b"")?;
        assert_eq!(vortex_integrity(&dir), VortexIntegrity::Truncated);
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn discover_timeframes_rejects_stale_partial_tf() -> Result<()> {
        // Mixed dataset: M1 complete, M5 stale-partial, M15 legacy.
        // Expect M1 + M15 to surface; M5 rejected.
        let root = unique_temp_root("discover_stale");
        let symbol = "AUDUSD";

        let m1_dir = make_tf_dir(&root, symbol, "M1")?;
        write_valid_ohlcv_vortex(&m1_dir.join("data.vortex"))?;
        fs::write(m1_dir.join("data.vortex.complete"), b"")?;

        let m5_dir = make_tf_dir(&root, symbol, "M5")?;
        write_valid_ohlcv_vortex(&m5_dir.join("data.vortex"))?;
        fs::write(m5_dir.join("data.vortex.partial"), b"")?;
        // No .complete — STALE

        let m15_dir = make_tf_dir(&root, symbol, "M15")?;
        write_valid_ohlcv_vortex(&m15_dir.join("data.vortex"))?;
        // No markers — LEGACY

        let tfs = discover_timeframes(&root, symbol)?;
        assert!(tfs.contains(&"M1".to_string()), "M1 missing: {:?}", tfs);
        assert!(
            !tfs.contains(&"M5".to_string()),
            "M5 (stale partial) should be rejected: {:?}",
            tfs
        );
        assert!(
            tfs.contains(&"M15".to_string()),
            "M15 (legacy no marker) should be accepted: {:?}",
            tfs
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn load_symbol_timeframe_rejects_stale_partial() -> Result<()> {
        // Belt-and-braces guard: even when discover_timeframes is
        // bypassed, the loader must reject stale-partial data.
        let root = unique_temp_root("load_stale");
        let symbol = "AUDUSD";
        let dir = make_tf_dir(&root, symbol, "M15")?;
        write_valid_ohlcv_vortex(&dir.join("data.vortex"))?;
        fs::write(dir.join("data.vortex.partial"), b"")?;
        // No .complete

        let err = load_symbol_timeframe(&root, symbol, "M15")
            .expect_err("load must reject stale-partial vortex");
        let msg = err.to_string();
        assert!(
            msg.contains("partial") || msg.contains("REJECTED"),
            "unexpected error: {msg}"
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
