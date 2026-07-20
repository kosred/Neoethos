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
pub use crate::core::footprint_features::*;
pub use crate::core::resample::*;
pub use crate::core::session_features::*;
pub use crate::core::slicing::{slice_ohlcv, slice_ohlcv_by_date_range_ms};
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

    // Perf (2026-07-02, operator: ">24h on dense TFs, one core pinned"): the
    // five indicator families used to run one-after-another and four of them
    // are internally single-threaded, so on M1/M3 (millions of rows) this
    // phase held ONE core for hours. Run the families CONCURRENTLY via a
    // rayon::join nest (classic_ta's internal par_iter work-steals within the
    // same pool). Memory: peak is unchanged — all five column sets existed
    // simultaneously before this change too (they were pushed into `columns`).
    //
    // PARITY-CRITICAL: column ORDER feeds effective_feature_names and every
    // discovery artifact — it must stay EXACTLY smc → classic → quant →
    // session → regime. rayon::join returns results by POSITION (not by
    // completion), so the chained collection below is deterministic.
    let (smc, (classic, (quant, (session, (regime, footprint))))) = rayon::join(
        || compute_smc_feature_columns(ohlcv),
        || {
            rayon::join(
                || compute_classic_ta_columns(ohlcv),
                || {
                    rayon::join(
                        || compute_quant_feature_columns(ohlcv),
                        || {
                            rayon::join(
                                || compute_session_feature_columns(ohlcv),
                                || {
                                    rayon::join(
                                        || compute_regime_feature_columns(ohlcv),
                                        // Footprint family (2026-07-02): bar-level
                                        // effort-vs-result order-flow proxies.
                                        // APPENDED LAST so every pre-existing
                                        // portfolio's column order is unchanged
                                        // (projection matches by name).
                                        || compute_footprint_feature_columns(ohlcv),
                                    )
                                },
                            )
                        },
                    )
                },
            )
        },
    );

    for (name, col) in smc
        .into_iter()
        .chain(classic)
        .chain(quant)
        .chain(session)
        .chain(regime)
        .chain(footprint)
    {
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
    //
    // Perf (2026-07-02): parallel per-column copy — on an M1 cube this loop
    // is ~1.8e9 scalar writes; column-parallel cuts it to seconds. Each rayon
    // worker owns one output COLUMN (disjoint writes; identical values/order
    // to the old serial loop — pure data-parallel copy, no parity impact).
    {
        use ndarray::Axis;
        use rayon::prelude::*;
        data.axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(columns.par_iter())
            .for_each(|(mut out_col, col)| {
                for (r, &val) in col.iter().enumerate() {
                    out_col[r] = val as f32;
                }
            });
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
        data: crate::core::features::FeatureData::InMemory(data),
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

/// Temp path for a discovery feature-store, unique per (symbol, base_tf,
/// process). The store auto-deletes when its `FeatureFrame` drops, so this
/// path collides only if a prior run crashed mid-build (truncate-on-create
/// reclaims it).
fn discovery_feature_store_path(symbol: &str, base_tf: &str) -> std::path::PathBuf {
    let sanitize = |s: &str| -> String {
        s.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    };
    let mut dir = std::env::temp_dir();
    dir.push("neoethos_feature_store");
    let _ = std::fs::create_dir_all(&dir);
    // Sweep ORPHAN feature stores left by FORCE-KILLED prior runs (Drop's
    // delete_on_drop only fires on graceful exit; a kill leaks the file — and
    // the multi-TF M3 cube is ~12 GB each, so a few killed runs fill the disk).
    // Best-effort: on Windows a file still mmap'd by a LIVE process refuses
    // deletion, so this only removes genuine orphans from dead processes.
    sweep_orphan_feature_stores(&dir);
    dir.push(format!(
        "{}_{}_{}.fstore",
        sanitize(symbol),
        sanitize(base_tf),
        std::process::id()
    ));
    dir
}

/// Delete `.fstore` files orphaned by force-killed discovery runs. Runs once per
/// process (before this process creates any of its own stores). On Windows a
/// live process's mapped file refuses `remove_file`, so live runs are protected.
fn sweep_orphan_feature_stores(dir: &std::path::Path) {
    use std::sync::OnceLock;
    static SWEPT: OnceLock<()> = OnceLock::new();
    if SWEPT.set(()).is_err() {
        return; // already swept this process
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut freed = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("fstore") {
            continue;
        }
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        if std::fs::remove_file(&path).is_ok() {
            freed += size;
        }
    }
    if freed > 0 {
        tracing::info!(
            target: "neoethos_data::feature_store",
            freed_mb = freed / (1024 * 1024),
            "swept orphan feature stores from force-killed runs"
        );
    }
}

/// Stream one in-RAM `[samples × cols]` feature block to the mmap store, column
/// by column (each column becomes one feature-major row), normalising each
/// series in place when enabled, then letting the caller free the block. Keeps
/// peak RAM at one timeframe's block rather than the whole multi-TF cube.
fn append_feature_block(
    writer: &mut crate::core::feature_store::FeatureStoreWriter,
    block: &Array2<f32>,
    normalize: bool,
) -> Result<()> {
    for c in 0..block.ncols() {
        let mut series: Vec<f32> = block.column(c).to_vec();
        if normalize {
            // D09: fit robust-z stats on the training prefix, not the full
            // series, so the OOS tail is not leaked into and the values stay
            // stable under future appends.
            let fit_rows = crate::core::normalization::norm_fit_rows(series.len());
            crate::core::normalization::normalize_feature_series_in_place(&mut series, fit_rows);
        }
        writer.append_feature(&series)?;
    }
    Ok(())
}

/// Compute + align ONE higher-timeframe feature block onto the base grid.
/// Returns `None` when the higher TF equals the base or is absent from the
/// dataset. Shared by the in-RAM and streaming (mmap) cube builders so the
/// F-308 stale-data handling and the alignment live in a single place.
fn compute_aligned_higher_block(
    ds: &SymbolDataset,
    base_tf: &str,
    base_ns: &[i64],
    h_tf: &str,
    profile: FeatureProfile,
) -> Result<Option<(Vec<String>, Array2<f32>)>> {
    if h_tf == base_tf {
        return Ok(None);
    }
    let Some(h_ohlcv) = ds.frames.get(h_tf) else {
        return Ok(None);
    };
    let h_feats = compute_hpc_feature_frame(h_ohlcv, profile)?;
    let h_ns = h_ohlcv
        .timestamp
        .as_ref()
        .context("higher tf has no timestamps")?;
    // Audit D02 (2026-07-13): higher-TF bars are OPEN-stamped, so a bar's
    // final feature values only exist at stamp + period (its close). The
    // alignment must therefore lag availability by ONE FULL PERIOD —
    // otherwise every base bar inside a still-forming higher-TF bucket
    // reads that bucket's FINAL values (up to a period of lookahead: 4h on
    // H4, a day on D1), and live — which sees a partial forming bar —
    // silently diverges from the backtest. Refuse to align without a
    // resolvable period: silent lookahead is worse than a hard error.
    //
    // Audit D01 (same day): the period must be expressed in the SAME UNIT
    // as the timestamps — and this codebase has carried both (datasets are
    // normalized to MILLISECONDS at the load boundary and the live path
    // builds ms, while this file's older `*_ns` constants assumed
    // nanoseconds, which is why the F-308 staleness cap never fired in
    // production). Instead of guessing the unit, derive the period from
    // the BASE GRID itself: one higher-TF period = median base spacing ×
    // (higher minutes / base minutes). Unit-agnostic — correct for ms, ns,
    // and synthetic test grids alike.
    let h_mins = parse_timeframe_to_minutes(h_tf)
        .ok()
        .filter(|m| *m > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "cannot resolve the bar period for higher timeframe '{h_tf}' — refusing \
                 to align its features without close-availability (that would reintroduce \
                 up to one period of lookahead into the feature cube)"
            )
        })?;
    let base_mins = parse_timeframe_to_minutes(base_tf)
        .ok()
        .filter(|m| *m > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "cannot resolve the bar period for base timeframe '{base_tf}' — needed to \
                 express the higher-TF availability lag in the timestamp grid's own unit"
            )
        })?;
    let median_base_step = {
        let mut steps: Vec<i64> = base_ns.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0).collect();
        if steps.is_empty() {
            anyhow::bail!(
                "base timeframe '{base_tf}' has no positive timestamp spacing — cannot \
                 derive the higher-TF availability lag (dataset looks degenerate)"
            );
        }
        let mid = steps.len() / 2;
        steps.select_nth_unstable(mid);
        steps[mid]
    };
    let period_units = median_base_step
        .saturating_mul(h_mins as i64)
        .checked_div(base_mins as i64)
        .filter(|p| *p > 0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "derived a non-positive higher-TF period for '{h_tf}' over base '{base_tf}' \
                 (median base step {median_base_step}) — refusing to align"
            )
        })?;
    // F-308: cap forward-fill at 2× the higher-TF period so stale higher-TF
    // data becomes NaN (flagged downstream) instead of a frozen-constant
    // column that would feed the GA zero / look-alike signals. Measured
    // from the bar's CLOSE (availability) since D02; expressed in the
    // grid's own unit since D01 (the old ×1e9 cap could never fire on the
    // production ms grids).
    let max_age_units = Some(period_units.saturating_mul(2));
    let base_last = base_ns.last().copied().unwrap_or(0);
    let h_last = h_ns.last().copied().unwrap_or(0);
    if base_last > 0 && h_last > 0 && base_last > h_last {
        if let Some(max_age) = max_age_units {
            let staleness = base_last - h_last;
            if staleness > max_age {
                tracing::warn!(
                    target: "neoethos_data::prepare_multitimeframe_features",
                    base_tf = base_tf,
                    higher_tf = h_tf,
                    staleness_grid_units = staleness,
                    max_age_grid_units = max_age,
                    "higher-TF last bar is older than 2× period — feature columns past max_age will be NaN. Re-run --bootstrap-data for this (symbol, timeframe) to refresh."
                );
            }
        }
    }
    let h_names: Vec<String> = h_feats
        .names
        .iter()
        .map(|n| format!("{}_{}", h_tf, n))
        .collect();
    let h_block = match h_feats.data {
        crate::core::features::FeatureData::InMemory(a) => a,
        crate::core::features::FeatureData::Mmap(_)
        | crate::core::features::FeatureData::MmapWindow { .. } => {
            anyhow::bail!("compute_hpc_feature_frame must return an in-memory frame")
        }
    };
    let aligned = align_features_by_ns(base_ns, h_ns, &h_block, true, max_age_units, period_units);
    Ok(Some((h_names, aligned)))
}

/// Normalise each column of an in-RAM feature block in place (robust z-score),
/// matching the per-series normalisation `append_feature_block` applies on the
/// streaming path so the two cube layouts stay identical.
/// Normalize every column in place. Generic over the storage so it works on an
/// owned block AND on a mutable VIEW into the final cube — the in-RAM assembly
/// normalizes each timeframe's columns after placing them, avoiding a second
/// copy of the block.
fn normalize_block_columns<S>(block: &mut ndarray::ArrayBase<S, ndarray::Ix2>)
where
    S: ndarray::DataMut<Elem = f32>,
{
    let fit_rows = crate::core::normalization::norm_fit_rows(block.nrows());
    for mut col in block.columns_mut() {
        let mut series: Vec<f32> = col.to_vec();
        // D09: fit stats on the training prefix (see append_feature_block).
        crate::core::normalization::normalize_feature_series_in_place(&mut series, fit_rows);
        for (dst, src) in col.iter_mut().zip(series) {
            *dst = src;
        }
    }
}

/// Decide whether the multi-TF feature cube is assembled in RAM (fast, no disk
/// write/cleanup) or streamed to a disk-mmap store (the NEVER-OOM fallback).
///
/// `NEOETHOS_FEATURE_CUBE_MODE = ram | disk | auto` forces the choice.
///
/// The `auto` requirement is derived from the assembly's ACTUAL peak, which is
/// why it changed on 2026-07-20. The old assembly built every per-TF block and
/// then `concatenate`d them, so both the blocks and the result were live at
/// once — a ~2× peak, demanded as 2.5× for margin. That was an artifact of the
/// implementation, not a property of the data: a 10.5 GB cube "needed" 28 GB
/// and went to disk on a 32 GB machine with 22 GB free, paying a 10 GB write
/// plus mmap page-fault traffic for the whole GA.
///
/// [`try_assemble_cube_in_ram`] now allocates the cube ONCE and fills each
/// timeframe's columns in place, freeing each block immediately — so the peak
/// is the cube plus ONE timeframe block (~1.1× on a 10-TF build). We require
/// 1.5× plus the same 2 GB floor for the OS and the GA's working buffers. A
/// failed free-RAM probe (0) still takes the safe disk path, and any surprise
/// during assembly falls back to disk rather than growing memory.
fn should_build_cube_in_ram(cube_bytes: u64) -> bool {
    match std::env::var("NEOETHOS_FEATURE_CUBE_MODE")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("ram") => return true,
        Some("disk") => return false,
        _ => {}
    }
    let available = neoethos_core::available_memory_bytes();
    if available == 0 {
        return false;
    }
    let needed = (cube_bytes as f64) * 1.5 + 2.0e9;
    needed < available as f64
}

/// Assemble the multi-timeframe cube directly in RAM, allocating it ONCE and
/// writing each timeframe into its own column range.
///
/// Peak memory is the cube plus a single timeframe block, because each block is
/// dropped as soon as its columns are copied (the previous `concatenate`
/// approach held every block AND the result simultaneously — a 2× peak).
///
/// Returns `Ok(None)` — never a partial cube — if the real column widths do not
/// match the estimate the RAM/disk decision was made from. The caller then
/// takes the streaming disk path, so a width surprise costs time but never
/// loses features and never grows past the budget that was approved.
#[allow(clippy::too_many_arguments)]
fn try_assemble_cube_in_ram(
    ds: &SymbolDataset,
    base_tf: &str,
    base_ns: &[i64],
    opts: &FeatureBuildOptions,
    active_higher: &[String],
    base_block: &Array2<f32>,
    base_names: &[String],
    normalize: bool,
    n_samples: usize,
    est_features: usize,
) -> Result<Option<FeatureFrame>> {
    use ndarray::s;

    let base_cols = base_block.ncols();
    if base_cols > est_features {
        return Ok(None);
    }
    let mut cube = Array2::<f32>::zeros((n_samples, est_features));
    let mut names: Vec<String> = Vec::with_capacity(est_features);
    let mut col = 0usize;

    // Base timeframe: copy in, then normalize the destination view in place.
    {
        let mut dst = cube.slice_mut(s![.., col..col + base_cols]);
        dst.assign(base_block);
        if normalize {
            normalize_block_columns(&mut dst);
        }
    }
    names.extend(base_names.iter().cloned());
    col += base_cols;

    for h_tf in active_higher {
        let Some((h_names, aligned)) =
            compute_aligned_higher_block(ds, base_tf, base_ns, h_tf, opts.profile)?
        else {
            // Every entry in `active_higher` was filtered to exist in the
            // dataset, so this is the width-surprise case: bail to disk.
            return Ok(None);
        };
        let w = aligned.ncols();
        if col + w > est_features {
            return Ok(None);
        }
        cube.slice_mut(s![.., col..col + w]).assign(&aligned);
        // Free this timeframe BEFORE the next one is computed — this is what
        // keeps the peak at cube + one block.
        drop(aligned);
        if normalize {
            let mut dst = cube.slice_mut(s![.., col..col + w]);
            normalize_block_columns(&mut dst);
        }
        names.extend(h_names);
        col += w;
    }

    if col != est_features || names.len() != est_features {
        // Fewer columns than the allocation: returning the cube as-is would
        // append all-zero phantom features. Shrinking here would need a second
        // full copy (the very peak this function exists to avoid), so hand the
        // build to the streaming path instead.
        return Ok(None);
    }

    Ok(Some(FeatureFrame {
        timestamps: base_ns.to_vec(),
        names,
        data: crate::core::features::FeatureData::InMemory(cube),
    }))
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

    let n_samples = base_ns.len();

    // ── RAM-aware multi-resolution feature cube ───────────────────────────
    //
    // The dense cube is `n_samples × features × 4 B` (~13 GB for full M1). When
    // the machine has enough free RAM we assemble it directly in memory (no
    // disk write, nothing to clean up); otherwise we stream each TF into a
    // feature-major mmap store so peak RAM stays at ONE timeframe's compute —
    // the NEVER-OOM fallback that lets discovery run on any hardware. The
    // sink is chosen per (symbol, TF) from the cube estimate vs free RAM.
    let normalize = current_data_runtime_overrides().normalize_features;

    // Base timeframe — native resolution, no alignment. Computed up front
    // (needed by both sink paths) and used to size the cube.
    let base_feats = compute_hpc_feature_frame(base_ohlcv, opts.profile)?;
    let base_names: Vec<String> = base_feats
        .names
        .iter()
        .map(|n| {
            if opts.prefix_base_features {
                format!("{}_{}", base_tf, n)
            } else {
                n.clone()
            }
        })
        .collect();
    let base_block = match base_feats.data {
        crate::core::features::FeatureData::InMemory(a) => a,
        crate::core::features::FeatureData::Mmap(_)
        | crate::core::features::FeatureData::MmapWindow { .. } => {
            anyhow::bail!("compute_hpc_feature_frame must return an in-memory frame")
        }
    };

    // Active higher TFs (present in the dataset, not the base).
    let active_higher: Vec<String> = opts
        .higher_tfs
        .iter()
        .filter(|h| h.as_str() != base_tf && ds.frames.contains_key(h.as_str()))
        .cloned()
        .collect();

    // Per-TF feature count = base block width (same registry for every TF);
    // estimate the whole cube and pick RAM vs disk-mmap accordingly.
    let per_tf = base_block.ncols().max(1);
    let est_features = per_tf.saturating_mul(1 + active_higher.len());
    let cube_bytes = (n_samples as u64)
        .saturating_mul(est_features as u64)
        .saturating_mul(4);
    let in_ram = should_build_cube_in_ram(cube_bytes);
    tracing::info!(
        target: "neoethos_data::prepare_multitimeframe_features",
        symbol = %ds.symbol,
        base_tf = base_tf,
        rows = n_samples,
        per_tf_features = per_tf,
        timeframes = 1 + active_higher.len(),
        est_cube_gb = format!("{:.2}", cube_bytes as f64 / 1e9),
        available_ram_gb = format!("{:.1}", neoethos_core::available_memory_bytes() as f64 / 1e9),
        sink = if in_ram { "RAM (no disk write)" } else { "disk mmap" },
        "feature cube build plan"
    );

    if in_ram {
        // Allocate the cube ONCE and fill it timeframe by timeframe, freeing
        // each block as it lands (see `try_assemble_cube_in_ram`). `base_block`
        // stays borrowed so the disk path below can still use it if the
        // assembly bails.
        if let Some(frame) = try_assemble_cube_in_ram(
            ds,
            base_tf,
            base_ns,
            opts,
            &active_higher,
            &base_block,
            &base_names,
            normalize,
            n_samples,
            est_features,
        )? {
            return Ok(frame);
        }
        tracing::warn!(
            target: "neoethos_data::prepare_multitimeframe_features",
            symbol = %ds.symbol,
            base_tf = base_tf,
            est_features,
            "in-RAM cube assembly bailed (per-timeframe feature widths did not match \
             the estimate the RAM budget was approved from) — streaming to the disk \
             store instead. No features are lost; this is slower, please report it."
        );
    }

    // ── Streaming disk-mmap path (NEVER-OOM fallback for large cubes) ──────
    let store_path = discovery_feature_store_path(&ds.symbol, base_tf);
    let mut writer =
        crate::core::feature_store::FeatureStoreWriter::create(&store_path, n_samples)?;
    let mut all_names = base_names;
    append_feature_block(&mut writer, &base_block, normalize)?;
    drop(base_block);

    for h_tf in &active_higher {
        if let Some((h_names, aligned)) =
            compute_aligned_higher_block(ds, base_tf, base_ns, h_tf, opts.profile)?
        {
            all_names.extend(h_names);
            append_feature_block(&mut writer, &aligned, normalize)?;
        }
    }

    // Normalisation (opt-in via `NEOETHOS_BOT_NORMALIZE_FEATURES=1`) is applied
    // per-series during the streaming append above. Robust z-score is
    // per-column independent, so normalising each series before it is written
    // to the store is identical to normalising the assembled matrix (see
    // `normalize_feature_series_in_place`). Why opt-in: it is correct
    // architecture (puts every column on the same scale, kills NaN
    // propagation, fixes the EURJPY/XAUUSD empty-portfolio bug at the root) but
    // the GA's `random_coarse_threshold = [0.15..0.55]` is calibrated for the
    // un-normalized magnitude regime; enabling it without re-calibrating
    // thresholds breaks discovery for symbols that currently work
    // (EURUSD/GBPUSD/AUDUSD). Threshold re-calibration is a follow-up.
    let n_features = writer.finish()?;
    let store = crate::core::feature_store::FeatureStore::open(
        &store_path,
        n_features,
        n_samples,
        /* delete_on_drop */ true,
    )?;

    Ok(FeatureFrame {
        timestamps: base_ns.clone(),
        names: all_names,
        data: crate::core::features::FeatureData::Mmap(std::sync::Arc::new(store)),
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

#[cfg(test)]
mod cube_assembly_tests {
    use super::*;

    /// Build a small multi-timeframe dataset: a base grid plus one higher
    /// timeframe resampled from it, so `prepare_multitimeframe_features` has
    /// real work to align.
    fn tiny_dataset(n: usize) -> SymbolDataset {
        let base_ms = 1_700_000_000_000_i64;
        let mut timestamp = Vec::with_capacity(n);
        let (mut open, mut high, mut low, mut close, mut volume) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            // Deterministic, non-degenerate series (constant columns would
            // normalize to zero everywhere and weaken the comparison).
            let t = i as f64;
            let px = 1.10 + (t * 0.7).sin() * 0.01 + t * 1e-5;
            timestamp.push(base_ms + (i as i64) * 60_000);
            open.push(px);
            high.push(px + 0.0008);
            low.push(px - 0.0008);
            close.push(px + (t * 0.3).cos() * 0.0004);
            volume.push(100.0 + (i % 17) as f64);
        }
        let m1 = Ohlcv {
            timestamp: Some(timestamp),
            open,
            high,
            low,
            close,
            volume: Some(volume),
        };
        let m5 = resample_ohlcv(&m1, "M5").expect("resample M5");
        let mut frames = std::collections::HashMap::new();
        frames.insert("M1".to_string(), m1);
        frames.insert("M5".to_string(), m5);
        SymbolDataset {
            symbol: "TESTFX".to_string(),
            frames,
        }
    }

    /// The in-RAM assembly (allocate once, fill per timeframe) must produce a
    /// cube byte-identical to the streaming disk-mmap path. If these ever
    /// diverge, discovery results depend on how much free RAM the machine
    /// happened to have — the worst kind of non-determinism.
    #[test]
    fn ram_and_disk_cubes_are_identical() {
        let ds = tiny_dataset(6000);
        let opts = FeatureBuildOptions {
            higher_tfs: vec!["M5".to_string()],
            ..Default::default()
        };

        // SAFETY: single-threaded test setup; the env var is read inside
        // `should_build_cube_in_ram` on this same thread.
        unsafe { std::env::set_var("NEOETHOS_FEATURE_CUBE_MODE", "ram") };
        let ram = prepare_multitimeframe_features_with_options(&ds, "M1", &opts, None)
            .expect("in-RAM cube");
        unsafe { std::env::set_var("NEOETHOS_FEATURE_CUBE_MODE", "disk") };
        let disk = prepare_multitimeframe_features_with_options(&ds, "M1", &opts, None)
            .expect("disk cube");
        unsafe { std::env::remove_var("NEOETHOS_FEATURE_CUBE_MODE") };

        assert!(
            matches!(ram.data, crate::core::features::FeatureData::InMemory(_)),
            "ram mode must not touch disk"
        );
        assert_eq!(ram.names, disk.names, "column ORDER + names must match");
        assert_eq!(ram.timestamps, disk.timestamps);
        assert_eq!(ram.n_samples(), disk.n_samples());
        assert_eq!(ram.names.len(), disk.names.len());
        assert!(ram.n_samples() > 0 && !ram.names.is_empty());

        for r in 0..ram.n_samples() {
            for c in 0..ram.names.len() {
                let a = ram.feature_at(r, c);
                let b = disk.feature_at(r, c);
                match (a.is_nan(), b.is_nan()) {
                    (true, true) => {}
                    _ => assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "cube mismatch at row {r} col {c} ({})",
                        ram.names[c]
                    ),
                }
            }
        }
    }

    #[test]
    fn in_ram_budget_tracks_available_memory_and_honours_the_override() {
        // The decision must scale with the cube AND the machine — not a fixed
        // fraction. A byte-sized cube always fits; an absurd one never does.
        unsafe { std::env::remove_var("NEOETHOS_FEATURE_CUBE_MODE") };
        assert!(should_build_cube_in_ram(1));
        assert!(!should_build_cube_in_ram(u64::MAX / 4));

        // The 1.5x + 2 GB rule, checked against the real probe.
        let available = neoethos_core::available_memory_bytes();
        if available > 8_000_000_000 {
            let just_fits = (((available as f64) - 2.0e9) / 1.5) as u64;
            assert!(should_build_cube_in_ram(just_fits.saturating_sub(1_000_000)));
            assert!(!should_build_cube_in_ram(just_fits + 1_000_000_000));
        }

        unsafe { std::env::set_var("NEOETHOS_FEATURE_CUBE_MODE", "disk") };
        assert!(!should_build_cube_in_ram(1), "explicit disk wins");
        unsafe { std::env::set_var("NEOETHOS_FEATURE_CUBE_MODE", "ram") };
        assert!(should_build_cube_in_ram(u64::MAX / 4), "explicit ram wins");
        unsafe { std::env::remove_var("NEOETHOS_FEATURE_CUBE_MODE") };
    }
}
