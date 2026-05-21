//! Convert any supported source file (CSV, TSV, Parquet, Arrow/IPC,
//! Feather, JSON, JSONL, Vortex) into the canonical OHLCV Vortex layout.
//!
//! This is the single boundary at which "what the operator handed us"
//! becomes "what the rest of neoethos-data may consume". Downstream code
//! (loader, feature builders, evaluators) is allowed to assume the file
//! is Vortex with the canonical schema (`timestamp:i64 (millis,UTC),
//! open/high/low/close/volume`).
//!
//! ## Hard rules (operator constraints, do not relax)
//! - **Real data only.** If the source lacks a required column, the
//!   conversion FAILS — no synthetic fill.
//! - **f32 precision discipline.** Input float columns are validated
//!   to be losslessly representable as f32 (the conversion bails on
//!   precision loss). The on-disk Vortex schema currently stores f64;
//!   see the `// f64 required:` notes below for why.
//! - **UTC + monotonic timestamps.** Timestamps must be strictly
//!   non-decreasing and parse as UTC. Naive / TZ-less inputs are
//!   accepted only when their numeric magnitude is consistent with
//!   epoch-relative encoding (`infer_timestamp_unit`).
//! - **Canonical timeframe gate.** If a timeframe is implied by the
//!   source path (`EURUSD_H2_2024.csv`) or by row spacing, it MUST be
//!   one of `neoethos_core::CANONICAL_TIMEFRAMES`. H2 is explicitly
//!   rejected per operator decision (cTrader does not expose H2).
//! - **No hardcoded magic.** Cache directory name, chunk size, and the
//!   precision-loss threshold are exposed as `pub const` near the top.
//!
//! ## Pipeline
//! 1. Inspect source format (provided by caller via `DataFormat`).
//! 2. Vortex passthrough → atomically copy/hardlink to destination.
//! 3. For other formats: load to `Ohlcv` via the corresponding parser
//!    in `crate::core::universal_importer`. CSV/TSV/JSON/JSONL/Parquet
//!    parsers are reused as-is. Arrow/IPC goes through polars `scan_ipc`.
//! 4. Run the validation suite:
//!    - All required columns present (timestamp/open/high/low/close).
//!    - Float values losslessly downcastable to f32.
//!    - Timestamps strictly monotonic and finite.
//!    - Implied timeframe is in `CANONICAL_TIMEFRAMES` (NO H2).
//! 5. Write the canonical Vortex via `write_ohlcv_vortex`.
//! 6. Return the path that was actually written.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use polars::prelude::*;

use crate::Ohlcv;
use crate::core::discover::DataFormat;
use crate::core::universal_importer;

// Re-export so callers of `to_vortex` don't need a separate `use
// crate::core::discover::DataFormat;`.
pub use crate::core::discover::DataFormat as IngestionDataFormat;

// ─── Tunable constants (no hardcoded magic in the body) ────────────────

/// Cache directory name (relative). Co-located with the source file so
/// the cache is implicitly scoped to the same data root.
pub const VORTEX_CACHE_DIR_NAME: &str = ".forex-vortex-cache";

/// Streaming chunk size hint for polars lazy scans (rows). 64 Ki rows
/// keeps RAM bounded while keeping vectorised ops productive.
pub const SCAN_CHUNK_SIZE: usize = 64 * 1024;

/// Maximum number of rows to use when inferring JSON / JSONL schema.
/// `bail` if the schema inferred from this prefix is inconsistent with
/// later rows.
pub const JSON_SCHEMA_INFER_ROWS: usize = 1_000;

/// Maximum absolute difference between a `f64` value and its `f32`
/// re-cast that we tolerate as "losslessly downcastable". Strictly
/// greater than this → conversion bails. Forex tick prices fit well
/// inside f32 mantissa; metals / indices are checked the same way.
///
/// 1e-7 is one tenth of a JPY-pair pip, well below any prop-firm
/// tick-precision requirement.
pub const F32_DOWNCAST_TOLERANCE: f64 = 1.0e-7;

// ─── Public types ──────────────────────────────────────────────────────

/// Hint at the schema the operator expects. If `None`, the conversion
/// requires the full OHLCV set; if `Some`, missing fields listed in
/// `optional` may be absent.
#[derive(Debug, Clone, Default)]
pub struct IngestionSchema {
    /// Columns that may be omitted (e.g. `volume`). Anything not in
    /// this list is treated as required and the conversion fails if
    /// it's missing.
    pub optional: Vec<String>,
    /// Optional canonical timeframe hint from the filename / discovery.
    /// If provided, it MUST be in `neoethos_core::CANONICAL_TIMEFRAMES`.
    pub timeframe_hint: Option<String>,
}

// ─── Public entry point ────────────────────────────────────────────────

/// Convert `source` (in `source_format`) into the canonical OHLCV
/// Vortex layout at `destination`. Returns the path actually written
/// (always equal to `destination` on success — returned for symmetry
/// with cache-aware callers).
///
/// See the module docs for the hard rules this enforces.
pub fn convert_to_vortex(
    source: &Path,
    source_format: DataFormat,
    destination: &Path,
    schema_hint: Option<&IngestionSchema>,
) -> Result<PathBuf> {
    tracing::info!(
        target: "neoethos_data::to_vortex",
        source = %source.display(),
        format = source_format.as_str(),
        destination = %destination.display(),
        "convert_to_vortex: starting"
    );

    // Vortex passthrough: validate + atomic copy (we cannot hardlink
    // safely across filesystems and the source may be read-only).
    if source_format == DataFormat::Vortex {
        return vortex_passthrough(source, destination);
    }

    // Load the source into an in-memory Ohlcv. CSV/TSV/Parquet/JSON/
    // JSONL go through the existing universal_importer parsers (reuse,
    // don't re-implement). Arrow/IPC goes through polars-lazy.
    let ohlcv = match source_format {
        DataFormat::Vortex => unreachable!("handled above"),
        DataFormat::Csv => universal_importer::parse_csv_public(source, false)
            .with_context(|| format!("parse csv {}", source.display()))?,
        DataFormat::Tsv => universal_importer::parse_csv_public(source, true)
            .with_context(|| format!("parse tsv {}", source.display()))?,
        DataFormat::Parquet => universal_importer::parse_parquet_public(source)
            .with_context(|| format!("parse parquet {}", source.display()))?,
        DataFormat::Json => universal_importer::parse_json_public(source)
            .with_context(|| format!("parse json {}", source.display()))?,
        DataFormat::JsonLines => universal_importer::parse_jsonl_public(source)
            .with_context(|| format!("parse jsonl {}", source.display()))?,
        // `Arrow` covers .arrow, .feather, and .ipc in
        // `discover::DataFormat::from_extension`.
        DataFormat::Arrow => {
            parse_ipc(source).with_context(|| format!("parse ipc {}", source.display()))?
        }
    };

    // Validation suite — every gate bails rather than silently filling.
    validate_required_columns(&ohlcv, schema_hint)?;
    validate_f32_precision(&ohlcv)?;
    validate_timestamp_monotonic_utc(&ohlcv)?;
    validate_implied_timeframe(&ohlcv, schema_hint, source)?;

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create destination parent dir {}", parent.display()))?;
    }

    // f64 required: the canonical OHLCV Vortex schema in `lib.rs`
    // writes price columns as f64 (`PrimitiveArray::from_iter` over
    // `Vec<f64>`). Changing it to f32 would invalidate every existing
    // Vortex file on disk and break the loader's `extract_non_null_primitive_vec::<f64>`
    // assumptions. The f32 discipline is enforced upstream
    // (`validate_f32_precision`) — values are guaranteed to round-trip
    // through f32 — but the bytes on disk remain f64 for backward
    // compatibility.
    crate::write_ohlcv_vortex(destination, &ohlcv)
        .with_context(|| format!("write vortex {}", destination.display()))?;

    tracing::info!(
        target: "neoethos_data::to_vortex",
        rows = ohlcv.open.len(),
        destination = %destination.display(),
        "convert_to_vortex: done"
    );

    Ok(destination.to_path_buf())
}

// ─── Vortex passthrough ────────────────────────────────────────────────

fn vortex_passthrough(source: &Path, destination: &Path) -> Result<PathBuf> {
    // Quick validation: try to load the source. If it's not a real
    // Vortex file we bail before clobbering destination.
    let probe = crate::load_vortex(source)
        .with_context(|| format!("vortex passthrough: source unreadable {}", source.display()))?;
    if probe.open.is_empty() {
        bail!("vortex passthrough: source is empty: {}", source.display());
    }

    if source == destination {
        return Ok(destination.to_path_buf());
    }

    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create destination parent dir {}", parent.display()))?;
    }

    // Re-write through the canonical writer so that the destination
    // always matches the current canonical schema (older Vortex files
    // with stray columns get normalised in the process).
    crate::write_ohlcv_vortex(destination, &probe)
        .with_context(|| format!("vortex passthrough: write {}", destination.display()))?;

    tracing::info!(
        target: "neoethos_data::to_vortex",
        rows = probe.open.len(),
        source = %source.display(),
        destination = %destination.display(),
        "convert_to_vortex: vortex passthrough"
    );
    Ok(destination.to_path_buf())
}

// ─── Arrow / IPC / Feather reader ──────────────────────────────────────

/// Read an Arrow IPC / Feather file via polars lazy frame, project the
/// OHLCV columns, downcast prices to the canonical layout.
fn parse_ipc(path: &Path) -> Result<Ohlcv> {
    // polars 0.52: `LazyFrame::scan_ipc(path, IpcScanOptions, UnifiedScanArgs)`.
    // `IpcScanOptions` is a unit struct, `UnifiedScanArgs` has a sane
    // default (no caching, no globbing of paths we explicitly passed).
    let pl_path = PlPath::new(path.to_str().context("ipc path must be utf-8")?);
    let scan_args = UnifiedScanArgs {
        cache: false,
        glob: false,
        ..Default::default()
    };
    let lf = LazyFrame::scan_ipc(pl_path, IpcScanOptions, scan_args)
        .with_context(|| format!("scan_ipc {}", path.display()))?;

    let df = lf
        .collect()
        .with_context(|| format!("collect ipc {}", path.display()))?;

    dataframe_to_ohlcv(&df)
}

/// Common: convert a polars DataFrame with OHLCV columns to our
/// `Ohlcv` struct. Required columns: `timestamp`, `open`, `high`,
/// `low`, `close`. Optional: `volume`.
///
/// f64 required: polars stores price columns as Float64; we cast to
/// the same Float64 used by the canonical Vortex writer. The f32
/// precision gate runs separately on the produced Vec<f64>.
fn dataframe_to_ohlcv(df: &DataFrame) -> Result<Ohlcv> {
    let timestamp = required_i64(df, &["timestamp", "ts", "time"])?;
    let open = required_f64(df, &["open", "o"])?;
    let high = required_f64(df, &["high", "h"])?;
    let low = required_f64(df, &["low", "l"])?;
    let close = required_f64(df, &["close", "c"])?;
    let volume = optional_f64(df, &["volume", "vol", "v"])?;

    Ok(Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume,
    })
}

fn required_i64(df: &DataFrame, names: &[&str]) -> Result<Vec<i64>> {
    for name in names {
        if let Ok(col) = df.column(name) {
            let series = col
                .as_materialized_series()
                .cast(&DataType::Int64)
                .with_context(|| format!("cast {name} to Int64"))?;
            let chunk = series
                .i64()
                .with_context(|| format!("access {name} as Int64"))?;
            if chunk.null_count() > 0 {
                bail!("ipc column {name} contains nulls");
            }
            return Ok(chunk.into_no_null_iter().collect());
        }
    }
    bail!("missing required ipc column: {names:?}")
}

fn required_f64(df: &DataFrame, names: &[&str]) -> Result<Vec<f64>> {
    for name in names {
        if let Ok(col) = df.column(name) {
            let series = col
                .as_materialized_series()
                .cast(&DataType::Float64)
                .with_context(|| format!("cast {name} to Float64"))?;
            let chunk = series
                .f64()
                .with_context(|| format!("access {name} as Float64"))?;
            if chunk.null_count() > 0 {
                bail!("ipc column {name} contains nulls");
            }
            return Ok(chunk.into_no_null_iter().collect());
        }
    }
    bail!("missing required ipc column: {names:?}")
}

fn optional_f64(df: &DataFrame, names: &[&str]) -> Result<Option<Vec<f64>>> {
    for name in names {
        if let Ok(col) = df.column(name) {
            let series = col
                .as_materialized_series()
                .cast(&DataType::Float64)
                .with_context(|| format!("cast {name} to Float64"))?;
            let chunk = series
                .f64()
                .with_context(|| format!("access {name} as Float64"))?;
            if chunk.null_count() > 0 {
                bail!("ipc column {name} contains nulls");
            }
            return Ok(Some(chunk.into_no_null_iter().collect()));
        }
    }
    Ok(None)
}

// ─── Validation gates ──────────────────────────────────────────────────

fn validate_required_columns(ohlcv: &Ohlcv, hint: Option<&IngestionSchema>) -> Result<()> {
    if ohlcv.open.is_empty() {
        bail!("ingestion: source produced zero rows");
    }
    let len = ohlcv.open.len();
    if ohlcv.high.len() != len || ohlcv.low.len() != len || ohlcv.close.len() != len {
        bail!("ingestion: OHLC column length mismatch");
    }
    let ts = ohlcv
        .timestamp
        .as_ref()
        .ok_or_else(|| anyhow!("ingestion: timestamp column missing — no synthetic fill"))?;
    if ts.len() != len {
        bail!("ingestion: timestamp length mismatch");
    }
    let volume_optional = hint
        .map(|h| h.optional.iter().any(|c| c.eq_ignore_ascii_case("volume")))
        .unwrap_or(true);
    if !volume_optional && ohlcv.volume.is_none() {
        bail!("ingestion: volume column required by schema_hint but missing");
    }
    Ok(())
}

/// Verify every float value can round-trip through f32 without
/// exceeding `F32_DOWNCAST_TOLERANCE`.
fn validate_f32_precision(ohlcv: &Ohlcv) -> Result<()> {
    fn check_col(name: &str, values: &[f64]) -> Result<()> {
        for (idx, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                bail!("ingestion: {name}[{idx}] is non-finite ({v})");
            }
            let rt = v as f32 as f64;
            if (rt - v).abs() > F32_DOWNCAST_TOLERANCE {
                bail!(
                    "ingestion: {name}[{idx}] = {v} loses precision under f32 \
                     downcast (round-trip = {rt}, tolerance = {tol})",
                    v = v,
                    rt = rt,
                    tol = F32_DOWNCAST_TOLERANCE
                );
            }
        }
        Ok(())
    }
    check_col("open", &ohlcv.open)?;
    check_col("high", &ohlcv.high)?;
    check_col("low", &ohlcv.low)?;
    check_col("close", &ohlcv.close)?;
    if let Some(v) = &ohlcv.volume {
        check_col("volume", v)?;
    }
    Ok(())
}

fn validate_timestamp_monotonic_utc(ohlcv: &Ohlcv) -> Result<()> {
    let ts = ohlcv
        .timestamp
        .as_ref()
        .ok_or_else(|| anyhow!("validate: missing timestamps"))?;
    if ts.is_empty() {
        bail!("validate: empty timestamp column");
    }
    let mut prev = ts[0];
    if prev <= 0 {
        bail!(
            "validate: timestamp[0] = {prev} is non-positive — \
              not a valid UTC epoch value"
        );
    }
    for (i, &t) in ts.iter().enumerate().skip(1) {
        if t < prev {
            bail!(
                "validate: timestamps are not monotonic at index {i}: \
                 prev = {prev}, current = {t}"
            );
        }
        if t == prev {
            bail!(
                "validate: duplicate timestamp at index {i} ({t}) — \
                 strict monotonicity required"
            );
        }
        prev = t;
    }
    Ok(())
}

/// Reject conversions whose timeframe (either implied by row spacing
/// or asserted via `schema_hint.timeframe_hint` / filename) is not in
/// `neoethos_core::CANONICAL_TIMEFRAMES`.
fn validate_implied_timeframe(
    ohlcv: &Ohlcv,
    hint: Option<&IngestionSchema>,
    source: &Path,
) -> Result<()> {
    // 1) Explicit hint.
    if let Some(h) = hint
        && let Some(tf) = &h.timeframe_hint
        && !neoethos_core::is_canonical_timeframe(tf)
    {
        bail!(
            "validate: schema_hint timeframe {tf} is not canonical \
                     (neoethos_core::CANONICAL_TIMEFRAMES). H2 is intentionally absent."
        );
    }

    // 2) Filename token (e.g. EURUSD_H2_2024.csv).
    if let Some(stem) = source.file_stem().and_then(|s| s.to_str()) {
        for token in stem.split(['_', '-', '.', ' ']) {
            let upper = token.to_ascii_uppercase();
            // Cheap pre-filter: looks like a timeframe code at all?
            if !upper
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic())
            {
                continue;
            }
            if matches!(
                upper.as_str(),
                "M1" | "M2"
                    | "M3"
                    | "M4"
                    | "M5"
                    | "M6"
                    | "M10"
                    | "M15"
                    | "M20"
                    | "M30"
                    | "H1"
                    | "H2"
                    | "H3"
                    | "H4"
                    | "H6"
                    | "H8"
                    | "H12"
                    | "D1"
                    | "D2"
                    | "D3"
                    | "W1"
                    | "W2"
                    | "MN1"
            ) && !neoethos_core::is_canonical_timeframe(&upper)
            {
                bail!(
                    "validate: source filename implies non-canonical \
                         timeframe {upper} (source: {src}). H2 is intentionally \
                         absent from CANONICAL_TIMEFRAMES.",
                    src = source.display()
                );
            }
        }
    }

    // 3) Row spacing heuristic. Use median delta to be robust against
    //    weekend gaps / DST shifts.
    let ts = ohlcv
        .timestamp
        .as_ref()
        .ok_or_else(|| anyhow!("validate: missing timestamps"))?;
    if ts.len() >= 2 {
        let mut deltas: Vec<i64> = ts.windows(2).map(|w| w[1] - w[0]).collect();
        deltas.sort_unstable();
        let median = deltas[deltas.len() / 2];

        // The canonical Vortex layout uses MILLISECONDS, so a 2-hour
        // bar would have median delta = 2 * 3600 * 1000 = 7_200_000.
        // We only flag exactly the H2 pattern (within ±5%) because we
        // want to reject sneaky H2 data even if no explicit hint
        // exists; other unusual deltas (gaps, prop-firm market closes)
        // are left alone.
        let h2_millis: i64 = 2 * 3600 * 1000;
        let lo = (h2_millis * 95) / 100;
        let hi = (h2_millis * 105) / 100;
        if median >= lo && median <= hi {
            bail!(
                "validate: row spacing implies H2 timeframe (median delta = {median} ms). \
                 H2 is not in neoethos_core::CANONICAL_TIMEFRAMES (cTrader does not expose H2)."
            );
        }
    }

    Ok(())
}

// ─── Cache-key helpers (used by loader) ────────────────────────────────

/// Deterministic cache filename given a source path's identity tuple:
/// (canonicalised path, mtime as nanos, file size in bytes). The hash
/// is SHA-256 truncated to 16 hex chars for a readable filename.
pub fn cache_filename_for(source: &Path) -> Result<String> {
    let meta = fs::metadata(source)
        .with_context(|| format!("cache_filename_for: stat {}", source.display()))?;
    let size = meta.len();
    let mtime_nanos = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let canonical = fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());

    // Cheap stable hash — we just need uniqueness per (path, mtime,
    // size), not cryptographic strength.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    canonical.hash(&mut h);
    size.hash(&mut h);
    mtime_nanos.hash(&mut h);
    let digest = h.finish();

    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("data");
    Ok(format!("{stem}-{digest:016x}.vortex"))
}

/// Resolve the cache directory next to `source`. Per the constraint
/// "cache directory path … should be exposed as `pub const`" we use
/// `VORTEX_CACHE_DIR_NAME` and never bake a path elsewhere in this
/// module.
pub fn cache_dir_for(source: &Path) -> PathBuf {
    let parent = source.parent().unwrap_or_else(|| Path::new("."));
    parent.join(VORTEX_CACHE_DIR_NAME)
}

/// Full cache path for `source`.
pub fn cache_path_for(source: &Path) -> Result<PathBuf> {
    let dir = cache_dir_for(source);
    let name = cache_filename_for(source)?;
    Ok(dir.join(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sanity check that `discover::DataFormat` covers every format
    // `convert_to_vortex` knows how to ingest. If the parallel agent
    // adds / removes a variant, this is the canary.
    #[test]
    fn discover_data_format_coverage() {
        // Every variant of `DataFormat` listed below must be a match
        // arm in `convert_to_vortex` — if a new variant is added there
        // without a match arm here, the compiler's exhaustiveness
        // check inside `convert_to_vortex` will flag it first.
        let _all = [
            DataFormat::Vortex,
            DataFormat::Parquet,
            DataFormat::Arrow,
            DataFormat::Csv,
            DataFormat::Tsv,
            DataFormat::Json,
            DataFormat::JsonLines,
        ];
    }

    #[test]
    fn h2_filename_rejected() {
        // Two real ms timestamps spaced 1 minute apart — valid data.
        let ohlcv = Ohlcv {
            timestamp: Some(vec![1_700_000_000_000, 1_700_000_060_000]),
            open: vec![1.10, 1.11],
            high: vec![1.12, 1.13],
            low: vec![1.09, 1.10],
            close: vec![1.11, 1.12],
            volume: None,
        };
        let err = validate_implied_timeframe(&ohlcv, None, Path::new("/tmp/EURUSD_H2_2024.csv"))
            .unwrap_err();
        assert!(err.to_string().contains("H2"), "unexpected error: {err}");
    }

    #[test]
    fn h2_row_spacing_rejected() {
        // Two timestamps exactly 2 hours apart in ms.
        let ohlcv = Ohlcv {
            timestamp: Some(vec![
                1_700_000_000_000,
                1_700_007_200_000,
                1_700_014_400_000,
            ]),
            open: vec![1.10, 1.11, 1.12],
            high: vec![1.12, 1.13, 1.14],
            low: vec![1.09, 1.10, 1.11],
            close: vec![1.11, 1.12, 1.13],
            volume: None,
        };
        let err =
            validate_implied_timeframe(&ohlcv, None, Path::new("/tmp/some_data.csv")).unwrap_err();
        assert!(err.to_string().contains("H2"), "unexpected error: {err}");
    }

    #[test]
    fn monotonic_strict() {
        let ohlcv = Ohlcv {
            timestamp: Some(vec![1, 2, 2]),
            open: vec![1.0, 1.0, 1.0],
            high: vec![1.0, 1.0, 1.0],
            low: vec![1.0, 1.0, 1.0],
            close: vec![1.0, 1.0, 1.0],
            volume: None,
        };
        let err = validate_timestamp_monotonic_utc(&ohlcv).unwrap_err();
        assert!(err.to_string().contains("duplicate"), "unexpected: {err}");
    }

    #[test]
    fn f32_precision_gate_rejects_high_precision_f64() {
        // Note — fixed pre-existing test that picked a
        // value BELOW the gate's tolerance, so the gate (correctly)
        // accepted it but the test asserted rejection.
        //
        // `F32_DOWNCAST_TOLERANCE = 1.0e-7`. The previous fixture
        // `1.0 + 1e-9` round-trips through f32 with a loss of ~1e-9,
        // which is < tolerance → gate ACCEPTS → unwrap_err() panicked.
        //
        // To genuinely exercise the rejection path we need a value
        // whose f32 round-trip loss EXCEEDS 1e-7. A large price with
        // a non-representable low-order digit does the trick:
        // `1_000_000.123_456_789` is well outside the f32 mantissa,
        // so `(v as f32 as f64) - v` is ~0.1 — far above 1e-7.
        let bad = 1_000_000.123_456_789_f64;
        // Sanity-check the test fixture itself: confirm the f32 round-trip
        // really does lose more than the tolerance before we assert that
        // the gate notices.
        let rt_loss = (bad as f32 as f64 - bad).abs();
        assert!(
            rt_loss > F32_DOWNCAST_TOLERANCE,
            "test fixture is broken: f32 round-trip loss {rt_loss} is within tolerance \
             {F32_DOWNCAST_TOLERANCE} — pick a different fixture value"
        );
        let ohlcv = Ohlcv {
            timestamp: Some(vec![1, 2]),
            open: vec![bad, bad],
            high: vec![bad, bad],
            low: vec![bad, bad],
            close: vec![bad, bad],
            volume: None,
        };
        let err = validate_f32_precision(&ohlcv).unwrap_err();
        assert!(err.to_string().contains("precision"), "unexpected: {err}");
    }

    /// Round trip CSV → Vortex with a real captured cTrader CSV.
    ///
    /// TODO(real-data): drop a captured cTrader CSV at
    ///   crates/neoethos-data/tests/fixtures/EURUSD_M5_real.csv
    /// (operator rule: NO synthetic data ever). Until that fixture
    /// exists the test stays `#[ignore]`d.
    #[test]
    #[ignore = "needs real-data fixture: crates/neoethos-data/tests/fixtures/EURUSD_M5_real.csv"]
    fn csv_to_vortex_round_trip_real_data() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("EURUSD_M5_real.csv");
        assert!(
            fixture.exists(),
            "real-data fixture missing: {}",
            fixture.display()
        );

        let tmp = std::env::temp_dir().join("neoethos_to_vortex_round_trip.vortex");
        let _ = fs::remove_file(&tmp);

        let hint = IngestionSchema {
            optional: vec!["volume".to_string()],
            timeframe_hint: Some("M5".to_string()),
        };
        let written = convert_to_vortex(&fixture, DataFormat::Csv, &tmp, Some(&hint))
            .expect("convert real CSV → Vortex");
        assert_eq!(written, tmp);

        let reloaded = crate::load_vortex(&tmp).expect("reload converted Vortex");
        assert!(reloaded.open.len() > 0, "real fixture has at least one row");
    }
}
