use crate::{
    Ohlcv, load_vortex, normalize_ohlcv, symbol_timeframe_vortex_path, write_ohlcv_vortex,
};
use anyhow::{Context, Result, bail};
use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyParquetJob {
    pub symbol: String,
    pub timeframe: String,
    pub parquet_path: PathBuf,
    pub vortex_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyParquetMigrationStatus {
    Converted,
    SkippedExisting,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyParquetMigrationRecord {
    pub job: LegacyParquetJob,
    pub status: LegacyParquetMigrationStatus,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyParquetMigrationFailure {
    pub job: LegacyParquetJob,
    pub error: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LegacyParquetMigrationSummary {
    pub converted: Vec<LegacyParquetMigrationRecord>,
    pub skipped: Vec<LegacyParquetMigrationRecord>,
    pub failed: Vec<LegacyParquetMigrationFailure>,
}

impl LegacyParquetMigrationSummary {
    pub fn total_converted(&self) -> usize {
        self.converted.len()
    }

    pub fn total_skipped(&self) -> usize {
        self.skipped.len()
    }

    pub fn total_failed(&self) -> usize {
        self.failed.len()
    }
}

pub fn find_legacy_parquet_jobs(root: impl AsRef<Path>) -> Result<Vec<LegacyParquetJob>> {
    let mut jobs = Vec::new();
    collect_legacy_parquet_jobs(root.as_ref(), root.as_ref(), &mut jobs)?;
    jobs.sort_by(|lhs, rhs| lhs.parquet_path.cmp(&rhs.parquet_path));
    Ok(jobs)
}

pub fn migrate_legacy_parquet_tree(
    root: impl AsRef<Path>,
    force: bool,
    delete_source: bool,
) -> Result<LegacyParquetMigrationSummary> {
    let jobs = find_legacy_parquet_jobs(&root)?;
    let mut summary = LegacyParquetMigrationSummary::default();

    for job in jobs {
        match migrate_legacy_parquet_job(&job, force, delete_source) {
            Ok(record) => match record.status {
                LegacyParquetMigrationStatus::Converted => summary.converted.push(record),
                LegacyParquetMigrationStatus::SkippedExisting => summary.skipped.push(record),
            },
            Err(err) => summary.failed.push(LegacyParquetMigrationFailure {
                job,
                error: err.to_string(),
            }),
        }
    }

    Ok(summary)
}

pub fn migrate_legacy_parquet_job(
    job: &LegacyParquetJob,
    force: bool,
    delete_source: bool,
) -> Result<LegacyParquetMigrationRecord> {
    if !force && job.vortex_path.exists() {
        let existing = load_vortex(&job.vortex_path).with_context(|| {
            format!(
                "existing vortex dataset is unreadable {}",
                job.vortex_path.display()
            )
        })?;
        if delete_source && job.parquet_path.exists() {
            fs::remove_file(&job.parquet_path).with_context(|| {
                format!(
                    "failed to delete legacy parquet source {}",
                    job.parquet_path.display()
                )
            })?;
        }
        return Ok(LegacyParquetMigrationRecord {
            job: job.clone(),
            status: LegacyParquetMigrationStatus::SkippedExisting,
            rows: existing.len(),
        });
    }

    let source = read_legacy_parquet_ohlcv(&job.parquet_path)?;
    let normalized = normalize_ohlcv(&source)?;
    write_ohlcv_vortex(&job.vortex_path, &normalized)?;
    let verified = load_vortex(&job.vortex_path)?;
    verify_equivalent_ohlcv(&normalized, &verified)?;

    if delete_source {
        fs::remove_file(&job.parquet_path).with_context(|| {
            format!(
                "failed to delete migrated parquet source {}",
                job.parquet_path.display()
            )
        })?;
    }

    Ok(LegacyParquetMigrationRecord {
        job: job.clone(),
        status: LegacyParquetMigrationStatus::Converted,
        rows: normalized.len(),
    })
}

pub fn read_legacy_parquet_ohlcv(path: impl AsRef<Path>) -> Result<Ohlcv> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open parquet dataset {}", path.display()))?;
    let df = ParquetReader::new(file)
        .finish()
        .with_context(|| format!("failed to read parquet dataset {}", path.display()))?;

    if df.height() == 0 {
        bail!("parquet dataset is empty: {}", path.display());
    }

    let timestamps = required_i64_column(&df, &["timestamp"])?;
    let open = required_f64_column(&df, &["open", "o"])?;
    let high = required_f64_column(&df, &["high", "h"])?;
    let low = required_f64_column(&df, &["low", "l"])?;
    let close = required_f64_column(&df, &["close", "c"])?;
    let volume = optional_f64_column(&df, &["volume", "vol", "v"])?;

    Ok(Ohlcv {
        timestamp: Some(timestamps),
        open,
        high,
        low,
        close,
        volume,
    })
}

fn collect_legacy_parquet_jobs(
    root: &Path,
    current: &Path,
    jobs: &mut Vec<LegacyParquetJob>,
) -> Result<()> {
    for entry in fs::read_dir(current)
        .with_context(|| format!("failed to scan directory {}", current.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_legacy_parquet_jobs(root, &path, jobs)?;
            continue;
        }
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("data.parquet"))
        {
            let (symbol, timeframe) = parse_symbol_timeframe_from_path(root, &path)?;
            jobs.push(LegacyParquetJob {
                vortex_path: symbol_timeframe_vortex_path(root, &symbol, &timeframe),
                parquet_path: path,
                symbol,
                timeframe,
            });
        }
    }
    Ok(())
}

fn parse_symbol_timeframe_from_path(root: &Path, path: &Path) -> Result<(String, String)> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("failed to relativize {}", path.display()))?;
    let mut symbol = None;
    let mut timeframe = None;

    for component in relative.components() {
        let segment = component.as_os_str().to_string_lossy();
        if let Some(value) = segment.strip_prefix("symbol=") {
            symbol = Some(value.to_ascii_uppercase());
        } else if let Some(value) = segment.strip_prefix("timeframe=") {
            timeframe = Some(value.to_ascii_uppercase());
        }
    }

    match (symbol, timeframe) {
        (Some(symbol), Some(timeframe)) => Ok((symbol, timeframe)),
        _ => bail!(
            "failed to infer symbol/timeframe from legacy parquet path {}",
            path.display()
        ),
    }
}

fn required_i64_column(df: &DataFrame, names: &[&str]) -> Result<Vec<i64>> {
    for name in names {
        if let Ok(column) = df.column(name) {
            let series = column
                .as_materialized_series()
                .cast(&DataType::Int64)
                .with_context(|| format!("failed to cast {name} to Int64"))?;
            let chunk = series
                .i64()
                .with_context(|| format!("failed to access {name} as Int64"))?;
            if chunk.null_count() > 0 {
                bail!("column {name} contains nulls");
            }
            return Ok(chunk.into_no_null_iter().collect());
        }
    }

    bail!("missing required column {:?}", names)
}

fn required_f64_column(df: &DataFrame, names: &[&str]) -> Result<Vec<f64>> {
    for name in names {
        if let Ok(column) = df.column(name) {
            let series = column
                .as_materialized_series()
                .cast(&DataType::Float64)
                .with_context(|| format!("failed to cast {name} to Float64"))?;
            let chunk = series
                .f64()
                .with_context(|| format!("failed to access {name} as Float64"))?;
            if chunk.null_count() > 0 {
                bail!("column {name} contains nulls");
            }
            return Ok(chunk.into_no_null_iter().collect());
        }
    }

    bail!("missing required column {:?}", names)
}

fn optional_f64_column(df: &DataFrame, names: &[&str]) -> Result<Option<Vec<f64>>> {
    for name in names {
        if let Ok(column) = df.column(name) {
            let series = column
                .as_materialized_series()
                .cast(&DataType::Float64)
                .with_context(|| format!("failed to cast {name} to Float64"))?;
            let chunk = series
                .f64()
                .with_context(|| format!("failed to access {name} as Float64"))?;
            if chunk.null_count() > 0 {
                return Ok(None);
            }
            return Ok(Some(chunk.into_no_null_iter().collect()));
        }
    }

    Ok(None)
}

fn verify_equivalent_ohlcv(expected: &Ohlcv, actual: &Ohlcv) -> Result<()> {
    if expected.timestamp != actual.timestamp {
        bail!("timestamp mismatch after parquet -> vortex conversion");
    }

    verify_f64_vec("open", &expected.open, &actual.open)?;
    verify_f64_vec("high", &expected.high, &actual.high)?;
    verify_f64_vec("low", &expected.low, &actual.low)?;
    verify_f64_vec("close", &expected.close, &actual.close)?;

    match (&expected.volume, &actual.volume) {
        (Some(lhs), Some(rhs)) => verify_f64_vec("volume", lhs, rhs)?,
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => bail!("volume mismatch after conversion"),
    }

    Ok(())
}

fn verify_f64_vec(label: &str, expected: &[f64], actual: &[f64]) -> Result<()> {
    if expected.len() != actual.len() {
        bail!("{label} length mismatch after conversion");
    }

    for (idx, (lhs, rhs)) in expected.iter().zip(actual.iter()).enumerate() {
        if (lhs - rhs).abs() > 1e-12 {
            bail!("{label} mismatch at row {idx}: expected {lhs}, got {rhs}");
        }
    }

    Ok(())
}
