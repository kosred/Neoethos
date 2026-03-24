use crate::app_services::ctrader_bootstrap::NormalizedBar;
use anyhow::{Context, Result};
use polars::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct BootstrapParquetWriter {
    root: PathBuf,
}

impl BootstrapParquetWriter {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn target_path(&self, symbol: &str, timeframe: &str) -> PathBuf {
        bootstrap_parquet_path(&self.root, symbol, timeframe)
    }

    pub fn write_normalized_bars(
        &self,
        symbol: &str,
        timeframe: &str,
        bars: &[NormalizedBar],
    ) -> Result<PathBuf> {
        let target_path = self.target_path(symbol, timeframe);
        write_normalized_bars_to_path(&target_path, bars)?;
        Ok(target_path)
    }
}

pub fn bootstrap_parquet_path(data_root: &Path, symbol: &str, timeframe: &str) -> PathBuf {
    data_root
        .join(format!("symbol={}", normalize_symbol_segment(symbol)))
        .join(format!("timeframe={}", normalize_timeframe_segment(timeframe)))
        .join("data.parquet")
}

pub fn write_bootstrap_parquet(
    data_root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    bars: &[NormalizedBar],
) -> Result<PathBuf> {
    BootstrapParquetWriter::new(data_root.as_ref().to_path_buf()).write_normalized_bars(
        symbol,
        timeframe,
        bars,
    )
}

pub fn write_normalized_bars_to_path(target_path: impl AsRef<Path>, bars: &[NormalizedBar]) -> Result<()> {
    let target_path = target_path.as_ref();
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create bootstrap parquet parent directory {}",
                parent.display()
            )
        })?;
    }

    let tmp_path = bootstrap_temp_path(target_path);
    let guard = TempFileGuard::new(tmp_path.clone());
    let mut frame = normalized_bars_to_frame(bars)?;
    let file = std::fs::File::create(&tmp_path)
        .with_context(|| format!("failed to create bootstrap temp file {}", tmp_path.display()))?;
    ParquetWriter::new(file)
        .finish(&mut frame)
        .with_context(|| format!("failed to write bootstrap parquet {}", tmp_path.display()))?;

    atomic_replace_file(&tmp_path, target_path)?;
    guard.commit();
    Ok(())
}

fn normalized_bars_to_frame(bars: &[NormalizedBar]) -> Result<DataFrame> {
    let timestamp = Series::new(
        "timestamp".into(),
        bars.iter().map(|bar| bar.timestamp_ns).collect::<Vec<_>>(),
    )
    .cast(&DataType::Datetime(
        TimeUnit::Nanoseconds,
        Some(TimeZone::UTC),
    ))
    .context("failed to cast bootstrap timestamps to UTC ns datetime")?;

    let columns: Vec<Column> = vec![
        timestamp.into(),
        Series::new(
            "open".into(),
            bars.iter().map(|bar| bar.open).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "high".into(),
            bars.iter().map(|bar| bar.high).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "low".into(),
            bars.iter().map(|bar| bar.low).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "close".into(),
            bars.iter().map(|bar| bar.close).collect::<Vec<_>>(),
        )
        .into(),
        Series::new(
            "volume".into(),
            bars.iter().map(|bar| bar.volume).collect::<Vec<_>>(),
        )
        .into(),
    ];

    DataFrame::new(columns).context("failed to build bootstrap parquet frame")
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

fn bootstrap_temp_path(target_path: &Path) -> PathBuf {
    let file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("data.parquet");
    let pid = std::process::id();
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    target_path.with_file_name(format!("{file_name}.tmp-{pid}-{nonce}"))
}

fn atomic_replace_file(src: &Path, dst: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        atomic_replace_file_windows(src, dst)
    }

    #[cfg(not(windows))]
    {
        fs::rename(src, dst).with_context(|| {
            format!(
                "failed to atomically replace {} with {}",
                dst.display(),
                src.display()
            )
        })
    }
}

#[cfg(windows)]
fn atomic_replace_file_windows(src: &Path, dst: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x0000_0001;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x0000_0008;

    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(
            lpExistingFileName: *const u16,
            lpNewFileName: *const u16,
            dwFlags: u32,
        ) -> i32;
    }

    let src_wide: Vec<u16> = src.as_os_str().encode_wide().chain(Some(0)).collect();
    let dst_wide: Vec<u16> = dst.as_os_str().encode_wide().chain(Some(0)).collect();

    let ok = unsafe {
        MoveFileExW(
            src_wide.as_ptr(),
            dst_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };

    if ok == 0 {
        Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "failed to atomically replace {} with {}",
                dst.display(),
                src.display()
            )
        })
    } else {
        Ok(())
    }
}

struct TempFileGuard {
    path: PathBuf,
    committed: bool,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed: false,
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forex_data::load_symbol_timeframe;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_root(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forex_app_bootstrap_writer_{}_{}_{}",
            test_name,
            std::process::id(),
            nonce
        ))
    }

    fn sample_bars() -> Vec<NormalizedBar> {
        vec![
            NormalizedBar {
                timestamp_ns: 1_700_000_000_000_000_000,
                open: 1.1,
                high: 1.2,
                low: 1.0,
                close: 1.15,
                volume: 10.0,
            },
            NormalizedBar {
                timestamp_ns: 1_700_000_060_000_000_000,
                open: 1.15,
                high: 1.25,
                low: 1.1,
                close: 1.2,
                volume: 12.0,
            },
        ]
    }

    #[test]
    fn target_path_uses_expected_layout() {
        let path = bootstrap_parquet_path(Path::new("data"), "eurusd", "m15");

        assert_eq!(
            path,
            PathBuf::from("data")
                .join("symbol=EURUSD")
                .join("timeframe=M15")
                .join("data.parquet")
        );
    }

    #[test]
    fn write_normalized_bars_round_trips_through_loader() {
        let root = unique_temp_root("roundtrip");
        let writer = BootstrapParquetWriter::new(&root);
        let path = writer
            .write_normalized_bars("EURUSD", "M1", &sample_bars())
            .expect("bootstrap write");

        assert_eq!(
            path,
            root.join("symbol=EURUSD")
                .join("timeframe=M1")
                .join("data.parquet")
        );

        let loaded = load_symbol_timeframe(&root, "EURUSD", "M1").expect("load written parquet");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.timestamp.as_ref().map(Vec::len), Some(2));
        assert!((loaded.open[0] - 1.1).abs() < 1e-9);
        assert_eq!(loaded.volume.as_ref().and_then(|values| values.first().copied()), Some(10.0));

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn write_normalized_bars_replaces_existing_file() {
        let root = unique_temp_root("replace");
        let writer = BootstrapParquetWriter::new(&root);
        let path = writer
            .write_normalized_bars("EURUSD", "M1", &sample_bars())
            .expect("first write");

        writer
            .write_normalized_bars(
                "EURUSD",
                "M1",
                &[NormalizedBar {
                    timestamp_ns: 1_700_000_120_000_000_000,
                    open: 2.0,
                    high: 2.1,
                    low: 1.9,
                    close: 2.05,
                    volume: 4.0,
                }],
            )
            .expect("replacement write");

        let loaded = load_symbol_timeframe(&root, "EURUSD", "M1").expect("load replacement");
        assert_eq!(loaded.len(), 1);
        assert!((loaded.open[0] - 2.0).abs() < 1e-9);
        assert_eq!(path, writer.target_path("EURUSD", "M1"));

        fs::remove_dir_all(root).ok();
    }
}
