use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;

pub mod core;
pub use crate::core::indicators::*;
pub use crate::core::smc::*;
pub use crate::core::resample::*;
pub use crate::core::features::*;
pub use crate::core::loader::*;

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
    pub fn len(&self) -> usize { self.close.len() }
    pub fn is_empty(&self) -> bool { self.close.is_empty() }
}

#[derive(Debug, Clone)]
pub struct SymbolDataset {
    pub symbol: String,
    pub frames: HashMap<String, Ohlcv>,
}

impl SymbolDataset {
    pub fn timeframe(&self, tf: &str) -> Option<&Ohlcv> { self.frames.get(tf) }
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

pub fn discover_timeframes(root: impl AsRef<Path>, symbol: &str) -> Result<Vec<String>> {
    let path = PathBuf::from(root.as_ref()).join(format!("symbol={}", symbol));
    if !path.exists() { return Ok(Vec::new()); }
    let mut tfs = HashSet::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("timeframe=") {
            tfs.insert(name.replace("timeframe=", "").to_uppercase());
        }
    }
    let mut out: Vec<String> = tfs.into_iter().collect();
    out.sort_by_key(|tf| parse_timeframe_to_minutes(tf).unwrap_or(999999));
    Ok(out)
}

pub fn load_symbol_timeframe(root: impl AsRef<Path>, symbol: &str, timeframe: &str) -> Result<Ohlcv> {
    let path = PathBuf::from(root.as_ref())
        .join(format!("symbol={}", symbol))
        .join(format!("timeframe={}", timeframe));
    
    if !path.exists() { bail!("path not found: {:?}", path); }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|ext| ext == "parquet") {
            return load_parquet(entry.path());
        }
    }
    bail!("no parquet files found for {} {}", symbol, timeframe);
}

pub fn load_symbol_dataset(root: impl AsRef<Path>, symbol: &str) -> Result<SymbolDataset> {
    let tfs = discover_timeframes(&root, symbol)?;
    let mut frames = HashMap::new();
    for tf in tfs {
        let ohlcv = load_symbol_timeframe(&root, symbol, &tf)
            .with_context(|| format!("failed to load dataset timeframe {} {}", symbol, tf))?;
        frames.insert(tf, ohlcv);
    }
    Ok(SymbolDataset { symbol: symbol.to_string(), frames })
}

pub fn load_symbol_dataset_with_timeframes(root: impl AsRef<Path>, symbol: &str, target_tfs: &[&str]) -> Result<SymbolDataset> {
    let mut frames = HashMap::new();
    for tf in target_tfs {
        let ohlcv = load_symbol_timeframe(&root, symbol, tf)
            .with_context(|| format!("failed to load requested dataset timeframe {} {}", symbol, tf))?;
        frames.insert(tf.to_string(), ohlcv);
    }
    Ok(SymbolDataset { symbol: symbol.to_string(), frames })
}

pub fn load_parquet(path: impl AsRef<Path>) -> Result<Ohlcv> {
    let file = std::fs::File::open(path)?;
    let df = ParquetReader::new(file).finish()?;
    
    let timestamp = match df.column("timestamp").or_else(|_| df.column("time")) {
        Ok(series) => {
            let series_name = series.name().to_string();
            let casted = series
                .as_materialized_series()
                .cast(&DataType::Int64)
                .with_context(|| format!("failed to cast {series_name} column to Int64"))?;
            let ints = casted
                .i64()
                .with_context(|| format!("{series_name} column is not Int64 after cast"))?;
            let values = ints
                .into_iter()
                .enumerate()
                .map(|(idx, value)| {
                    value.with_context(|| format!("{series_name} column contains null at row {idx}"))
                })
                .collect::<Result<Vec<_>>>()?;
            Some(values)
        }
        Err(_) => None,
    };
    
    let get_col = |names: &[&str]| -> Result<Vec<f64>> {
        for n in names {
            if let Ok(col) = df.column(n) {
                return Ok(col.as_materialized_series().cast(&DataType::Float64)?.f64()?.into_iter().map(|v| v.unwrap_or(0.0)).collect());
            }
        }
        bail!("missing column");
    };

    Ok(Ohlcv {
        timestamp,
        open: get_col(&["open", "o"])?,
        high: get_col(&["high", "h"])?,
        low: get_col(&["low", "l"])?,
        close: get_col(&["close", "c"])?,
        volume: get_col(&["volume", "vol", "v"]).ok(),
    })
}

pub fn compute_talib_features(ohlcv: &Ohlcv) -> Result<FeatureFrame> {
    compute_talib_feature_frame(ohlcv, FeatureProfile::Standard)
}

pub fn compute_talib_feature_frame(ohlcv: &Ohlcv, _profile: FeatureProfile) -> Result<FeatureFrame> {
    let mut names = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();
    
    for (name, col) in compute_smc_feature_columns(ohlcv) {
        names.push(name);
        columns.push(col);
    }
    
    let n_rows = ohlcv.len();
    let n_cols = columns.len();
    let mut data = Array2::zeros((n_rows, n_cols));
    for (c, col) in columns.iter().enumerate() {
        for (r, &val) in col.iter().enumerate() {
            data[(r, c)] = val as f32;
        }
    }
    
    Ok(FeatureFrame {
        timestamps: ohlcv.timestamp.clone().unwrap_or_default(),
        names,
        data,
    })
}

pub fn prepare_multitimeframe_features(ds: &SymbolDataset, base_tf: &str, higher_tfs: &[&str], cache: Option<&FeatureCache>) -> Result<FeatureFrame> {
    let opts = FeatureBuildOptions {
        higher_tfs: higher_tfs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    prepare_multitimeframe_features_with_options(ds, base_tf, &opts, cache)
}

pub fn prepare_multitimeframe_features_with_options(ds: &SymbolDataset, base_tf: &str, opts: &FeatureBuildOptions, _cache: Option<&FeatureCache>) -> Result<FeatureFrame> {
    let base_ohlcv = ds.frames.get(base_tf).ok_or_else(|| anyhow::anyhow!("base tf missing"))?;
    let base_ns = base_ohlcv.timestamp.as_ref().context("base has no timestamps")?;
    
    let mut all_names = Vec::new();
    let mut all_data_parts: Vec<Array2<f32>> = Vec::new();

    let base_feats = compute_talib_feature_frame(base_ohlcv, opts.profile)?;
    all_names.extend(base_feats.names.iter().map(|n| format!("{}_{}", base_tf, n)));
    all_data_parts.push(base_feats.data);

    for h_tf in &opts.higher_tfs {
        if let Some(h_ohlcv) = ds.frames.get(h_tf) {
            let h_feats = compute_talib_feature_frame(h_ohlcv, opts.profile)?;
            let h_ns = h_ohlcv.timestamp.as_ref().context("higher tf has no timestamps")?;
            let aligned = align_features_by_ns(base_ns, h_ns, &h_feats.data, true);
            all_names.extend(h_feats.names.iter().map(|n| format!("{}_{}", h_tf, n)));
            all_data_parts.push(aligned);
        }
    }

    let total_cols = all_data_parts.iter().map(|p| p.ncols()).sum();
    let mut merged = Array2::zeros((base_ns.len(), total_cols));
    let mut curr_col = 0;
    for part in all_data_parts {
        let ncols = part.ncols();
        merged.slice_mut(ndarray::s![.., curr_col..curr_col+ncols]).assign(&part);
        curr_col += ncols;
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
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_root(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "forex_data_{}_{}_{}",
            test_name,
            std::process::id(),
            nonce
        ))
    }

    fn write_valid_ohlcv_parquet(path: &Path) -> Result<()> {
        let df = DataFrame::new(vec![
            Series::new("timestamp".into(), vec![1_i64, 2]).into(),
            Series::new("open".into(), vec![1.0_f64, 1.1]).into(),
            Series::new("high".into(), vec![1.2_f64, 1.3]).into(),
            Series::new("low".into(), vec![0.9_f64, 1.0]).into(),
            Series::new("close".into(), vec![1.05_f64, 1.2]).into(),
        ])?;
        let file = File::create(path)?;
        ParquetWriter::new(file).finish(&mut df.clone())?;
        Ok(())
    }

    #[test]
    fn load_symbol_dataset_rejects_unreadable_discovered_timeframe() -> Result<()> {
        let root = unique_temp_root("unreadable_discovered_timeframe");
        let m1_dir = root.join("symbol=EURUSD").join("timeframe=M1");
        let m5_dir = root.join("symbol=EURUSD").join("timeframe=M5");
        fs::create_dir_all(&m1_dir)?;
        fs::create_dir_all(&m5_dir)?;

        write_valid_ohlcv_parquet(&m1_dir.join("data.parquet"))?;
        let mut corrupt = File::create(m5_dir.join("data.parquet"))?;
        corrupt.write_all(b"not a parquet file")?;

        let err = load_symbol_dataset(&root, "EURUSD")
            .expect_err("discovered unreadable timeframe must fail the dataset load");
        assert!(
            err.to_string().contains("M5") || err.to_string().contains("parquet"),
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

        write_valid_ohlcv_parquet(&m1_dir.join("data.parquet"))?;
        let mut corrupt = File::create(m5_dir.join("data.parquet"))?;
        corrupt.write_all(b"not a parquet file")?;

        let err = load_symbol_dataset_with_timeframes(&root, "EURUSD", &["M1", "M5"])
            .expect_err("requested unreadable timeframe must fail the dataset load");
        assert!(
            err.to_string().contains("M5") || err.to_string().contains("parquet"),
            "unexpected error: {err}"
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn load_parquet_rejects_invalid_timestamp_column() -> Result<()> {
        let root = unique_temp_root("invalid_timestamp_column");
        fs::create_dir_all(&root)?;
        let parquet_path = root.join("invalid_timestamp.parquet");

        let mut df = DataFrame::new(vec![
            Series::new("timestamp".into(), vec!["bad", "worse"]).into(),
            Series::new("open".into(), vec![1.0_f64, 1.1]).into(),
            Series::new("high".into(), vec![1.2_f64, 1.3]).into(),
            Series::new("low".into(), vec![0.9_f64, 1.0]).into(),
            Series::new("close".into(), vec![1.05_f64, 1.2]).into(),
        ])?;
        let file = File::create(&parquet_path)?;
        ParquetWriter::new(file).finish(&mut df)?;

        let err = load_parquet(&parquet_path)
            .expect_err("invalid timestamp values must fail instead of becoming synthetic zeroes");
        assert!(
            err.to_string().contains("timestamp"),
            "unexpected error: {err}"
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
