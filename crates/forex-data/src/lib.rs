use anyhow::{Context, Result, bail};
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::{PrimitiveArray, StructArray};
use vortex_array::dtype::NativePType;

pub mod core;
pub use crate::core::features::*;
pub use crate::core::hpc_ta::*;
pub use crate::core::indicators::*;
pub use crate::core::loader::*;
pub use crate::core::parquet_migration::*;
pub use crate::core::quant_features::*;
pub use crate::core::regime_detection::*;
pub use crate::core::resample::*;
pub use crate::core::session_features::*;
pub use crate::core::smc::*;
pub use crate::core::timestamps::*;
pub use crate::core::vortex_io::*;

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

pub fn discover_timeframes(root: impl AsRef<Path>, symbol: &str) -> Result<Vec<String>> {
    let path = PathBuf::from(root.as_ref()).join(format!("symbol={}", symbol));
    if !path.exists() {
        return Ok(Vec::new());
    }
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

pub fn load_symbol_timeframe(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
) -> Result<Ohlcv> {
    let path = symbol_timeframe_vortex_path(root, symbol, timeframe);
    if !path.exists() {
        bail!("vortex dataset not found: {}", path.display());
    }
    load_vortex(path)
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
    let timestamps = ohlcv
        .timestamp
        .as_ref()
        .context("OHLCV dataset has no timestamps")?;
    let volume = ohlcv.volume.as_ref();
    let expected_len = timestamps.len();

    if ohlcv.open.len() != expected_len
        || ohlcv.high.len() != expected_len
        || ohlcv.low.len() != expected_len
        || ohlcv.close.len() != expected_len
        || volume.is_some_and(|values| values.len() != expected_len)
    {
        bail!("OHLCV column length mismatch");
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

    let timestamp = Some(extract_non_null_primitive_vec::<i64>(
        struct_array
            .unmasked_field_by_name("timestamp")
            .context("timestamp field missing")?,
        "timestamp",
    )?);

    let get_col = |names: &[&str]| -> Result<Vec<f64>> {
        for name in names {
            if let Some(field) = struct_array.unmasked_field_by_name_opt(name) {
                return extract_non_null_primitive_vec::<f64>(field, name);
            }
        }
        bail!("missing OHLCV column {:?}", names)
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

fn extract_non_null_primitive_vec<T: NativePType>(
    array: &vortex_array::ArrayRef,
    label: &str,
) -> Result<Vec<T>> {
    if !array
        .all_valid()
        .with_context(|| format!("failed to inspect {label} validity"))?
    {
        bail!("{label} contains nulls");
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
        bail!("non-finite OHLCV value detected");
    }
    if row.high < row.low
        || row.open < row.low
        || row.open > row.high
        || row.close < row.low
        || row.close > row.high
    {
        bail!("invalid OHLC row detected");
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
        .ok_or_else(|| anyhow::anyhow!("base tf missing"))?;
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
        merged
            .slice_mut(ndarray::s![.., curr_col..curr_col + ncols])
            .assign(&part);
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
    use std::fs;
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

    fn write_valid_ohlcv_vortex(path: &Path) -> Result<()> {
        write_ohlcv_vortex(
            path,
            &Ohlcv {
                timestamp: Some(vec![1_i64, 2]),
                open: vec![1.0_f64, 1.1],
                high: vec![1.2_f64, 1.3],
                low: vec![0.9_f64, 1.0],
                close: vec![1.05_f64, 1.2],
                volume: None,
            },
        )
    }

    #[test]
    fn normalize_ohlcv_sorts_deduplicates_and_validates_rows() -> Result<()> {
        let normalized = normalize_ohlcv(&Ohlcv {
            timestamp: Some(vec![3, 1, 1, 2]),
            open: vec![1.3, 1.1, 9.9, 1.2],
            high: vec![1.4, 1.2, 9.9, 1.3],
            low: vec![1.2, 1.0, 9.9, 1.1],
            close: vec![1.35, 1.15, 9.9, 1.25],
            volume: Some(vec![2.0, 1.0, 9.9, 1.5]),
        })?;

        assert_eq!(normalized.timestamp, Some(vec![1, 2, 3]));
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
        let mut corrupt = fs::File::create(m5_dir.join("data.vortex"))?;
        std::io::Write::write_all(&mut corrupt, b"not a vortex file")?;

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
        let mut corrupt = fs::File::create(m5_dir.join("data.vortex"))?;
        std::io::Write::write_all(&mut corrupt, b"not a vortex file")?;

        let err = load_symbol_dataset_with_timeframes(&root, "EURUSD", &["M1", "M5"])
            .expect_err("requested unreadable timeframe must fail the dataset load");
        assert!(
            err.to_string().contains("M5") || err.to_string().contains("vortex"),
            "unexpected error: {err}"
        );

        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
