use std::path::{Path, PathBuf};
use std::time::SystemTime;
use anyhow::Result;
use polars::prelude::*;
use super::super::FeatureFrame;

pub struct FeatureCache {
    pub dir: PathBuf,
    pub ttl_minutes: u64,
    pub enabled: bool,
}

impl FeatureCache {
    pub fn new(dir: &str, ttl_minutes: u64, enabled: bool) -> Self {
        Self { dir: PathBuf::from(dir), ttl_minutes, enabled }
    }

    fn is_fresh(&self, path: &Path) -> bool {
        let Ok(meta) = std::fs::metadata(path) else { return false; };
        let Ok(mod_time) = meta.modified() else { return false; };
        let Ok(elapsed) = SystemTime::now().duration_since(mod_time) else { return false; };
        elapsed.as_secs() <= self.ttl_minutes * 60
    }

    pub fn load(&self, key: &str) -> Result<Option<FeatureFrame>> {
        if !self.enabled { return Ok(None); }
        let mut path = self.dir.clone();
        path.push(format!("{key}.parquet"));
        if !path.exists() || !self.is_fresh(&path) { return Ok(None); }
        
        let file = std::fs::File::open(&path)?;
        let df = ParquetReader::new(file).finish()?;
        Ok(Some(df_to_feature_frame(&df)?))
    }

    pub fn store(&self, key: &str, frame: &FeatureFrame) -> Result<()> {
        if !self.enabled { return Ok(()); }
        std::fs::create_dir_all(&self.dir)?;
        let mut path = self.dir.clone();
        path.push(format!("{key}.parquet"));
        let mut df = feature_frame_to_df(frame)?;
        let file = std::fs::File::create(&path)?;
        ParquetWriter::new(file).finish(&mut df)?;
        Ok(())
    }
}

pub fn feature_frame_to_df(frame: &FeatureFrame) -> Result<DataFrame> {
    let mut cols: Vec<Column> = Vec::with_capacity(frame.names.len() + 1);
    cols.push(Series::new("timestamp".into(), frame.timestamps.clone()).into());
    for (idx, name) in frame.names.iter().enumerate() {
        let mut col = Vec::with_capacity(frame.data.nrows());
        for row in 0..frame.data.nrows() { col.push(frame.data[(row, idx)]); }
        cols.push(Series::new(name.as_str().into(), col).into());
    }
    Ok(DataFrame::new(cols)?)
}

pub fn df_to_feature_frame(df: &DataFrame) -> Result<FeatureFrame> {
    let ts_series = df.column("timestamp")?.as_materialized_series();
    let timestamps: Vec<i64> = ts_series.cast(&DataType::Int64)?.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect();
    let n_rows = timestamps.len();
    let n_cols = df.width() - 1;
    let mut names = Vec::with_capacity(n_cols);
    let mut data = ndarray::Array2::<f32>::zeros((n_rows, n_cols));
    let mut col_idx = 0usize;
    for col in df.get_columns() {
        let s = col.as_materialized_series();
        if s.name() == "timestamp" { continue; }
        names.push(s.name().to_string());
        let vals: Vec<f32> = s.cast(&DataType::Float32)?.f32()?.into_iter().map(|v| v.unwrap_or(0.0)).collect();
        for i in 0..n_rows { data[(i, col_idx)] = vals[i]; }
        col_idx += 1;
    }
    Ok(FeatureFrame { timestamps, names, data })
}
