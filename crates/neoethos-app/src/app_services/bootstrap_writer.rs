use crate::app_services::ctrader_bootstrap::NormalizedBar;
use anyhow::Result;
use neoethos_data::{Ohlcv, symbol_timeframe_vortex_path, write_symbol_timeframe_vortex};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct BootstrapVortexWriter {
    root: PathBuf,
}

impl BootstrapVortexWriter {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn write_normalized_bars(
        &self,
        symbol: &str,
        timeframe: &str,
        bars: &[NormalizedBar],
    ) -> Result<PathBuf> {
        write_symbol_timeframe_vortex(
            &self.root,
            symbol,
            timeframe,
            &normalized_bars_to_ohlcv(bars),
        )
    }
}

pub fn bootstrap_vortex_path(data_root: &Path, symbol: &str, timeframe: &str) -> PathBuf {
    symbol_timeframe_vortex_path(data_root, symbol, timeframe)
}

pub fn write_bootstrap_vortex(
    data_root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
    bars: &[NormalizedBar],
) -> Result<PathBuf> {
    BootstrapVortexWriter::new(data_root.as_ref().to_path_buf())
        .write_normalized_bars(symbol, timeframe, bars)
}

fn normalized_bars_to_ohlcv(bars: &[NormalizedBar]) -> Ohlcv {
    Ohlcv {
        timestamp: Some(bars.iter().map(|bar| bar.timestamp_ns).collect()),
        open: bars.iter().map(|bar| bar.open).collect(),
        high: bars.iter().map(|bar| bar.high).collect(),
        low: bars.iter().map(|bar| bar.low).collect(),
        close: bars.iter().map(|bar| bar.close).collect(),
        volume: Some(bars.iter().map(|bar| bar.volume).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_data::load_symbol_timeframe;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_root(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("vortex_{test_name}_{nonce}"))
    }

    fn sample_bars() -> Vec<NormalizedBar> {
        vec![NormalizedBar {
            timestamp_ns: 1,
            open: 1.1,
            high: 1.2,
            low: 1.0,
            close: 1.15,
            volume: 10.0,
        }]
    }

    #[test]
    fn roundtrip_vortex() {
        let root = unique_temp_root("roundtrip");
        let writer = BootstrapVortexWriter::new(&root);
        writer
            .write_normalized_bars("EURUSD", "M1", &sample_bars())
            .unwrap();
        let loaded = load_symbol_timeframe(&root, "EURUSD", "M1").unwrap();
        assert_eq!(loaded.len(), 1);
        let _ = fs::remove_dir_all(root);
    }
}
