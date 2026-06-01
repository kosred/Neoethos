use super::super::{Ohlcv, SymbolDataset};
use anyhow::{Result, bail};

pub fn parse_timeframe_to_minutes(tf: &str) -> Result<i64> {
    let tf = tf.to_uppercase();
    if tf.starts_with('M') && !tf.starts_with("MN") {
        return Ok(tf[1..].parse::<i64>()?);
    }
    match tf.as_str() {
        "H1" => Ok(60),
        "H4" => Ok(240),
        "H6" => Ok(360),
        "H8" => Ok(480),
        "H12" => Ok(720),
        "D1" => Ok(1440),
        "W1" => Ok(10080),
        "MN1" => Ok(43200),
        _ => bail!(
            "Unsupported timeframe: '{}'. Valid: M1, M5, M15, M30, H1, H4, H12, D1, W1, MN1.",
            tf
        ),
    }
}

pub fn resample_ohlcv(src: &Ohlcv, target_tf: &str) -> Result<Ohlcv> {
    let mins = parse_timeframe_to_minutes(target_tf)?;
    let period_ns = mins * 60 * 1_000_000_000;

    let ts = src
        .timestamp
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("source has no timestamps"))?;
    if ts.is_empty() {
        return Ok(src.clone());
    }

    let mut resampled_ts = Vec::new();
    let mut resampled_open = Vec::new();
    let mut resampled_high = Vec::new();
    let mut resampled_low = Vec::new();
    let mut resampled_close = Vec::new();
    let mut resampled_volume = if src.volume.is_some() {
        Some(Vec::new())
    } else {
        None
    };

    let mut current_bucket_start = ts[0].div_euclid(period_ns) * period_ns;
    let mut b_open = src.open[0];
    let mut b_high = src.high[0];
    let mut b_low = src.low[0];
    let mut b_close = src.close[0];
    let mut b_vol = src.volume.as_ref().map(|v| v[0]).unwrap_or(0.0);

    for i in 1..ts.len() {
        let bucket = ts[i].div_euclid(period_ns) * period_ns;
        if bucket > current_bucket_start {
            resampled_ts.push(current_bucket_start);
            resampled_open.push(b_open);
            resampled_high.push(b_high);
            resampled_low.push(b_low);
            resampled_close.push(b_close);
            if let Some(ref mut v) = resampled_volume {
                v.push(b_vol);
            }

            current_bucket_start = bucket;
            b_open = src.open[i];
            b_high = src.high[i];
            b_low = src.low[i];
            b_close = src.close[i];
            b_vol = src.volume.as_ref().map(|v| v[i]).unwrap_or(0.0);
        } else {
            b_high = b_high.max(src.high[i]);
            b_low = b_low.min(src.low[i]);
            b_close = src.close[i];
            b_vol += src.volume.as_ref().map(|v| v[i]).unwrap_or(0.0);
        }
    }
    // Last bucket
    resampled_ts.push(current_bucket_start);
    resampled_open.push(b_open);
    resampled_high.push(b_high);
    resampled_low.push(b_low);
    resampled_close.push(b_close);
    if let Some(ref mut v) = resampled_volume {
        v.push(b_vol);
    }

    Ok(Ohlcv {
        timestamp: Some(resampled_ts),
        open: resampled_open,
        high: resampled_high,
        low: resampled_low,
        close: resampled_close,
        volume: resampled_volume,
    })
}

/// Subset of `neoethos_core::CANONICAL_TIMEFRAMES` that downstream pipelines
/// (discovery feature build, training MTF prep) require to be present —
/// missing timeframes will be resampled from the base timeframe.
///
/// This list intentionally stays small (the most commonly used six) to
/// avoid forcing every job to materialize every canonical timeframe.
pub const MANDATORY_TFS: &[&str] = &["M1", "M5", "M15", "H1", "H4", "D1"];

/// Full canonical timeframe list, re-exported from `neoethos_core` for the
/// convenience of neoethos-data callers that already use `MANDATORY_TFS`
/// from this module.
pub use neoethos_core::CANONICAL_TIMEFRAMES;

pub fn ensure_timeframes_with_resample(
    ds: &SymbolDataset,
    base_tf: &str,
    target_tfs: &[&str],
) -> Result<SymbolDataset> {
    let mut new_frames = ds.frames.clone();
    let base_ohlcv = ds
        .frames
        .get(base_tf)
        .ok_or_else(|| anyhow::anyhow!("base timeframe {} not found", base_tf))?;
    let base_minutes = parse_timeframe_to_minutes(base_tf)?;

    // F-309 (2026-05-28): opt-in auto-rebuild of stale higher TFs.
    // When enabled, a present-but-stale higher-TF (last bar > K × period
    // before base's last bar) is REBUILT from the base via resample
    // instead of being passed through to the aligner. Without this, the
    // F-308 max-age guard would just NaN-out the stale tail; with it,
    // the operator gets a fresh tail derived from the base.
    //
    // Gated behind FOREX_BOT_REBUILD_STALE_HIGHER_TFS=1 because
    // unsolicited auto-rebuild could surprise operators who keep
    // intentionally-stale higher-TF data (e.g. a fixed regime window).
    let rebuild_stale = std::env::var("FOREX_BOT_REBUILD_STALE_HIGHER_TFS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let base_last_ts = base_ohlcv
        .timestamp
        .as_ref()
        .and_then(|ts| ts.last())
        .copied()
        .unwrap_or(0);

    for tf in target_tfs {
        let tf_minutes = parse_timeframe_to_minutes(tf)?;
        if tf_minutes <= base_minutes {
            continue;
        }
        if let Some(existing) = new_frames.get(*tf) {
            // Existing higher-TF: check freshness if requested.
            if rebuild_stale && base_last_ts > 0 {
                let h_last = existing
                    .timestamp
                    .as_ref()
                    .and_then(|ts| ts.last())
                    .copied()
                    .unwrap_or(0);
                // 2× period lag = stale (matches F-308 max_age policy)
                let max_lag_ms = (tf_minutes as i64).saturating_mul(60 * 1000).saturating_mul(2);
                if h_last > 0 && base_last_ts.saturating_sub(h_last) > max_lag_ms {
                    tracing::warn!(
                        target: "neoethos_data::ensure_timeframes_with_resample",
                        symbol = %ds.symbol,
                        tf = tf,
                        base_last_ms = base_last_ts,
                        h_last_ms = h_last,
                        lag_ms = base_last_ts - h_last,
                        max_lag_ms,
                        "F-309: rebuilding stale higher-TF from base via resample"
                    );
                    let resampled = resample_ohlcv(base_ohlcv, tf)?;
                    new_frames.insert(tf.to_string(), resampled);
                }
            }
            // else: existing is fresh enough (or rebuild not opted in) — keep
        } else {
            // Missing TF — original behaviour: resample from base.
            let resampled = resample_ohlcv(base_ohlcv, tf)?;
            new_frames.insert(tf.to_string(), resampled);
        }
    }
    Ok(SymbolDataset {
        symbol: ds.symbol.clone(),
        frames: new_frames,
    })
}
