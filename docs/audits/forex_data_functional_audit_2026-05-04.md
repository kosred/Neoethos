# Forex Data Functional Audit

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: full functional/refactor audit of `crates/forex-data`.

Files inspected:

- `crates/forex-data/Cargo.toml`
- `crates/forex-data/src/lib.rs`
- `crates/forex-data/src/core/mod.rs`
- `crates/forex-data/src/core/timestamps.rs`
- `crates/forex-data/src/core/resample.rs`
- `crates/forex-data/src/core/features.rs`
- `crates/forex-data/src/core/session_features.rs`
- `crates/forex-data/src/core/hpc_ta.rs`
- `crates/forex-data/src/core/quant_features.rs`
- `crates/forex-data/src/core/regime_detection.rs`
- `crates/forex-data/src/core/smc.rs`
- `crates/forex-data/src/core/indicators.rs`
- `crates/forex-data/src/core/all_indicators.rs`
- `crates/forex-data/src/core/loader.rs`
- `crates/forex-data/src/core/vortex_io.rs`
- `crates/forex-data/src/core/parquet_migration.rs`

No production code was changed by this audit.

## Scope limitation

Two large files were partially truncated by the GitHub connector:

- `quant_features.rs`
- `smc.rs`

The visible sections were enough to identify the important functional contracts and risks, but this audit should be treated as a full functional/refactor audit, not a literal every-line proof for the truncated tails.

## Core conclusion

`forex-data` is the foundation of the bot. Search, models, backtests, validation, and live inference all depend on this crate producing causal, timestamp-correct, schema-stable feature data.

The crate already has useful modules and good low-level I/O structure, but it needs a stronger central feature contract.

The most important required refactor is:

```text
raw OHLCV
-> explicit timestamp unit
-> explicit candle timestamp/availability policy
-> feature generation with source/timeframe metadata
-> feature schema + schema hash
-> dataset fingerprint
-> safe cache key
-> search/model/evaluator/live consumers
```

## Module surface

`core/mod.rs` exports:

```rust
all_indicators
features
hpc_ta
indicators
loader
parquet_migration
quant_features
regime_detection
resample
session_features
smc
timestamps
vortex_io
```

This is a good modular start. The next step is to make feature contracts explicit across these modules.

## VectorTA status

`crates/forex-data/Cargo.toml` depends on:

```toml
vector-ta = "0.2.4"
```

A targeted search did not find active TA-Lib references.

Therefore the correct architecture is VectorTA-first, not TA-Lib based.

The library-backed TA layer should be represented as:

```rust
VectorTaIndicatorRegistry
IndicatorDefinition
IndicatorParameterSet
IndicatorOutputSchema
IndicatorAvailabilityReport
```

The current static `ALL_INDICATORS` list should not be treated as the permanent source of truth unless it is generated/validated from VectorTA.

Custom SMC/ICT features are different. Those are project-owned features and should remain custom because there is no reliable complete external library replacement for the full SMC/ICT feature set.

Suggested split:

```text
VectorTA features: registry-backed library indicators
Custom SMC features: project-owned ICT/SMC logic
Custom session features: project-owned session logic
Custom quant/regime features: project-owned engineered features
```

## `timestamps.rs`

Positive:

- clean small module
- defines `TimestampUnit`
- supports seconds, milliseconds, microseconds, nanoseconds
- has inference helper
- has conversion helpers
- has monotonic validation
- has tests

This is the correct foundation for timestamp standardization.

Important note:

`infer_timestamp_unit` is a migration helper. Production datasets should eventually carry timestamp unit metadata instead of relying on magnitude guessing.

Potential cleanup:

`scale_to_millis` and `scale_from_millis` can be confusing because conversion direction is not always multiply-vs-divide. Prefer explicit functions:

```rust
timestamp_to_millis
timestamp_from_millis
```

## `resample.rs`

Major issue:

`resample_ohlcv` currently computes:

```rust
period_ns = minutes * 60 * 1_000_000_000
```

This assumes nanoseconds.

It then stores resampled candle timestamps at `current_bucket_start`.

That means a higher-timeframe candle has its high/low/close known only at the end of the candle, but its timestamp is the start of the bucket.

When this is later aligned using:

```rust
feature_ts <= base_ts
```

it can create multi-timeframe lookahead.

Required fix:

Introduce an explicit candle timestamp policy:

```rust
pub enum CandleTimestampPolicy {
    OpenTime,
    CloseTime,
}
```

and a separate availability timestamp:

```rust
available_at_ms
```

For causal features, higher-timeframe features should only become available after the higher-timeframe candle is complete.

## `features.rs`

Positive:

- small and clear
- defines `FeatureProfile`
- defines `FeatureBuildOptions`
- defines `FeatureFrame`
- owns MTF alignment helper

Major issue:

`FeatureFrame` is too weak as a contract.

Current fields:

```rust
timestamps
names
data
```

Missing fields/concepts:

```rust
timestamp_unit
source_timeframe
available_at
feature_schema_hash
dataset_fingerprint
original_feature_names
effective_feature_names
column_source_metadata
```

Major issue:

`align_features_by_ns` performs:

```rust
while feature_ns[feat_idx] <= base_ts
```

This is only causal if `feature_ns` is actually an availability timestamp. If it is candle open timestamp, this leaks future higher-timeframe high/low/close information.

Required direction:

```rust
FeatureFrame {
    timestamps,
    available_at,
    timestamp_unit,
    source_timeframe,
    names,
    data,
    schema,
}
```

## `lib.rs`

`lib.rs` currently owns orchestration around:

- `Ohlcv`
- `SymbolDataset`
- symbol/timeframe discovery
- Vortex read/write
- OHLCV normalization
- feature frame construction
- multi-timeframe feature preparation

Positive:

- `normalize_ohlcv` sorts by timestamp
- deduplicates timestamps
- validates OHLC rows
- validates finite values
- rejects negative volume

Problem:

`Ohlcv` does not carry timestamp unit metadata.

Current:

```rust
pub struct Ohlcv {
    pub timestamp: Option<Vec<i64>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
}
```

Required:

```rust
pub struct Ohlcv {
    pub timestamp: Option<Vec<i64>>,
    pub timestamp_unit: TimestampUnit,
    pub candle_timestamp_policy: CandleTimestampPolicy,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
}
```

or equivalent metadata wrapper.

## `session_features.rs`

Positive:

This file now uses:

```rust
infer_timestamp_unit
timestamp_to_millis
```

This is a good improvement.

Remaining issue:

Session windows are hardcoded UTC:

```text
Asian: 00:00-08:00 UTC
London: 07:00-16:00 UTC
NY: 12:00-21:00 UTC
Overlap: 12:00-16:00 UTC
```

This should become policy-driven:

```rust
SessionFeaturePolicy {
    timezone,
    asian_start,
    asian_end,
    london_start,
    london_end,
    ny_start,
    ny_end,
    overlap_windows,
}
```

The policy must match live/backtest broker timezone assumptions.

## `hpc_ta.rs`

Positive:

- VectorTA dispatch is used.
- Multi-output indicators are decomposed into columns.
- Multi-period variants are generated for important indicators.

Major issue:

The VectorTA `Candles` are built with:

```rust
timestamps = vec![0i64; n]
```

If any VectorTA indicator uses timestamps, session logic, calendar logic, or time spacing, this is wrong.

Required fix:

Pass real timestamps from `Ohlcv`, with an explicit timestamp policy. If a given indicator is time-agnostic, that should be known via registry metadata, not silently assumed.

Major architecture issue:

`ALL_INDICATORS` is a static list. Since TA-Lib has been replaced by VectorTA, the long-term source of truth should be VectorTA registry/discovery or a generated registry validated against VectorTA.

## `all_indicators.rs`

This is a static list of indicator IDs.

It is acceptable only as a temporary compatibility layer.

It should be replaced or generated by:

```rust
VectorTaIndicatorRegistry
```

Required checks:

- every listed indicator exists in VectorTA
- every indicator has output schema metadata
- every indicator has parameter schema metadata
- every indicator is labeled as causal/non-causal/time-aware if applicable
- unavailable indicators fail visibly, not silently

## `quant_features.rs`

Visible sections contain many useful engineered features:

- multi-horizon returns
- log returns
- realized volatility
- Garman-Klass volatility
- Parkinson volatility
- volatility ratio
- Hurst exponent
- autocorrelation
- efficiency ratio
- skewness/kurtosis
- Kyle lambda proxy
- VPIN proxy
- Amihud illiquidity
- Roll spread
- candle structure
- previous day/week distance proxies
- ORB
- AMD / Wyckoff / engulfing / pivot / z-score / fractal dimension / volume-derived features

Most visible features are causal if the feature row is used only after the current bar closes.

Important issue:

Some features use fixed bar-count proxies:

```text
24 bars = previous day proxy
120 bars = previous week proxy
ORB comments assume M5
```

This is not safe across M1, M5, H1, etc.

Required fix:

Introduce:

```rust
TimeframeContext {
    timeframe,
    bar_duration_ms,
    bars_per_day,
    bars_per_week,
    session_calendar,
}
```

Previous day/week/session features should use actual calendar/session boundaries, not fixed bar counts unless explicitly configured as bar-count features.

## `regime_detection.rs`

Positive:

- clean module
- no env/random seen
- mostly causal rolling logic
- useful regime features

Main requirement:

Same as quant features: the feature row must be treated as available only after the current bar closes.

## `smc.rs`

Visible sections show project-owned SMC/ICT feature logic:

- order blocks
- FVG / IFVG
- liquidity sweeps
- premium/discount array
- macro windows
- displacement
- breaker/mitigation blocks
- MSS / BOS
- equal highs/lows
- inducement
- Asian range
- Silver Bullet
- Judas Swing
- NWOG/NDOG
- ICT macro windows
- Fibonacci levels
- rejection/propulsion/unicorn concepts

This custom logic should remain project-owned. It is not equivalent to a TA library wrapper.

Major issue:

The visible code uses:

```rust
Utc.timestamp_millis_opt(t_ms)
```

without timestamp unit inference/conversion.

If the input timestamp is nanoseconds or microseconds, all time/session/ICT-window features become wrong.

Required fix:

Use the same timestamp conversion contract as `session_features.rs`:

```rust
let unit = infer_timestamp_unit(...)
let ts_ms = timestamp_to_millis(raw_ts, unit)
```

or better: read unit from dataset metadata.

Second issue:

ICT/session windows are hardcoded UTC. These should become `SmcFeaturePolicy` / `SessionFeaturePolicy` settings.

Suggested:

```rust
SmcFeaturePolicy {
    timestamp_unit,
    timezone,
    killzones,
    macro_windows,
    silver_bullet_windows,
    asian_session,
    swing_fractal,
    ipda_lookback,
    displacement_lookback,
    displacement_multiplier,
}
```

## `indicators.rs`

Positive:

- small helper module
- no env/random
- causal rolling calculations

This is a good example of a focused helper file.

## `loader.rs`

Positive:

- clean feature cache module
- Vortex feature serialization/deserialization
- validates frame shape before writing
- removes corrupt cache file on failed read

Issue:

Feature cache freshness is currently TTL/path based.

The cache key should include:

```text
dataset fingerprint
feature profile
timestamp unit
candle timestamp policy
feature availability policy
VectorTA registry version
SMC policy hash
session policy hash
feature schema hash
code/config version
```

Otherwise stale cached features may be reused after a feature contract change.

## `vortex_io.rs`

Positive:

- clean I/O module
- atomic write using temp file and rename
- temp guard cleanup
- mmap read
- central Vortex session/runtime

This is a good small module and should stay focused.

## `parquet_migration.rs`

Positive:

- scans legacy parquet tree
- reads required OHLCV columns
- normalizes OHLCV
- writes Vortex
- verifies round-trip equivalence
- records conversion/skipped/failure summary

Issue:

Legacy parquet migration preserves raw timestamp integers but does not attach timestamp unit metadata.

Required fix:

Migration should infer or require timestamp unit and write it into dataset metadata/provenance.

## Required new contracts

### Timestamp contract

```rust
TimestampUnit
CandleTimestampPolicy
FeatureAvailabilityPolicy
```

### Dataset contract

```rust
DatasetFingerprint
OhlcvSchema
OhlcvMetadata
SymbolDatasetMetadata
```

### Feature contract

```rust
FeatureFrame
FeatureSchema
FeatureColumnMetadata
FeatureSchemaHash
FeatureSourceKind
FeatureAvailability
```

### Context contracts

```rust
TimeframeContext
SessionFeaturePolicy
SmcFeaturePolicy
VectorTaIndicatorRegistry
```

## Required tests

Add tests for:

- timestamp unit normalization across seconds/ms/us/ns
- resampling with open-time vs close-time policy
- higher-timeframe feature availability after candle close only
- MTF alignment does not leak HTF candle high/low/close before close
- VectorTA candles receive real timestamps
- SMC timestamp conversion works for ms/ns input
- session features match configured timezone/session policy
- quant previous-day/week features are timeframe-aware
- feature cache invalidates when feature schema or policy changes
- parquet migration records timestamp unit metadata

## Refactor target structure

Suggested future structure:

```text
forex-data/src/
  lib.rs
  ohlcv.rs
  dataset.rs
  timestamps.rs
  timeframe.rs
  features/
    mod.rs
    frame.rs
    schema.rs
    availability.rs
    builder.rs
    align.rs
    cache.rs
    vector_ta.rs
    smc.rs
    session.rs
    quant.rs
    regime.rs
  io/
    vortex.rs
    parquet_migration.rs
```

Keep files small and focused.

## Bottom line

`forex-data` is close to being a strong foundation, but it needs explicit data contracts.

The most urgent issues are:

1. timestamp unit must be metadata, not guessed everywhere,
2. resampled higher-timeframe candles must not leak via bucket-start timestamps,
3. feature rows need `available_at`, not only `timestamp`,
4. VectorTA must be the registry-backed TA layer,
5. static indicator lists should be generated/validated or removed,
6. custom SMC stays project-owned but must use the same timestamp/session policy,
7. feature cache keys must include schema and policy hashes.

If these are fixed, search/model/backtest/live can consume one reliable feature pipeline instead of each layer making its own assumptions.
