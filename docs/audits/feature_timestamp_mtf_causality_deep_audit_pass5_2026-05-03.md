# Feature / Timestamp / Multi-Timeframe Causality Deep Audit — Pass 5

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master
Scope: feature generation, timestamp units, resampling, multi-timeframe alignment, session features, VectorTA timestamp handling, causality before CPU/GPU evaluation.

## Summary

CPU/GPU parity is necessary but not sufficient. If the feature matrix contains lookahead or inconsistent timestamp semantics, then both CPU and GPU will consistently evaluate the wrong input.

This pass identifies a critical data/feature contract issue: timestamp unit handling is inconsistent across modules, and higher-timeframe feature alignment can leak future HTF candle information.

## Findings

### 1. Higher-timeframe resampling labels candles at bucket start

`resample_ohlcv` computes `period_ns` and assigns resampled timestamps as `current_bucket_start`.

However, the resampled OHLC values contain the full bucket's open/high/low/close.

A higher-timeframe bar's high/low/close is not known at bucket start. It is known only after the bucket closes.

**Risk:** H1/H4/D1 features can become available too early when aligned to lower timeframe rows.

**Severity:** Critical.

**Fix direction:** resampled HTF candles must carry both:

- `bar_start_ts`
- `bar_close_ts` / `available_at_ts`

Feature alignment must use `available_at_ts`, not bucket start.

---

### 2. `align_features_by_ns` can forward-fill future HTF features

`align_features_by_ns` selects the latest feature where:

```rust
feature_ns[feat_idx] <= base_ts
```

This is correct only if `feature_ns` represents the time when the feature is actually known.

If `feature_ns` is the HTF bucket start, then lower timeframe rows inside the same bucket can see HTF high/low/close before that HTF candle has closed.

**Risk:** classic multi-timeframe lookahead.

**Severity:** Critical.

**Fix direction:** alignment must use feature availability timestamps. For HTF bars, availability should be close time, not open time.

---

### 3. Timestamp unit is inconsistent across modules

Some data functions use variable names like `base_ns`, `h_ns`, and compute resample periods in nanoseconds.

Other modules expect milliseconds:

- `session_features.rs` uses `Utc.timestamp_millis_opt(ts)`
- `eval.rs` day keys use `ts / 86_400_000`
- `simulate_trades_core` computes durations using `/ 3_600_000.0`
- CUDA `timestamp_delta_ms` treats raw deltas as milliseconds
- validation passes day keys to functions expecting timestamps in milliseconds

**Risk:** session features, kill-zones, daily grouping, duration, gap handling, validation diagnostics, and GPU gap logic can all be wrong depending on timestamp unit.

**Severity:** Critical.

**Fix direction:** introduce a typed timestamp contract:

```rust
pub enum TimestampUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}
```

Normalize to one internal canonical unit before feature/eval. Prefer milliseconds for chrono/session/eval, or nanoseconds with explicit conversions.

---

### 4. Session features assume millisecond timestamps

`compute_session_feature_columns` parses timestamps with:

```rust
Utc.timestamp_millis_opt(ts)
```

If OHLCV timestamps are nanoseconds, session features will be incorrect.

**Risk:** London/NY/Asian session features, daily features, session open gaps, VWAP distances, and overlap features become invalid.

**Severity:** Critical.

**Fix direction:** pass timestamp unit into session feature generation or normalize OHLCV timestamps before feature computation.

---

### 5. VectorTA receives zero timestamps

`compute_classic_ta_columns` builds VectorTA candles using:

```rust
let timestamps = vec![0i64; n];
```

If any VectorTA indicator depends on timestamps, anchors, sessions, calendar logic, or time deltas, results can be wrong.

If VectorTA indicators only use OHLCV arrays, this may be harmless, but that must be proven by tests or restricted by contract.

**Risk:** hidden timestamp-insensitive assumption inside external TA computation.

**Severity:** Medium-High.

**Fix direction:** pass real normalized timestamps into VectorTA candles. If some indicators are timestamp-independent, document and test it.

---

### 6. Fixed bar-count “daily/weekly” quant features are timeframe-dependent

Quant features use fixed periods such as 24 bars for “previous day” and 120 bars for “previous week”. Comments imply H1/M5 proxies, but these features are computed regardless of actual timeframe.

**Risk:** on M1/M5/M15/H1, the same feature name means different calendar spans.

**Severity:** High.

**Fix direction:** compute calendar-aware daily/weekly features from timestamps and timeframe metadata, or encode the actual bar-count/timeframe in the feature name.

---

### 7. Multi-timeframe feature schema must include availability semantics

`FeatureFrame` currently stores:

- timestamps
- names
- data

It does not store:

- timestamp unit
- bar start time
- bar close time
- feature availability time
- source timeframe per column
- feature generator version
- lookback/warmup requirements
- causality status

**Risk:** search and export cannot prove that feature indices correspond to causal, available-at-the-time features.

**Severity:** High.

**Fix direction:** add a `FeatureSchema` / `FeatureColumnMeta` structure.

---

### 8. Base timeframe can duplicate through higher timeframe list

If `higher_tfs` includes `base_tf`, the project previously showed risk of duplicate base features. Some paths skip duplicates, others may not consistently enforce it.

**Risk:** duplicated features bias search and make feature schemas inconsistent across entrypoints.

**Severity:** Medium.

**Fix direction:** canonicalize timeframe lists once in config: remove duplicates, remove base from higher list, sort by timeframe duration, and export the effective list.

---

## Required feature contract

Introduce:

```rust
pub struct FeatureFrameContract {
    pub timestamp_unit: TimestampUnit,
    pub base_timeframe: String,
    pub feature_profile: FeatureProfile,
    pub base_bar_start_ts: Vec<i64>,
    pub base_bar_close_ts: Vec<i64>,
    pub effective_higher_timeframes: Vec<String>,
    pub columns: Vec<FeatureColumnMeta>,
    pub schema_hash: String,
}

pub struct FeatureColumnMeta {
    pub name: String,
    pub source_timeframe: String,
    pub generator: String,
    pub lookback_bars: usize,
    pub available_at: FeatureAvailability,
    pub causal: bool,
}

pub enum FeatureAvailability {
    SameBarClose,
    PreviousBarClose,
    HigherTimeframeClose,
    SessionAccumulatedToCurrentBar,
    DerivedFromPastOnly,
}
```

## Required causality rule

A feature value at base row `i` may only use information available no later than the execution decision time.

If the evaluator enters at bar `i` based on signal from bar `i-1`, then feature row `i-1` must not use information from bar `i` or later.

For HTF features, this means an H1 candle ending at 10:00 may only be visible to M1 rows at or after 10:00, not from 09:00 onward.

## Required tests

1. `resampled_htf_feature_available_only_after_bucket_close`
2. `align_features_uses_available_at_not_bucket_start`
3. `timestamp_unit_milliseconds_session_features_valid`
4. `timestamp_unit_nanoseconds_normalized_before_session_features`
5. `eval_day_keys_consistent_after_timestamp_normalization`
6. `cuda_gap_logic_receives_milliseconds`
7. `vector_ta_receives_real_timestamps`
8. `fixed_bar_day_week_features_are_timeframe_aware`
9. `higher_tfs_canonicalized_no_base_duplicate`
10. `feature_schema_hash_changes_when_feature_order_or_availability_changes`

## Implementation order

1. Add `TimestampUnit` and timestamp normalization utilities.
2. Decide canonical internal timestamp unit.
3. Update OHLCV/FeatureFrame to carry timestamp unit or normalized timestamps.
4. Change resampling to produce bar start and bar close/available timestamps.
5. Change MTF alignment to use availability timestamps.
6. Update session features to use normalized timestamp units.
7. Pass real timestamps into VectorTA candles.
8. Add feature column metadata and schema hash.
9. Make all entrypoints export effective feature schema.
10. Only allow search/backtest if feature contract is causal and validated.

## Bottom line

The feature layer must become causal and timestamp-explicit before CPU/GPU parity can be trusted. A perfectly matched CPU and GPU evaluator is still wrong if both are fed higher-timeframe features that leaked future candle data or timestamps interpreted in different units.
