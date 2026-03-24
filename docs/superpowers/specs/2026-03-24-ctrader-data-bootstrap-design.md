# cTrader Data Bootstrap Design

## Goal

Allow an operator to bootstrap missing local market data by selecting pairs, timeframes, and years in the app, fetching documented historical bars from `cTrader`, and writing them into the existing local parquet layout that the current discovery and training pipeline already consumes.

## Scope

This tranche adds a `bars-first` historical bootstrap flow. It covers:

- operator input for pairs, timeframes, and years
- source fallback from existing local parquet data to `cTrader` and then `MT5`
- chunked historical bar fetching
- normalization, cleaning, validation, and parquet writing
- explicit coverage reporting when the requested history is only partially satisfied

This tranche does not add tick bootstrap, background sync, automatic scheduling, or direct train/discover execution against remote APIs.

## Existing Contract

The current repo already has:

- local market data under `data/symbol=<PAIR>/timeframe=<TF>/data.parquet`
- runtime configuration pointing at `system.data_dir`
- strict parquet loading via `forex-data`
- `cTrader` symbol and historical bar access in `ctrader_data.rs`

The current repo does not yet expose historical bar retrieval through the `MT5` bridge, so this tranche must add that bridge seam before `MT5` can serve as a real fallback source.

The existing parquet layout and schema must remain unchanged so that current CLI and UI research flows continue to work without modification.

The observed local schema is:

- `timestamp`: `Datetime(ns, UTC)`
- `open`, `high`, `low`, `close`, `volume`: `Float64`

## Official API Contract

This design is based on the documented `cTrader Open API` behavior:

- historical bars are retrieved via `ProtoOAGetTrendbarsReq`
- requests use `ctidTraderAccountId`, `symbolId`, `period`, `fromTimestamp`, `toTimestamp`, and optional `count`
- timeframe-dependent limits mean large requests must be split into chunks
- live market subscriptions and tick bootstrap exist separately but are not part of this first tranche

The fallback `MT5` path uses documented range-based bars retrieval, but only within the history currently available in the terminal.

## Architecture

### 1. Bootstrap Orchestrator

Add a dedicated app service that accepts:

- selected pairs
- selected timeframes
- requested years

For each pair/timeframe request it:

- inspects existing local parquet coverage
- computes the uncovered time range
- fetches missing bars using the source ladder
- runs cleaning and validation
- writes the resulting parquet file atomically
- reports requested range, covered range, warnings, and uncovered gaps

### 2. Source Fallback Ladder

Each pair/timeframe request uses this order:

1. `Local cache`
   - inspect current parquet coverage first
   - skip remote fetch entirely if the request is already satisfied
2. `cTrader`
   - primary remote historical bars source
   - fetch in chunks using documented trendbar requests
3. `MT5`
   - fallback source only for still-uncovered segments and only when available
4. `Partial result`
   - if the requested range is still not fully covered, the result is `Degraded`, never fake success

### 3. Requested Range Semantics

The `years` input means a trailing UTC time range ending at bootstrap start time:

- `range_end_ms`: current UTC bootstrap start time
- `range_start_ms`: `range_end_ms - years * 365 days`
- `range_end_ms` is treated as an exclusive upper bound for coverage math

This tranche does not use calendar-year alignment. The goal is deterministic fetch and coverage planning for operators who want “the last N years” of data.

### 4. Cleaning And Preparation Pipeline

Fetched data must pass through a strict normalization pipeline before it can be written:

1. `Normalize`
   - map source payloads into a common `NormalizedBar`
   - canonical timestamp unit: UTC nanoseconds
   - canonical columns: `timestamp/open/high/low/close/volume`

2. `Clean`
   - sort ascending by timestamp
   - deduplicate by timestamp
   - drop or reject non-finite numeric values
   - reject impossible OHLC rows:
     - `high < low`
     - `open` or `close` outside `[low, high]`
     - negative volume

3. `Validate`
   - enforce monotonic timestamps
   - verify no duplicates remain
   - detect missing coverage
   - preserve warnings instead of inventing synthetic candles

No synthetic bars should be generated in this tranche.

### 5. Parquet Writer

The writer must:

- write exactly the existing schema
- target exactly the existing directory structure
- write atomically to avoid partial or corrupt files

Target path format:

- `data/symbol=<PAIR>/timeframe=<TF>/data.parquet`

## UI Contract

The `System` area gains a `Data Bootstrap` operator section with:

- pair selection
- timeframe selection
- years input
- source/fallback summary
- start action
- progress and coverage reporting

The UI remains thin. All fetch, cleaning, fallback, and writing logic belongs in app services.

Bootstrap progress reaches the UI through the existing job-style reporting model:

- a `Bootstrap` job snapshot
- stage/progress updates
- counters/highlights/warnings/errors
- final coverage report

The bootstrap service may expose these updates through a synchronous callback or the same snapshot/event pattern already used by discovery and training, but the transport must be explicit and testable.

## Error Handling

- Missing `cTrader` authenticated account: fail explicitly.
- Missing required symbol metadata: fail explicitly.
- Chunk request failure: surface the failing segment and continue to fallback if possible.
- Insufficient primary-source history: fall back to the next source.
- Insufficient total history after all sources: mark the result `Degraded` and report uncovered range.
- Validation failure after cleaning: fail explicitly and do not write parquet.
- Parquet write failure: fail explicitly and leave existing parquet intact.

## Testing

This tranche requires TDD coverage for:

- chunk planning by timeframe and year range
- source fallback behavior
- cleaning and deduplication
- validation failures for bad OHLC/timestamp rows
- atomic parquet writes to the expected layout
- operator-visible progress/result states

## Non-Goals

- tick bootstrap
- background incremental sync
- auto-refresh of local history
- direct training/discovery against `cTrader` APIs
- synthetic gap-filling
