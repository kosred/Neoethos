# Capture only the direct timeframes needed by an experiment

`neoethos-canonical-trendbar-bulk` accepts repeated `--timeframe` flags. Without
them, it preserves the existing behavior and captures all fourteen canonical
timeframes for every explicitly selected symbol. An explicit selection creates
a smaller immutable plan; it never resamples candles or expands the scope to
unrequested timeframes.

For an H1 experiment using direct H4/D1 context and GBP account-currency cost
assumptions, include EURUSD and its GBPUSD conversion instrument in the same
account-bound plan. Use symbol IDs observed in that account's broker catalog.

```powershell
cargo build --locked -p neoethos-broker-history --bin neoethos-canonical-trendbar-bulk
target/debug/neoethos-canonical-trendbar-bulk.exe `
  --environment demo --account-id <exact-account-id> `
  --symbol <eurusd-id>=EURUSD --symbol <gbpusd-id>=GBPUSD `
  --timeframe H1 --timeframe H4 --timeframe D1 `
  --to-ms-exclusive <fixed-unix-ms> `
  --data-root C:/path/to/new/data `
  --authority-root C:/path/to/new/authority
```

This example has six cells rather than twenty-eight. The source is direct
broker trendbars from 2016-01-01 to the fixed exclusive cutoff. The resulting
matrix proves completion of exactly those six cells. It does not claim that
the other timeframes were acquired. Duplicate and noncanonical timeframe names
are refused before capture.

Cancellation and transient-error resumes keep the original plan unchanged. A
checkpoint from a different timeframe selection cannot be attached to the new
plan. The JSON completion result contains the plan, checkpoint and matrix
SHA-256 receipts required by
[`canonical-research`](canonical-cpu-research.md). Missing higher timeframes or
currency-conversion series still fail the research preflight; include them
explicitly when planning acquisition.

The capture and research commands issue no trading permit. Bid/Ask diagnostics
remain a separate execution-evidence input, outside features and models; see
[`neoethos-tick-inspect`](offline-tick-diagnostics.md).

```powershell
cargo test --locked -p neoethos-broker-history --lib bulk_cli
cargo test --locked -p neoethos-broker-history --test canonical_trendbar_bulk_cli_contract
```
