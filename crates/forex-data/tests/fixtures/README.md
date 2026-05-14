## Real-data fixtures

Per the operator rule "no synthetic data ever", this directory holds
real captured ticks / candles used by `#[ignore]`d round-trip tests.

### Expected files

- `EURUSD_M5_real.csv` — TODO(real-data): drop a captured cTrader M5
  CSV here. Headers must include at least:
  `time,open,high,low,close,volume`. Timestamps in UTC, monotonic.
  Used by `to_vortex::tests::csv_to_vortex_round_trip_real_data`.

### Running the ignored tests

```
cargo test -p forex-data -- --ignored
```

Tests that depend on fixtures missing from this directory will skip /
fail with a clear "fixture missing" message.
