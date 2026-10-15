# Verify archived Bid/Ask before using cost assumptions

`neoethos-tick-inspect` reads an existing raw cTrader tick archive without
loading credentials, connecting to the broker, modifying the archive, or
evaluating a strategy. It verifies every page through the production signed
delta decoder, exact account/request identity, pagination cursor, raw-response
digest and Vortex file hash chain. It ignores `progress.json` claims.

```powershell
cargo build --locked -p neoethos-broker-history --bin neoethos-tick-inspect
target/debug/neoethos-tick-inspect.exe `
  --archive C:/path/to/exact/archive `
  --expected-page-hash-chain <64-character-recorded-sha256> `
  --from-ms <inclusive-unix-ms> --to-ms <exclusive-unix-ms> `
  --max-quote-age-ms 1000 --max-events 2000000
```

Supply the expected hash chain from a preserved acquisition record. Matching
this digest proves consistency with those recorded bytes; it does not establish
independent broker authenticity or historical financial authority. The command
refuses incomplete archives, gaps, corrupt records, identity changes, digest
mismatches and concurrent writers. Cancellation emits no success report.

The diagnostic interval is at most seven days. All pages are verified even when
the requested interval is small. Only ticks in the interval and its freshness
seed padding are retained, subject to the explicit event ceiling (maximum ten
million). Metadata reads are bounded too.

The JSON report binds the archive, saved symbol observation, diagnostic interval
and policy with SHA-256. It includes the full verified page/tick census and
time-weighted mean, median, p95, p99 and extrema of reconstructed spreads.
Pips use the saved symbol's `pipPosition`. This observation supplies decoding
metadata, not historical commission, swap or conversion rates.

Each timestamp is grouped without asserting an order between tied Bid and Ask
events. Distinct prices on the same side in one millisecond invalidate that
side until a new unambiguous update. Identical repeats remain usable. Both
sides must have age strictly below `max-quote-age-ms`. A fresh book with Ask
below Bid is counted separately and excluded from spread statistics. Missing,
stale, ambiguous and crossed durations are reported, and together with valid
duration partition the entire requested interval, including market closures.
Quantiles weight elapsed valid-book milliseconds rather than update count.

For comparison with recorded deal times, repeat `--at-ms <unix-ms>` in strictly
increasing order (at most 1024 distinct times, all inside the diagnostic interval).
The v2 report includes `point_observations`, each with its book status, exact
integer price units (divide by 100000), side timestamps and ages. An ambiguous
side has a null price; a side with no retained preceding update is null. Both
the times and the explicit point policy are bound into the report digest.

These observations use the same grouped book as the spread statistics but only
updates **strictly earlier** than the requested time. All updates in the same
millisecond are excluded, with `same_timestamp_quotes_excluded` indicating that
such a group exists. Thus no undocumented ordering against a broker execution
is inferred. A `valid` observation passes the reconstruction freshness and price
checks; it is still not an executable quote. Compare a recorded buy with Ask and
a recorded sell with Bid only when the corresponding book is valid, and report
any price difference as a diagnostic observation, not measured execution slippage.

These are **ResearchOnly / NotPromotionEligible** diagnostics under an explicit
reconstruction policy, not exact executable spreads, actual fills, a reviewed
quote-replay ledger, or an account-profitability result. They do not update
screening assumptions automatically: account, symbol, historical scope and
financial inputs must agree first. Quotes remain outside features and models.

The broker documents separate quote-type requests, newest-first tick responses,
relative prices and the `hasMore` page limit in the official
[historical tick tutorial](https://help.ctrader.com/open-api/symbol-data/#attain-historical-tick-data)
and [message reference](https://help.ctrader.com/open-api/messages/#proto-oagettickdatares).

```powershell
cargo test --locked -p neoethos-broker-history --lib tick_archive
cargo test --locked -p neoethos-broker-history --test tick_archive_cli
```
