# Capture the account's actual broker money

The existing desktop account reader already uses `ProtoOATrader`, the same
account's asset registry, broker unrealized PnL and signed deal components.
`neoethos-broker-account-snapshot` exposes that reader without loading models or
starting trading. It uses the exact configured environment/account and the
existing secure token store. It sends only authentication and read requests.

```powershell
cargo build --locked -p neoethos-broker-history --bin neoethos-broker-account-snapshot
& ./target/debug/neoethos-broker-account-snapshot.exe `
  --environment demo --account-id <exact-account-id> `
  --out-dir C:/path/to/new-account-snapshot
```

The parent directory must exist; the output directory must be new. The output
retains exact trader, reconcile, deal, unrealized-PnL and asset responses, their
SHA-256 hashes, capture timestamps and a parsed `account-snapshot.json`. It
never writes credential requests or authentication responses. Every account
response is checked by the shared desktop reader before publication.

Currency comes from `depositAssetId` resolved through the account's actual asset
list, rather than `system.account_currency`. Broker money uses the response's
`moneyDigits`. Commission and swap are signed broker components; missing fields
remain null. The component sum is explicitly derived, not independently reported
net PnL. Equity is balance plus the broker's net unrealized PnL.

Each closing row also preserves the broker's balance and balance version,
quote-to-deposit conversion rate and closed volume. The execution's own raw
commission and scale remain separate from the commission allocated to closed
volume. Sum closing money components once; adding opening execution commission
again can double count it. Labels/comments are retained as broker observations,
not proof of which strategy version produced a trade. Missing observations stay
null. The shared desktop parser and journal use the same reported closing balance.

By default the existing runtime request covers the latest 24 hours with at most
100 deals. To inspect older executions, supply both `--deals-from-ms` and
`--deals-to-ms` as inclusive Unix-millisecond bounds, optionally with
`--max-deals` (1 to 10,000). For example, add
`--deals-from-ms 1767225600000 --deals-to-ms 1769903999999 --max-deals 1000`
to inspect January 2026. Bounds must not extend into the future. Returned deal
timestamps and row counts must fit the exact recorded query.

The snapshot retains that request and `hasMore`; a truncated page is never
reported as complete. An omitted `hasMore` remains null and cannot prove
completeness. Even a complete page proves only its requested window,
not all account history. No deals in that window does not mean zero costs.
For a truncated page, repeat in new output directories with smaller windows;
do not advance a cursor past the last timestamp, which could skip tied deals.

Capture a current full-symbol commission/swap contract separately using the
existing `neoethos-broker-symbol-contract` command. Neither current account
money nor a current symbol contract proves historical commission/swap policy
for ten years. Exact historical Bid/Ask replay and causal conversion data have
their own existing acquisition and validation boundaries.

`canonical-research` is a separate screening command. Its configured spread and
slippage plus last-D1 conversion are assumptions, even when symbol inputs came
from the broker. Its candidate rejection is not a broker-cost validation of
the full trading system. Keep that experiment's evidence, but do not substitute
its monetary results for an execution through the existing quote/economics path.

References: [cTrader currency conversion](https://help.ctrader.com/open-api/symbol-rate-conversion/)
and [broker model fields](https://help.ctrader.com/open-api/model-messages/).
