# Run receipt-bound CPU research

`canonical-research` connects the existing exact acquisition matrix, broker
symbol snapshot, D1 screening-cost builder, streaming discovery and chronological
holdout evaluation in one invocation. It uses the explicitly supplied settings
file for startup, CPU admission, runtime overrides and discovery.

Candidate selection uses screening cost assumptions. By default, so does the
final holdout. An explicitly supplied saved quote-replay bundle connects the
existing quote and execution-economics consumer after the portfolio is locked.
A negative screening run is not a verdict on the broker-cost pipeline.
Use [the account evidence entrypoint](broker-account-evidence.md) to inspect the
actual account currency, broker PnL and reported commission/swap components.

```powershell
cargo build --locked -p neoethos-cli
& ./target/debug/neoethos-cli.exe canonical-research `
  --authority-root C:/path/to/canonical-trendbar-authority-v1 `
  --data-root C:/path/to/data `
  --plan-sha256 <acquisition-plan-sha256> `
  --matrix-sha256 <acquisition-matrix-sha256> `
  --broker-symbol-contract C:/path/to/bsc1-snapshot.json `
  --broker-account-snapshot C:/path/to/account/account-snapshot.json `
  --settings-source C:/path/to/config.yaml `
  --symbol EURUSD --base-timeframe H4 --higher D1 `
  --out-dir C:/path/to/new-research-run
```

The matrix selects the exact broker scope and base identity. An unrelated
dataset with the same display symbol cannot win an ambiguous inventory lookup.
Every requested higher timeframe must exist directly in that same matrix.
Settings, broker facts and cost assumptions are checked before feature work.
An existing output directory is refused so previous runs remain recoverable.

The default exploratory budget is 64 candidates per generation, 16 generations,
five indicators per gene, 16 finalist candidates and at most two streaming
batches. Override these explicitly with `--population`, `--generations`,
`--max-indicators`, `--candidates` and `--max-batches`; each accepts only a
positive integer. Zero does not mean unlimited work here. `--cpu-threads` is the
existing process CPU cap. Omitted `--higher` uses the supplied settings' ladder.

For a saved reviewed replay, supply all three flags:
`--quote-replay-bundle <json> --quote-replay-sha256 <sha256> --quote-replay-store <dir>`
and `--max-batches 1`. The store is the existing broker-truth acquisition store.
The bundle is `StoredFinalistQuoteReplayV1` schema 1: the exact acquisition outcome,
review metadata, and per-position entry financial inputs plus closed execution
economics. Positions are keyed by strategy lane and canonical decision row, with
an exact quote-ledger hash check; identical quotes in different lanes retain
their separate sizing and cash. The input hash and
acquisition link are checked before feature work. Replay reopens the reviewed
Vortex evidence and validates the locked portfolio, selection scope, sizing,
causal conversion and signed cash components. Missing, duplicate, unused or
inconsistent position inputs are refused. Deserialized metadata grants no
authority; the existing validator must independently reopen the immutable evidence.

The bundle must match the portfolio selected by this invocation, including its
final SMC policy. A different selection is refused rather than silently rebinding
old evidence. This is an explicit continuation interface, not automatic broker
acquisition. Current account/deal snapshots alone are not sufficient to produce
this bundle or establish historical fee policy. The local integration regression
uses synthetic observations in real Vortex storage; it does not establish that a
broker-backed finalist bundle has been acquired or that a strategy is profitable.

The streaming index records `quote_replay_status`, the input SHA256 and the
evidence path. `ReplayedResearchOnly` saves the existing V3 evidence to
`research.json.quote-replay.json`. `NotExecutedNoSelectedPortfolio` means the
quote consumer did not run; `Failed` preserves the batch failure. Neither is a
successful quote validation. With no bundle, the status is `NotRequested`.

Each run saves the exact `settings.yaml` and `broker-symbol-contract.json`
snapshots, `run-arguments.json`, `screening-costs.json`, `discovery-arguments.json`, the exact
research envelopes and `research.json.streaming.json`. The streaming index
records all successful and failed batches, whether evidence is complete for the
attempted batches, the candidate-selection outcome and whether the streaming
space was exhausted. A completed experiment with no candidate portfolio exits
successfully. Missing or failed batch evidence still returns an error after
preserving available evidence.

Inspect `discovery_result.funnel_profile.stage1_evaluation_window` to see the
actual row range on which the GA evolved genes. The default fast screen scores
the earliest 25% of the selection window. To evolve against all selection rows,
set `models.discovery_runtime.funnel_stage1_pct: 1.0` in the explicit research
settings file. This changes the search objective's history, not the reserved
calibration or final holdout. Record every experiment; choosing a profile from
its results is further selection, not independent validation.

The GA anneals its SMC threshold while evolving. Before finalist ranking, every
archived candidate and the complete last population are measured again at the
actual final threshold. Replay uses the same exact evaluator in waves no larger
than the admitted population; cancellation or a refused/incomplete evaluation
fails the handoff. Old profits measured at earlier thresholds cannot select the
validation cohort. This does not relax quality or walk-forward requirements.

`funnel_profile.walkforward_selection_cohort` retains all completed internal WF
trials, including rejected candidates. Each links to the exact genome and its
index in `discovery_result.candidates`, selection scope and search config hash.
The compact fold records contain actual executed trade counts, PnL and risk
breaches; return arrays are released with each validation wave. Rejection reasons
distinguish nonpositive average PnL, fewer than 60% positive folds in Risky mode,
prop-firm rule failures and invalid/incomplete fold measurements. A zero-trade
fold remains part of the existing positive-fold denominator. Multiple reasons
can apply to one candidate. Older profiles omit these diagnostics rather than
inventing measurements. These selection diagnostics are not deployment artifacts.

Raw features must also be available at their historical decision row. A
train-only normalization fit cannot make a retrospective indicator causal.
`minmax` reads later neighbours before marking an earlier extremum;
`zig_zag_channels` fills earlier segments using later confirmed pivots and
extends the last segment to the end of the supplied history. Both families are
excluded from the production vocabulary and feature metadata, including
parameter sweeps. Raw vector-ta APIs remain mathematical/chart APIs. The
existing Classic FeaturePlan fingerprints the vocabulary source, so recorded
features/models must be regenerated under the changed plan.

The public data is in the separate
[neoethos-data release](https://github.com/kosred/neoethos-data/releases/tag/snapshot-2026-08-09).
Its unchanged `EURUSD.tar` has SHA-256
`042aa66fde29e697bd30d93d85c4762076c06e292097e42680ed0067b06ca061`.
On its first 8,192 M15 rows, actual CPU producers compared against natural
prefixes of 2,048/4,096/6,553 rows reproduced 162/274/183 changed historical
ZigZag values, plus retrospective extrema changes. These are raw-feature
causality measurements within selection, not GA/OOS profitability results or
proof that a particular historical winner used those columns. SMC, Quant,
Session, Regime and Footprint were bitwise stable in this sample; remaining
Classic short-window validity changes require separate investigation.

The follow-up replay after exclusion emitted 1,776 Classic columns and compared
21,538,159 valid paired cells. At 4,096 and 6,553 rows, both historical valid
values and validity reasons matched the 8,192-row run. At 2,048 rows, two MSW
outputs differed only by about 1e-14; no material numeric change remained at
the probe's relative tolerance of 1e-10. Four `bandpass_200` outputs still had
a length-dependent availability mismatch. Its derived HighPass period is
2,667: the raw backend rejects 2,048/2,666-row inputs, but accepts 2,667 rows
and marks values valid from row 2. Those accepted values match the longer
history bitwise. This separate availability/length-contract problem remains
open; the pivot exclusion does not repair it. The rare-event MFI output's
remaining difference was only `ComputeFailure` versus `Warmup`, with no valid
cell admitted on either side of the short-prefix comparison.

CPU research uses the existing normalization projection to exclude columns with
no valid cell in the in-sample normalization interval. Both base and aligned
higher-timeframe columns use that same interval. Holdout values cannot rescue a
column. The retained schema, projection option and fitted state are bound into
each batch's content receipt. Raw indicator values and strict GPU recipes retain
their existing semantics. A discovery failure also preserves its full
error chain in `failure.json` beside the original run inputs.

The account snapshot and all five broker responses are hash-checked, checked
against the exact acquisition account, and frozen beside the run. Configured
account currency must match the observed deposit currency; mismatches are
refused before creating outputs or building features. This check proves the
observed account currency, not historical commission or swap policy.

All outputs remain `ResearchOnly` / `NotPromotionEligible`. Candidate selection
and completing an exploratory budget do not demonstrate net profitability or
exhaust the strategy space. D1-derived conversion and commission plus configured
spread/slippage are screening assumptions. Raw Bid/Ask archives need independent
verification and the existing reviewed quote-replay authority before they can
support financial validation or promotion. Chronological holdout evaluation
within this run does not establish that the data was never examined previously.

Use a CPU build for this command. CUDA builds refuse it; their device execution
continues through the existing sealed `native-research` route. No broker request,
order, model installation or promotion is performed by this command.

Method references: [cTrader historical data](https://help.ctrader.com/open-api/symbol-data/)
and [Statistical Overfitting and Backtest Performance](https://www.davidhbailey.com/dhbpapers/overfitting.pdf).
