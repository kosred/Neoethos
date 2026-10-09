# Run receipt-bound CPU research

`canonical-research` connects the existing exact acquisition matrix, broker
symbol snapshot, D1 screening-cost builder, streaming discovery and chronological
holdout evaluation in one invocation. It uses the explicitly supplied settings
file for startup, CPU admission, runtime overrides and discovery.

This command always constructs screening cost assumptions. It does not use the
existing quote-replay/execution-economics consumer or the desktop's actual deal
money. A negative screening run is not a verdict on the broker-cost pipeline.
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
