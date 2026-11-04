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

Internal WF replays already evolved genes; it does not rerun feature selection
or the GA inside each fold. Its 70/30 window geometry chooses replay rows, not
a new training/validation experiment. The feature prefilter defaults to the
whole selection interval, normalization is fitted on that interval, and the
full-selection quality screen runs before WF. Consequently these folds have
already influenced candidate selection. Embargo separates replay boundaries;
it does not remove that earlier selection exposure. A passing internal WF is
a robustness screen within selection, not independent OOS evidence.

Every internal-WF survivor is replayed on the separate selection-calibration
interval before correlation and active portfolio capacity are applied. Its
finite metrics and positive net profit, expectancy, profit factor and sizing
fraction are required by the existing calibration gate. Inspect
`funnel_profile.selection_calibration_cohort.trials` for each measured summary
and rejection reason, including losing candidates. If no candidate survives,
the portfolio is empty and the reserved final holdout is not evaluated. That
outcome must not be described as a measured final-OOS failure.

The trial-return matrix records the candidates admitted to the post-GA quality
screen, before that screen's gates. It does not contain every genome evaluated
during evolution, nor the earlier feature-selection alternatives. Its DSR
uses the screen's offered count; its CSCV/PBO reranks the recorded, already
evolved candidates. These statistics describe that recorded cohort, not the
complete adaptive search. They cannot certify that the GA or repeated research
attempts have avoided overfitting. Merely increasing the DSR trial count cannot
reconstruct the omitted return paths or undo prior candidate selection.

To attribute a regression, compare the same frozen data, costs, policy, seeds
and budget with only the suspect mechanism changed. Record selection fitness,
actual generations/evaluation slots, every rejection stage and calibration
returns. A convergence fix must first change the false-stagnation behavior;
better unseen returns require a separate market-data experiment. Keep final
holdout results out of profile selection. The native convergence change also
needs actual CUDA validation; CPU/source checks do not supply that evidence.

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
