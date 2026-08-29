# Canonical Trendbar Research and Training Plan

## Goal

Run full CUDA discovery and CUDA-capable model training from one immutable
broker-native canonical-trendbar matrix. Historical data must come only from
direct `ProtoOAGetTrendbars` generations. Resampling and tick/Bid/Ask inputs are
out of scope and must remain unreachable.

## Safety boundary

- Canonical-trendbar discovery and training are always `ResearchOnly` and
  `NotPromotionEligible`.
- The existing broker-financial-truth capability remains mandatory for live
  execution, promotion, broker PnL reconciliation, and quote replay.
- The research lane never constructs, installs, or impersonates a
  `BrokerFinancialTruthCapabilityV1`.
- Every input is selected by an explicit content-addressed matrix/series
  receipt. No `current`, symbol inventory, first-account, timeframe synthesis,
  resampling, or tick fallback is allowed.

## Implementation sequence

1. Add a versioned canonical-trendbar research execution contract in
   `neoethos-search`. Bind the exact search-input receipt, symbol, account
   currency, pip size, pip value per lot, spread, round-trip commission, swap,
   and conversion-fee assumptions. Validate all finite/range/identity fields.
2. Add an exclusive run-scoped research authority visible to the parallel
   search workers. It may authorize historical numerical evaluation only while
   the exact receipt-bound discovery entrypoint owns the scope. Existing public
   broker-real entrypoints remain fail-closed.
3. Add a dedicated full-discovery wrapper returning a strict
   `ResearchOnly`/`NotPromotionEligible` envelope. Refactor internal discovery,
   validation, CPU, CubeCL, and native-CUDA evaluation gates to accept either
   real broker truth or the active exact research authority; live/promotion
   paths continue to require broker truth only.
4. Add exact selected-series loading to model training. Refactor the existing
   orchestrator so the legacy symbol/current loader remains broker-gated, while
   the new research method accepts a validated
   `CanonicalDatasetSeriesReceiptV1`, retains every generation lease, and uses
   the explicit research pip-size contract for labels. Require a real holdout
   scope and lock training strictly before its first bar, including the
   triple-barrier purge; persist that exact cutoff in the final evidence.
5. Add a small bridge CLI that opens an explicit
   `CanonicalTrendbarMatrixReceiptV1`, selects exactly one named series and base
   timeframe, prepares the canonical feature frame, runs full CUDA discovery,
   and trains the configured model inventory into a research-only output root.
   Before feature construction, reopen and hash the exact `CONFIG_FILE` bytes,
   prove they resolve to the in-process `Settings`, and verify a no-tick
   `ProtoOASymbolByIdRes` against the acquisition plan. Spread remains an
   explicit settings-bound research assumption. Pip size and quote-currency pip
   value are recomputed from broker `pipPosition` + `lotSize`; swap and PnL
   conversion fee must equal the broker contract; round-trip commission is
   recomputed from `commissionType` + `preciseTradingCommissionRate`, an exact
   selected target-symbol close, and the exact selected quote-to-account close.
   Every direct close is generation-SHA bound to the matrix. Persist the full
   resolved `Settings` and the exact UTF-8 cost/settings/symbol-contract source
   bytes inside the final receipt-hashed artifact. Fit feature normalization only to
   the exact discovery in-sample prefix and, for training, only to the purged
   pre-OOS prefix. Derive training-label barriers from the same exact broker pip
   size and the same spread/slippage plus broker-derived round-trip commission
   charged by discovery. Hash every completed model artifact tree into the final
   evidence and re-open it before publishing the receipt; persist a receipt even
   when the training pipeline or individual models fail.
6. Verify source contracts and warning-denied CPU builds, then build the CLI's
   single `gpu-nvidia-full` feature. It combines Data/Search/native model CUDA
   with the independently gated Burn CUDA backend. The exact run config pins
   `system.device=gpu:0`, GPU-only tree execution (including LightGBM), and the
   CUDA statistical policy; the CPU-only Bayesian logit model has one explicit
   per-model `cpu` policy rather than receiving a fallback. On RTX, require the
   full discovery and every CUDA-capable selected model surface to report the
   exact CUDA device with no skip/fallback, followed by the end-to-end
   search/OOS/CPCV/training run.
7. Remove superseded historical current-generation callers, stale aliases,
   non-canonical `H2` configuration, and any documentation/scripts that imply
   ticks or resampling are part of this workflow.

## Completion evidence

- Immutable 12-symbol x 14-timeframe matrix reopens and validates.
- Exact research contract and output envelope hashes reopen unchanged.
- Full discovery consumes the receipt-bound feature frame and reports only
  real CUDA evaluation engines.
- OOS, walk-forward, CPCV/DSR/PBO, and training artifacts bind the same receipt
  and research contract.
- All requested model artifacts train, infer, save/load, and infer on the exact
  CUDA ordinal; failures remain explicit in the final inventory.
- No production caller in this lane references trendbar resampling or any
  tick/Bid/Ask capture API.
