# Artifact Intent Clarification — Training vs Search Resume

Created: 2026-05-03 Europe/Berlin
Repository: kosred/forex-ai
Branch: master

## Why this note exists

During the search/live artifact audit, the user pointed out that some artifacts may not be intended for live trading. They may instead exist for model training, saved alpha strategies, or resuming work after interruption.

That correction is important. The repo should distinguish artifact intent instead of treating every saved JSON/database record as a live-trading portfolio.

## Clarified findings

### 1. Training artifacts and runtime profiles are real

`forex-models` contains training-oriented runtime artifacts and profiles.

`TrainingRuntimeProfile` includes fields such as:

- model name
- symbol
- base timeframe
- feature count
- dataset rows
- row budget
- label horizon
- higher timeframes
- backend/device/precision
- HPO configuration
- train/validation years
- `checkpoint_path`
- async flags

This supports the idea that part of the artifact system is for model training/runtime tracking, not necessarily live execution.

### 2. Strategy ledger storage exists

`forex-core/src/storage.rs` defines `StrategyLedger` and a SQLite table `alpha_strategies`, plus `trade_log`, `pending_intents`, settings, and live metrics.

It has a `save_alpha` path for saved strategy records.

This looks like storage/ledger infrastructure for strategies and runtime records, not automatically a direct live execution bridge.

### 3. Search/discovery resume is not yet proven

Searches for active `resume`, `save_state`, `load_state`, `resume_from`, and search checkpoint patterns did not reveal a clear implemented mechanism that resumes an interrupted discovery/search generation exactly where it stopped.

So the current evidence supports:

- model training profile/checkpoint metadata: yes
- saved alpha strategy ledger: yes
- automatic interrupted discovery/search resume: not proven from current search

### 4. Discovery portfolio export may have multiple possible intents

A discovery portfolio export could serve several different purposes:

- inspection/report artifact
- candidate pool for later model training
- saved alpha strategy knowledge
- seed material for future search
- future live-ready artifact after stricter contract work

These intents require different contracts.

## Recommended classification

Add an explicit artifact type field to every saved artifact:

- `training_runtime_profile`
- `model_artifact`
- `search_checkpoint`
- `search_candidate_export`
- `saved_alpha_strategy`
- `backtest_report`
- `forward_test_run`
- `live_strategy_runtime_artifact`

Each artifact type should have different validation rules.

## Recommended next steps

1. Do not delete or dismiss artifacts just because they are not live-ready.
2. Identify which artifacts are for training, which are for saved strategies, and which are for future live runtime.
3. If interrupted search resume is desired, implement an explicit `SearchCheckpoint` containing generation, population, archive, RNG state/seed, effective feature schema, config hash, and partial evaluation state.
4. If discovery exports are only for inspection/training, label them as such.
5. Only artifacts with a strict live runtime contract should be allowed near order execution.

## Bottom line

The user's memory is likely partly correct: the repo does contain training/runtime profiles and strategy storage infrastructure. What is not yet proven is a complete resume mechanism for interrupted discovery/search. The audit should therefore classify artifact intent rather than assume every saved portfolio is meant for live trading.
