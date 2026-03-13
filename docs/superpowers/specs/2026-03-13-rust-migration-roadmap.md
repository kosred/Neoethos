# Rust Migration Roadmap

## Goal

Track the long-term subsystem order for moving `forex-ai` from a mixed Python/Rust codebase to a Rust-owned runtime with Python only where external integrations still demand it.

## Phase Order

1. Milestone A+B: offline data core plus feature/label/discovery-input core
2. Milestone C: discovery and backtest kernels
3. Milestone D: training orchestration and pooled dataset assembly
4. Milestone E: runtime shell and CLI migration
5. Milestone F: MT5/live trading migration or permanent isolation as a thin external adapter

## Milestone Summaries

### Milestone C

- complete Rust ownership of discovery candidate evaluation
- complete portfolio/quality/truth gates in Rust
- remove Python-side discovery kernel drift

### Milestone D

- move pooled dataset merge and feature-space alignment out of Python
- move shard merge, sort/dedup, and train/eval split orchestration to Rust
- reduce `training_service.py` and `trainer.py` to thin adapters or delete them if fully superseded

### Milestone E

- move runtime orchestration out of `main.py` and `forex-ai.py`
- keep Python shell only if needed for temporary compatibility

### Milestone F

- isolate or port MT5/live execution after offline parity is stable
- avoid mixing broker integration work with core performance migration
