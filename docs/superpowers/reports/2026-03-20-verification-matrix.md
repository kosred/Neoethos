# Verification Matrix

| lane | command or probe | expected outcome | prerequisites | timeout | environment | result state | evidence location |
|---|---|---|---|---|---|---|---|
| `baseline-windows` | `cargo check --workspace` | exit `0` or captured findings | Rust toolchain, workspace checkout | `20m` | current Windows host | PASS WITH FINDINGS | `cache/audit/2026-03-20-command-log.txt` |
| `baseline-windows` | `cargo test --workspace` | exit `0` or captured findings | Rust toolchain, workspace checkout | `20m` | current Windows host | FAIL | `cache/audit/2026-03-20-command-log.txt` |
| `baseline-windows` | `cargo clippy --workspace --all-targets -- -D warnings` | exit `0` or captured findings | Rust toolchain, clippy | `20m` | current Windows host | FAIL | `cache/audit/2026-03-20-command-log.txt` |
| `baseline-windows` | `cargo build -p forex-cli` | successful link/build or captured findings | Rust toolchain, workspace checkout | `20m` | current Windows host | PASS WITH FINDINGS | `cache/audit/2026-03-20-command-log.txt` |
| `baseline-windows` | `cargo build -p forex-app` | successful link/build or captured findings | Rust toolchain, workspace checkout | `20m` | current Windows host | PASS WITH FINDINGS | `cache/audit/2026-03-20-command-log.txt` |
| `baseline-linux` | `review-only lane on non-Linux host` | explicit Linux baseline run or documented non-applicability | Linux host or CI lane | `N/A` | current Windows host | N/A | `docs/superpowers/reports/2026-03-20-repo-audit-report.md` |
| `python-contract` | `python -c "import sys; print(sys.version)"` | explicit success or captured failure | Python interpreter | `30s` | current Windows host | PASS | `cache/audit/2026-03-20-command-log.txt` |
| `python-contract` | `python -c "import forex_bindings"` | explicit success, failure, or `BLOCKED` | installed binding | `30s` | current Windows host | PASS | `cache/audit/2026-03-20-command-log.txt` |
| `python-contract` | `python -c "import MetaTrader5"` | explicit success, failure, or `BLOCKED` | MetaTrader5 Python module | `30s` | current Windows host | PASS | `cache/audit/2026-03-20-command-log.txt` |
| `optional-informational-heavy-features` | `cargo check -p forex-models` | findings marked informational unless baseline-affecting | Rust toolchain | `20m` | current Windows host | PASS WITH FINDINGS | `cache/audit/2026-03-20-command-log.txt` |
| `optional-informational-heavy-features` | `cargo check -p forex-search` | findings marked informational unless baseline-affecting | Rust toolchain | `20m` | current Windows host | PASS WITH FINDINGS | `cache/audit/2026-03-20-command-log.txt` |
| `runtime-headless` | `cargo run -p forex-app -- --headless --local --config config.yaml` | successful startup or concrete failure | runtime config, app binary | TBD | current Windows host | TBD | `cache/audit/2026-03-20-command-log.txt` |
| `runtime-gui` | `cargo run -p forex-app -- --local --config config.yaml` | `PASS`, `FAIL`, or `BLOCKED` | GUI environment, runtime config | TBD | current Windows host | TBD | `cache/audit/2026-03-20-command-log.txt` |
| `runtime-mt5` | `python -c "import MetaTrader5 as mt5; ok = mt5.initialize(); print({'initialize': ok, 'terminal_info': str(mt5.terminal_info()) if ok else None, 'last_error': None if ok else mt5.last_error()}); mt5.shutdown() if ok else None"` | explicit success, failure, or `BLOCKED` | MetaTrader5 installation | TBD | current Windows host | TBD | `cache/audit/2026-03-20-command-log.txt` |
