# Scripts Maintenance Status

This file tracks active vs legacy script entrypoints during the Rust-first migration.

## Active

- `run_prop_discovery.py`
  - Used by test imports and the current prop-discovery workflow.
- `train_multi_gpu.py`
  - Active launcher for one-process-per-GPU training.

## Candidate Legacy (review before removal)

- `sync_mt5_history.py`
  - No in-repo references found.
  - Likely manual operator utility.
- `verify_bindings.py`
  - No in-repo references found.
  - Legacy local debug helper for old `target/debug` import patterns.

## Removal Policy

Before deleting candidate legacy scripts:

1. Confirm no external usage in your ops/runbooks.
2. If needed, migrate required functionality into a maintained CLI entrypoint.
3. Remove script + add changelog note in the same PR.
