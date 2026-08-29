# Bayesian Logistic Full-GPU R5 RED evidence

This directory freezes the local, non-GPU TDD gates for the R5 acceptance bundle. Production source was not edited or executed on a GPU.

## Authority

- Original R4 authority: `45c64ccacd3c42d5bd07cccbf8985020b931ef47`
- Reviewed production base: `b114c01b036820e42c90ef6c10fa7aa35a1838de`
- Pre-freeze branch HEAD: `7c99c6db6ecd1e8c7d08e0accf84c9f216b5a242`
- Pre-freeze branch tree: `6d35d6ed564ba36132850c0397f99415a71878ac`
- Production diff count versus `b114c01` across `crates/neoethos-models/src`, `crates/neoethos-core/src`, and `crates/neoethos-execution-budget/src`: **0**
- Every Cargo command used `--locked --offline -j7`, `CARGO_INCREMENTAL=0`, and `RUSTFLAGS=-Dwarnings`.
- No GPU, network, VPS, registry, production run, or paid-card action occurred.

## Final local classifications

1. Support validators: exit 0; **7 passed, 0 failed, 0 ignored**.
2. Public lifecycle contract without `--ignored`: exit 0; **1 passed, 0 failed, 2 ignored**. The GPU lifecycle and private child remained unrun.
3. First embargo attempt: stopped before tests on unexpected test-harness `E0621`. The raw failure is preserved. The compiler-requested one-line test-only lifetime correction was then applied.
4. Corrected embargo behavior RED: **2 passed, 4 intended failed, 0 ignored**. The four failures are the typed four-way schema, all-three producer census, all-three validator census, and global alias-aware constructor inventory.
5. Acceptance compile RED: exactly one `E0599` at `BudgetedCpuExecutor::execute_with_lease(lease.into_transfer(), ...)`; no other warning/error family. The ignored RTX parent and private worker never executed.

The Windows tool reports process exit code 1 for the two expected RED commands. Classification is based on the complete raw compiler/test output, exact test counts, and diagnostic census rather than assuming a Unix Cargo exit code.

## Offline vendor closure

- Tracked files: **6,184**
- Total bytes: **188,376,499**
- Ignored or ordinary untracked vendor inputs: **0**
- Per-file ledger: `vendor-closure.sha256`
- Ledger SHA-256: `89d32887db055aa803b71d41f7b4c49ab152e62bffb58f08b209e91a29b683df`

## Evidence boundary

The raw logs include the exact command, target, environment, complete combined stdout/stderr, observed exit status, and classification. `manifest.sha256` hashes the source and evidence files that define this freeze, excluding itself. The final immutable Git commit/tree are to be recorded after review; the manifest does not hard-code a preimplementation authority as the tested implementation.
