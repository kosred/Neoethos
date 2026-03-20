# Repo-Wide Audit And Stabilization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete evidence-first audit of the current Rust-first repository, identify active warnings/errors/runtime regressions, and leave the project with a verified findings ledger, verification matrix, and prioritized stabilization backlog.

**Architecture:** The audit runs in five layers: repository census, static verification, real-path runtime validation, line-by-line subsystem review, and contract/operational hardening review. Every finding must be recorded with evidence, severity, root cause, and recommended fix direction before stabilization work begins.

**Tech Stack:** Rust workspace tooling (`cargo`, `clippy`, `cargo test`, `cargo build`), existing app/CLI binaries, PowerShell command automation, targeted Python contract probes where Python is still runtime-relevant, and official online documentation when any API or platform behavior is uncertain.

---

## File Structure

- Create: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Create: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Create: `docs/superpowers/reports/2026-03-20-stabilization-backlog.md`
- Create: `cache/audit/2026-03-20-file-manifest.txt`
- Create: `cache/audit/2026-03-20-command-log.txt`
- Create: `cache/audit/2026-03-20-findings.jsonl`

## Chunk 1: Audit Artifacts, Census, And Static Baseline

### Task 1: Create the audit artifacts and schemas

**Files:**
- Create: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Create: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Create: `docs/superpowers/reports/2026-03-20-stabilization-backlog.md`
- Create: `cache/audit/2026-03-20-command-log.txt`
- Create: `cache/audit/2026-03-20-findings.jsonl`

- [ ] **Step 1: Create the report skeleton**

Create sections:
- Executive Summary
- Repository Census
- Static Verification Findings
- Runtime Findings
- File-By-File Findings
- Contract And Operational Findings
- Warning Inventory
- Recommended Fix Tranches

- [ ] **Step 2: Create the verification matrix skeleton**

Create columns/sections for:
- lane
- command or probe
- expected outcome
- prerequisites
- timeout
- environment
- result state
- evidence location

Required lanes:
- `baseline-windows`
- `baseline-linux`
- `python-contract`
- `optional-informational-heavy-features`
- `runtime-gui`
- `runtime-headless`
- `runtime-mt5`

- [ ] **Step 3: Create the stabilization backlog skeleton**

Create sections:
- Build Blockers
- Runtime Blockers
- Warning Cleanup
- Contract Repairs
- Observability And Recovery
- Structural Cleanup Candidates
- UI Integration Entry Criteria

- [ ] **Step 4: Create every audit artifact on disk**

Run:
```powershell
New-Item -ItemType Directory -Force cache/audit | Out-Null
New-Item -ItemType Directory -Force docs/superpowers/reports | Out-Null
Set-Content docs/superpowers/reports/2026-03-20-repo-audit-report.md ''
Set-Content docs/superpowers/reports/2026-03-20-verification-matrix.md ''
Set-Content docs/superpowers/reports/2026-03-20-stabilization-backlog.md ''
Set-Content cache/audit/2026-03-20-command-log.txt ''
Set-Content cache/audit/2026-03-20-findings.jsonl ''
```
Expected: all six audit artifact files exist on disk.

- [ ] **Step 5: Verify every audit artifact contains the required structure**

Verify before the first commit:
- `2026-03-20-repo-audit-report.md` contains all required report sections
- `2026-03-20-verification-matrix.md` contains all required lanes and the `expected outcome` field
- `2026-03-20-stabilization-backlog.md` contains all required backlog sections
- `2026-03-20-command-log.txt` and `2026-03-20-findings.jsonl` exist and are writable

- [ ] **Step 6: Define the findings ledger schema**

Document that each JSON line must include:
- `category`
- `severity`
- `lane`
- `command`
- `file`
- `line`
- `summary`
- `evidence`
- `root_cause`
- `recommended_fix`

Allowed `category` values:
- `build breakage`
- `test failure`
- `lint/warning`
- `runtime breakage`
- `correctness bug`
- `contract mismatch`
- `dead or unreachable code`
- `observability gap`
- `performance risk`
- `architectural smell`

- [ ] **Step 7: Commit the audit scaffolding**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md docs/superpowers/reports/2026-03-20-stabilization-backlog.md cache/audit/2026-03-20-command-log.txt cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: scaffold repo audit artifacts"
```
Expected: one commit containing only the audit artifact files.

### Task 2: Generate the repository census and classification matrices

**Files:**
- Create: `cache/audit/2026-03-20-file-manifest.txt`
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `cache/audit/2026-03-20-command-log.txt`

- [ ] **Step 1: Generate the first-party tracked file manifest**

Run:
```powershell
git ls-files | Set-Content cache/audit/2026-03-20-file-manifest.txt
```
Expected: manifest contains the tracked repository surface.

- [ ] **Step 2: Record explicit excluded classes**

Add to the audit report:
- `vendor/`
- `cache/`
- `logs/`
- `target/`
- generated local output files

Each must be marked as `excluded` or `evidence-only` with a reason.

- [ ] **Step 3: Record high-level census counts**

Run:
```powershell
$tracked = @(git ls-files)
"tracked=$($tracked.Count)" | Add-Content cache/audit/2026-03-20-command-log.txt
"rust=$((@($tracked | Where-Object { $_ -like '*.rs' })).Count)" | Add-Content cache/audit/2026-03-20-command-log.txt
"python=$((@($tracked | Where-Object { $_ -like '*.py' })).Count)" | Add-Content cache/audit/2026-03-20-command-log.txt
"toml=$((@($tracked | Where-Object { $_ -like '*.toml' })).Count)" | Add-Content cache/audit/2026-03-20-command-log.txt
```
Expected: command log contains the census counts.

- [ ] **Step 4: Add the subsystem classification matrix**

Write one row per subsystem using only these statuses:
- `runtime-critical`
- `audit-required`
- `evidence-only`
- `excluded`

Required rows:
- `crates/forex-app`
- `crates/forex-cli`
- `crates/mt5-bridge`
- `crates/forex-core`
- `crates/forex-data`
- `crates/forex-search`
- `crates/forex-models`
- `crates/forex-bindings`
- `crates/forex-news`
- remaining Python files
- top-level config/script/service assets
- `tests/`
- `examples/`
- `vendor/`
- generated/cache/log/build output directories

- [ ] **Step 5: Add the active-file classification matrix**

Write one row per file class using only these statuses:
- `runtime-critical`
- `static-only`
- `audit-required bridge`
- `evidence-only`
- `excluded`

- [ ] **Step 6: Record workspace members, binaries, and Python-dependent seams**

Add to the audit report:
- workspace members from `Cargo.toml`
- binaries and entrypoints
- Python-dependent crates and bindings
- UI crate
- MT5 bridge crate
- remaining Python/shim presence
- test and examples surface

- [ ] **Step 7: Commit the census baseline**

Run:
```powershell
git add cache/audit/2026-03-20-file-manifest.txt cache/audit/2026-03-20-command-log.txt docs/superpowers/reports/2026-03-20-repo-audit-report.md
git commit -m "docs: record repo audit census"
```
Expected: one commit with the census artifacts only.

### Task 3: Capture the static verification baseline

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Modify: `cache/audit/2026-03-20-command-log.txt`
- Modify: `cache/audit/2026-03-20-findings.jsonl`

- [ ] **Step 1: Add the baseline verification lanes to the matrix**

Define:
- `baseline-windows`: supported local developer/runtime profile on the current OS
- `baseline-linux`: review-only or run elsewhere if available
- `python-contract`: active Python-dependent runtime contracts
- `optional-informational-heavy-features`: heavyweight optional integrations

- [ ] **Step 2: Run workspace check**

Run:
```powershell
cargo check --workspace
```
Expected: exit `0` or captured findings.

- [ ] **Step 3: Immediately record the workspace-check result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `baseline-windows` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 4: Run workspace tests**

Run:
```powershell
cargo test --workspace
```
Expected: exit `0` or captured findings.

- [ ] **Step 5: Immediately record the workspace-test result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `baseline-windows` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 6: Run clippy on the supported baseline**

Run:
```powershell
cargo clippy --workspace --all-targets -- -D warnings
```
Expected: exit `0` or captured findings.

- [ ] **Step 7: Immediately record the clippy result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `baseline-windows` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 8: Run the `forex-cli` binary build**

Run:
```powershell
cargo build -p forex-cli
```
Expected: successful link/build or captured findings.

- [ ] **Step 9: Immediately record the `forex-cli` build result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant verification-matrix row
- append any findings from this command to the findings ledger

- [ ] **Step 10: Run the `forex-app` binary build**

Run:
```powershell
cargo build -p forex-app
```
Expected: successful link/build or captured findings.

- [ ] **Step 11: Immediately record the `forex-app` build result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant verification-matrix row
- append any findings from this command to the findings ledger

- [ ] **Step 12: Run the Python version probe**

Run:
```powershell
python -c "import sys; print(sys.version)"
```
Expected: explicit success or captured failure.

- [ ] **Step 13: Immediately record the Python version result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `python-contract` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 14: Run the `forex_bindings` import probe**

Run:
```powershell
python -c "import forex_bindings"
```
Expected: explicit success, explicit failure, or documented `BLOCKED`.

- [ ] **Step 15: Immediately record the `forex_bindings` import result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `python-contract` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 16: Run the `MetaTrader5` import probe**

Run:
```powershell
python -c "import MetaTrader5"
```
Expected: explicit success, explicit failure, or a documented `BLOCKED` state where the environment is not expected to provide the dependency.

- [ ] **Step 17: Immediately record the `MetaTrader5` import result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `python-contract` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 18: Run the `forex-models` informational lane if needed**

Run:
```powershell
cargo check -p forex-models
```
Expected: findings marked `informational` unless they affect the supported baseline.

- [ ] **Step 19: Immediately record the `forex-models` informational result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `optional-informational-heavy-features` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 20: Run the `forex-search` informational lane if needed**

Run:
```powershell
cargo check -p forex-search
```
Expected: findings marked `informational` unless they affect the supported baseline.

- [ ] **Step 21: Immediately record the `forex-search` informational result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `optional-informational-heavy-features` row in the verification matrix
- append any findings from this command to the findings ledger

- [ ] **Step 22: Verify the findings ledger schema was respected**

For every issue, ensure each JSON line includes:
- category
- severity
- lane
- command
- file
- line
- summary
- evidence
- root_cause
- recommended_fix

- [ ] **Step 23: Verify the matrix uses only allowed result states**

Allowed states:
- `PASS`
- `FAIL`
- `PASS WITH FINDINGS`
- `BLOCKED`
- `N/A`

- [ ] **Step 24: Verify every command entry in the command log is complete**

Each command entry must include:
- timestamp
- lane
- command
- exit status
- output location or inline summary

- [ ] **Step 25: Commit the static verification baseline**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md cache/audit/2026-03-20-command-log.txt cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: capture static verification baseline"
```
Expected: one commit containing the static verification evidence only.

## Chunk 2: Runtime Sweep, File Audit, And Contract Review

### Task 4: Execute the real-path runtime sweep

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Modify: `cache/audit/2026-03-20-command-log.txt`
- Modify: `cache/audit/2026-03-20-findings.jsonl`

- [ ] **Step 1: Define runtime probe prerequisites and timeouts**

For every runtime probe, write:
- command or manual probe
- OS
- required files/config
- required services/modules
- timeout
- expected environment
- result state

- [ ] **Step 2: Run the `symbols` CLI data probe**

Run:
```powershell
cargo run -p forex-cli -- symbols --root data
```
Expected: successful output or concrete failures.

- [ ] **Step 3: Immediately record the `symbols` probe result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant runtime row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 4: Run the `timeframes` CLI data probe**

Run:
```powershell
cargo run -p forex-cli -- timeframes --root data --symbol EURUSD
```
Expected: successful output or concrete failures.

- [ ] **Step 5: Immediately record the `timeframes` probe result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant runtime row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 6: Run the `load` CLI data probe**

Run:
```powershell
cargo run -p forex-cli -- load --root data --symbol EURUSD --timeframe M1
```
Expected: successful output or concrete failures.

- [ ] **Step 7: Immediately record the `load` probe result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant runtime row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 8: Run the `features` CLI data probe**

Run:
```powershell
cargo run -p forex-cli -- features --root data --symbol EURUSD --timeframe M1
```
Expected: successful output or concrete failures.

- [ ] **Step 9: Immediately record the `features` probe result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant runtime row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 10: Run the `prepare` CLI runtime probe**

Run:
```powershell
cargo run -p forex-cli -- prepare --root data --symbol EURUSD --base M1 --higher M5,M15,H1
```
Expected: successful output or concrete failures.

- [ ] **Step 11: Immediately record the `prepare` probe result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant runtime row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 12: Run the `discover` CLI runtime probe**

Run:
```powershell
cargo run -p forex-cli -- discover --root data --symbol EURUSD --base M1 --higher M5,M15,H1 --population 10 --generations 1 --candidates 20 --portfolio-size 10
```
Expected: successful output or concrete failures.

- [ ] **Step 13: Immediately record the `discover` probe result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant runtime row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 14: Run the CLI training probe**

Run:
```powershell
cargo run -p forex-cli -- train --root data --symbol EURUSD --base M1 --config config.yaml --models-dir models
```
Expected: successful startup/completion or concrete failures.

- [ ] **Step 15: Immediately record the CLI training result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the relevant runtime row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 16: Run the app headless local path**

Run:
```powershell
cargo run -p forex-app -- --headless --local --config config.yaml
```
Expected: successful startup and keep-alive, or concrete failures.

- [ ] **Step 17: Immediately record the headless-app result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `runtime-headless` row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 18: Run the app GUI smoke path**

Run:
```powershell
cargo run -p forex-app -- --local --config config.yaml
```
Expected: `PASS`, `FAIL`, or `BLOCKED` if the current session cannot support GUI startup.

- [ ] **Step 19: Immediately record the GUI-app result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `runtime-gui` row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 20: Run the MT5 prerequisite probe**

Run:
```powershell
python -c "import MetaTrader5 as mt5; ok = mt5.initialize(); print({'initialize': ok, 'terminal_info': str(mt5.terminal_info()) if ok else None, 'last_error': None if ok else mt5.last_error()}); mt5.shutdown() if ok else None"
```
Expected: explicit success, explicit failure, or `BLOCKED` if the environment does not provide MT5.

- [ ] **Step 21: Immediately record the MT5 prerequisite result**

Before moving on:
- append the command, timestamp, exit status, and output summary to `cache/audit/2026-03-20-command-log.txt`
- update the `runtime-mt5` row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 22: Run the MT5 bridge contract probe**

Probe:
- launch `forex-app` in MT5 mode on Windows
- use the Trading tab connect action
- capture whether status changes and whether terminal-info is surfaced

Expected: `PASS`, `FAIL`, or `BLOCKED` with explicit reason. A graceful offline/not-available path is not a failure if it is handled explicitly and consistently.

- [ ] **Step 23: Immediately record the MT5 bridge-contract result**

Before moving on:
- append the probe description, timestamp, result state, and evidence summary to `cache/audit/2026-03-20-command-log.txt`
- update the `runtime-mt5` row in the verification matrix
- append any findings from this probe to the findings ledger

- [ ] **Step 24: Verify no runtime issue is claimed without evidence**

No runtime issue is known unless:
- the exact probe is recorded
- the observed result is captured
- the result state is assigned
- the finding is written to the ledger

- [ ] **Step 25: Commit the runtime evidence**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md cache/audit/2026-03-20-command-log.txt cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: capture runtime audit evidence"
```
Expected: one commit containing runtime evidence only.

### Task 5: Perform the line-by-line code audit by subsystem

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `cache/audit/2026-03-20-findings.jsonl`
- Inspect: `cache/audit/2026-03-20-file-manifest.txt`

- [ ] **Step 1: Audit workspace roots and manifests**

Review:
- `Cargo.toml`
- all crate `Cargo.toml` files
- `crates/forex-bindings/pyproject.toml`
- `config.yaml`
- service and script wrappers

- [ ] **Step 2: Audit `crates/forex-app/src/`**

Record UI runtime assumptions, async behavior, message flow issues, and startup/warning concerns.

- [ ] **Step 3: Audit `crates/mt5-bridge/src/`**

Record Python contract assumptions, error handling, bridge lifecycle issues, and MT5 failure visibility gaps.

- [ ] **Step 4: Audit `crates/forex-core/src/`**

Record stale comments, contract mismatches, utility drift, and warning-prone code.

- [ ] **Step 5: Audit `crates/forex-data/src/`**

Record data-loading assumptions, resampling/feature-prep issues, and runtime contract gaps.

- [ ] **Step 6: Audit `crates/forex-search/src/`**

Record discovery/search correctness risks, feature-gate assumptions, and warning-prone code.

- [ ] **Step 7: Audit `crates/forex-models/src/`**

Record model orchestration issues, Python dependency seams, error handling gaps, and performance/safety risks.

- [ ] **Step 8: Audit `crates/forex-bindings/src/`**

Record FFI contract issues, PyO3 lifecycle assumptions, and silent failure risks.

- [ ] **Step 9: Audit `crates/forex-news/src/` and `crates/forex-cli/src/`**

Record command-surface issues, integration mismatches, and stale migration assumptions.

- [ ] **Step 10: Audit remaining Python and shim files**

For each remaining Python file, classify it as:
- active runtime code
- temporary bridge
- dead/reachable compatibility seam
- config/bootstrap only

- [ ] **Step 11: Audit `tests/` and `examples/` as evidence-bearing assets**

Look for:
- stale assumptions
- missing coverage on active paths
- examples that no longer match runtime reality

- [ ] **Step 12: Verify uncertain API or platform assumptions online**

For any uncertain behavior, use official documentation and record the source URL next to the finding.

- [ ] **Step 13: Group findings by severity and subsystem**

The report must end this task with:
- Critical
- Important
- Minor

Each group must be subdivided by subsystem.

- [ ] **Step 14: Commit the line-by-line audit findings**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: complete repo line-by-line audit report"
```
Expected: one commit containing the completed findings ledger and report updates.

### Task 6: Perform the contract and operational hardening audit

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-stabilization-backlog.md`

- [ ] **Step 1: Audit CLI contracts**

Check:
- argument surfaces
- config expectations
- failure visibility
- reproducibility of commands

- [ ] **Step 2: Audit app-to-engine contracts**

Check:
- app state ownership
- UI action to runtime action mapping
- progress/result propagation
- explicit failure behavior

- [ ] **Step 3: Audit MT5 bridge operational behavior**

Check:
- initialize/shutdown lifecycle
- terminal-info surfacing
- explicit offline behavior
- dependency and environment visibility

- [ ] **Step 4: Audit logging, metrics, and progress reporting**

Check:
- actionable log messages
- hidden failures
- missing progress or checkpoint visibility

- [ ] **Step 5: Audit checkpointing and recovery points**

Check:
- recoverable long-running operations
- restart expectations
- persistence boundaries

- [ ] **Step 6: Record contract and operational findings**

Every finding must still include:
- evidence
- root cause
- recommended fix

- [ ] **Step 7: Commit the contract/operational audit**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-stabilization-backlog.md
git commit -m "docs: record contract and operational audit findings"
```
Expected: one commit containing only contract/operational audit updates.

## Chunk 3: Stabilization Backlog And Execution Handoff

### Task 7: Produce the stabilization backlog and handoff

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Modify: `docs/superpowers/reports/2026-03-20-stabilization-backlog.md`

- [ ] **Step 1: Convert findings into fix tranches**

Create tranches in this order:
- build blockers
- runtime blockers
- warning and lint cleanup
- contract repairs
- observability and recovery upgrades
- structural cleanup candidates

- [ ] **Step 2: Define entry criteria for UI integration**

The UI phase must not start until:
- app startup is stable
- backend contracts are explicit
- MT5 bridge behavior is verified or explicitly blocked by environment only
- no critical runtime findings remain unresolved

- [ ] **Step 3: Name the first concrete fix tranche**

This section must identify the first stabilization execution step after the audit, with rationale.

- [ ] **Step 4: Cross-check the backlog against the audit report**

Every critical or important finding must either:
- appear in the backlog
- be explicitly deferred with reason

- [ ] **Step 5: Commit the stabilization handoff**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md docs/superpowers/reports/2026-03-20-stabilization-backlog.md
git commit -m "docs: add stabilization backlog and handoff"
```
Expected: one commit containing the prioritized stabilization handoff.
