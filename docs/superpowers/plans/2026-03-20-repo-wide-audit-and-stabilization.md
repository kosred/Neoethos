# Repo-Wide Audit And Stabilization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete evidence-first audit of the current Rust-first repository, identify all active warnings/errors/runtime regressions, and leave the project with a verified findings ledger and a prioritized stabilization backlog.

**Architecture:** The audit runs in four layers: repository census, static verification, real-path runtime validation, and line-by-line code review. Every finding must be written into a shared audit report with file references, evidence, severity, and root-cause notes before any broad stabilization refactor begins.

**Tech Stack:** Rust workspace (`cargo`, `clippy`, `cargo test`), existing app/CLI binaries, PowerShell command automation, targeted web verification against official docs when behavior or APIs are uncertain.

---

## File Structure

- Create: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Create: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Create: `cache/audit/2026-03-20-file-manifest.txt`
- Create: `cache/audit/2026-03-20-command-log.txt`
- Create: `cache/audit/2026-03-20-findings.jsonl`
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`

## Chunk 1: Evidence Baseline And Findings Ledger

### Task 1: Create the audit ledger and verification matrix

**Files:**
- Create: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Create: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Create: `cache/audit/2026-03-20-command-log.txt`
- Create: `cache/audit/2026-03-20-findings.jsonl`

- [ ] **Step 1: Create the report skeleton**

Add sections for:
- Executive Summary
- Repository Census
- Build Findings
- Runtime Findings
- File-By-File Findings
- Contract Findings
- Warning Inventory
- Recommended Fix Tranches

- [ ] **Step 2: Create the verification matrix skeleton**

Add one row placeholder section for each critical path:
- workspace build
- workspace tests
- clippy
- CLI load/features/prepare
- CLI discover/train
- app headless local mode
- app GUI startup smoke
- MT5 bridge contract smoke

- [ ] **Step 3: Create the command log and findings ledger files**

Run:
```powershell
New-Item -ItemType Directory -Force cache/audit | Out-Null
New-Item -ItemType Directory -Force docs/superpowers/reports | Out-Null
Set-Content cache/audit/2026-03-20-command-log.txt ''
Set-Content cache/audit/2026-03-20-findings.jsonl ''
```
Expected: files exist and are empty.

- [ ] **Step 4: Commit the audit scaffolding**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md cache/audit/2026-03-20-command-log.txt cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: scaffold repo audit ledger"
```
Expected: one commit containing only the audit scaffolding files.

### Task 2: Generate the repository census manifest

**Files:**
- Create: `cache/audit/2026-03-20-file-manifest.txt`
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `cache/audit/2026-03-20-command-log.txt`

- [ ] **Step 1: Generate the current file manifest**

Run:
```powershell
@(
  '## Rust'
  (rg --files . -g "*.rs")
  '## Python'
  (rg --files . -g "*.py")
  '## TOML'
  (rg --files . -g "*.toml")
  '## YAML'
  (rg --files . -g "*.yaml" -g "*.yml")
  '## JSON'
  (rg --files . -g "*.json")
) | Set-Content cache/audit/2026-03-20-file-manifest.txt
```
Expected: one manifest file containing the current repository surface grouped by file type.

- [ ] **Step 2: Record the high-level census counts**

Run:
```powershell
"rust=$((rg --files . -g '*.rs').Count)" | Add-Content cache/audit/2026-03-20-command-log.txt
"python=$((rg --files . -g '*.py').Count)" | Add-Content cache/audit/2026-03-20-command-log.txt
"toml=$((rg --files . -g '*.toml').Count)" | Add-Content cache/audit/2026-03-20-command-log.txt
```
Expected: command log contains current file counts.

- [ ] **Step 3: Write the census summary into the audit report**

Add:
- workspace members
- binaries/entrypoints
- UI crate
- MT5 bridge crate
- Python-dependent crates and bindings
- remaining Python/shim presence
- test surface size

- [ ] **Step 4: Add the subsystem classification matrix**

Write one table row per subsystem and mark it as:
- `runtime-critical`
- `static-only`
- `audit-required bridge`
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

- [ ] **Step 5: Commit the census baseline**

Run:
```powershell
git add cache/audit/2026-03-20-file-manifest.txt cache/audit/2026-03-20-command-log.txt docs/superpowers/reports/2026-03-20-repo-audit-report.md
git commit -m "docs: record repo audit census"
```
Expected: one commit with the repository census artifacts.

### Task 3: Capture the static verification baseline

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Modify: `cache/audit/2026-03-20-command-log.txt`
- Modify: `cache/audit/2026-03-20-findings.jsonl`

- [ ] **Step 1: Run workspace build verification**

Before executing, write the supported verification lanes into the verification matrix:
- `baseline-windows`
- `baseline-linux` if reviewable without execution
- `python-contract`
- `optional-informational-heavy-features`

Run:
```powershell
cargo check --workspace
```
Expected: either exit `0` or concrete compiler findings captured.

- [ ] **Step 2: Run workspace tests**

Run:
```powershell
cargo test --workspace
```
Expected: either exit `0` or concrete failing tests captured.

- [ ] **Step 3: Run workspace clippy under warning denial**

Run:
```powershell
cargo clippy --workspace --all-targets -- -D warnings
```
Expected: either exit `0` or a full warning/error inventory.

- [ ] **Step 4: Run Python-contract validation for active runtime seams**

Run the minimal checks required to prove currently active Python-dependent contracts are present or explicitly broken, for example:
```powershell
python -c "import sys; print(sys.version)"
python -c "import forex_bindings"
python -c "import MetaTrader5"
```
Expected: explicit success, explicit failure, or a documented `BLOCKED` state where the environment is not expected to provide the dependency.

- [ ] **Step 5: Run informational heavy-feature lanes separately if needed**

Examples:
```powershell
cargo check -p forex-models
cargo check -p forex-search
```
Expected: findings labeled `informational` unless they affect the supported baseline.

- [ ] **Step 6: Record every failure or warning into the findings ledger**

For each issue, append one JSON line with:
- `category`
- `severity`
- `lane`
- `command`
- `file`
- `line`
- `summary`
- `evidence`

- [ ] **Step 7: Update the verification matrix and audit report**

Mark each command as:
- PASS
- FAIL
- PASS WITH FINDINGS
- BLOCKED
- N/A

- [ ] **Step 8: Commit the static verification results**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md cache/audit/2026-03-20-command-log.txt cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: capture static verification baseline"
```
Expected: one commit containing only evidence and report updates.

## Chunk 2: Runtime Sweep, Line-By-Line Audit, And Stabilization Triage

### Task 4: Execute the real-path runtime sweep

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`
- Modify: `cache/audit/2026-03-20-command-log.txt`
- Modify: `cache/audit/2026-03-20-findings.jsonl`

- [ ] **Step 1: Run the CLI data path smoke checks**

Before executing, write prerequisites and timeout rules into the verification matrix:
- command
- OS
- required files/config
- required services/modules
- timeout
- result state

Run:
```powershell
cargo run -p forex-cli -- symbols --root data
cargo run -p forex-cli -- timeframes --root data --symbol EURUSD
cargo run -p forex-cli -- load --root data --symbol EURUSD --timeframe M1
cargo run -p forex-cli -- features --root data --symbol EURUSD --timeframe M1
```
Expected: successful command output or concrete reproducible failures.

- [ ] **Step 2: Run the CLI preparation and discovery path**

Run:
```powershell
cargo run -p forex-cli -- prepare --root data --symbol EURUSD --base M1 --higher M5,M15,H1
cargo run -p forex-cli -- discover --root data --symbol EURUSD --base M1 --higher M5,M15,H1 --population 10 --generations 1 --candidates 20 --portfolio-size 10
```
Expected: either successful preparation/discovery output or concrete contract/runtime failures.

- [ ] **Step 3: Run the CLI training path**

Run:
```powershell
cargo run -p forex-cli -- train --root data --symbol EURUSD --base M1 --config config.yaml --models-dir models
```
Expected: either successful training startup/completion or concrete reproducible failures.

- [ ] **Step 4: Run the app headless local path**

Run:
```powershell
cargo run -p forex-app -- --headless --local --config config.yaml
```
Expected: successful startup and steady keep-alive behavior, or a concrete failure with logs.

- [ ] **Step 5: Run the app GUI smoke path**

Run:
```powershell
cargo run -p forex-app -- --local --config config.yaml
```
Expected: successful startup to a visible window, or a concrete startup failure.

- [ ] **Step 6: Run the MT5 bridge smoke path**

Use the app trading tab connection flow or a targeted minimal probe if startup alone is insufficient.
Expected: explicit success, explicit graceful offline behavior, a `BLOCKED` result because prerequisites are missing, or concrete bridge failure details.

- [ ] **Step 7: Record every runtime finding and update the verification matrix**

No runtime issue is “known” unless:
- the command is written down
- the exact observed result is captured
- the finding is added to the ledger

- [ ] **Step 8: Commit the runtime evidence**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md cache/audit/2026-03-20-command-log.txt cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: capture runtime audit evidence"
```
Expected: one commit containing runtime evidence updates only.

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

Record mismatches, stale deps, feature-gate drift, and startup inconsistencies.

- [ ] **Step 2: Audit app and MT5 boundary files**

Review all files listed in the manifest under:
- `crates/forex-app/src/`
- `crates/mt5-bridge/src/`

Record UI/backend contract gaps, async issues, MT5 bridge assumptions, and warning-prone code.

- [ ] **Step 3: Audit core/data/search/model crates**

Review all files listed in the manifest under:
- `crates/forex-core/src/`
- `crates/forex-data/src/`
- `crates/forex-search/src/`
- `crates/forex-models/src/`
- `crates/forex-news/src/`
- `crates/forex-cli/src/`
- `crates/forex-bindings/src/`

Record migration leftovers, contract mismatches, silent fallbacks, and error-handling gaps.

- [ ] **Step 4: Audit remaining Python/shim files**

Review every remaining active Python file listed in the manifest.
For each file, classify it as:
- active runtime code
- temporary bridge
- dead/reachable compatibility seam
- config/bootstrap only

- [ ] **Step 5: Verify uncertain API or platform assumptions online**

For any uncertain behavior, check official docs before writing the finding as actionable.
Record the source URL next to the finding.

- [ ] **Step 6: Group findings by severity and subsystem**

The report must end this task with:
- Critical
- Important
- Minor

Each group must be subdivided by subsystem.

- [ ] **Step 7: Commit the completed audit report**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md cache/audit/2026-03-20-findings.jsonl
git commit -m "docs: complete repo line-by-line audit report"
```
Expected: one commit containing the complete findings ledger and report updates.

### Task 6: Produce the stabilization backlog and execution handoff

**Files:**
- Modify: `docs/superpowers/reports/2026-03-20-repo-audit-report.md`
- Modify: `docs/superpowers/reports/2026-03-20-verification-matrix.md`

- [ ] **Step 1: Convert findings into fix tranches**

Create tranches in this order:
- build blockers
- runtime blockers
- warnings and lint cleanup
- contract repairs
- observability/checkpoint upgrades
- structural cleanup candidates

- [ ] **Step 2: Add explicit entry criteria for the UI integration phase**

Define what must be true before UI/backend integration starts:
- stable app startup
- stable engine contract
- stable MT5 bridge behavior
- no unresolved critical runtime findings

- [ ] **Step 3: Add a short “recommended next execution order” section**

This section must name the first concrete fix tranche to execute after the audit.

- [ ] **Step 4: Commit the stabilization handoff**

Run:
```powershell
git add docs/superpowers/reports/2026-03-20-repo-audit-report.md docs/superpowers/reports/2026-03-20-verification-matrix.md
git commit -m "docs: add stabilization backlog and handoff"
```
Expected: one commit containing the prioritized stabilization handoff.
