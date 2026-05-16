# Full Audit Continuation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `origin/claude/v0.4.1-full-audit` from a compileable WIP branch into an evidence-backed, mergeable branch by auditing claims against code and fixing only verified gaps.

**Architecture:** Start with a claim-to-code matrix, then execute small TDD fixes against proven findings. Shared helpers are introduced only where duplicated production logic is demonstrated across files.

**Tech Stack:** Rust workspace, Cargo, PowerShell, ripgrep, Git worktrees, existing `forex-*` crates.

---

## File Structure

- Create: `docs/audits/codex_full_audit_continuation_gap_report.md`
  - Owns the current claim-to-code matrix and priority queue.
- Modify only as findings require:
  - `crates/forex-core/src/**` for domain/risk/config duplication or missing behavior.
  - `crates/forex-app/src/**` for wizard, HALT, cTrader, trading, and UI-runtime gaps.
  - `crates/forex-cli/src/**` for CLI/TUI wizard gaps.
  - `crates/forex-search/src/**` and `crates/forex-models/src/**` for audit-proven model/search issues.
  - packaging/workflow files only if a release-gate claim is false.

## Chunk 1: Claim Inventory And Gap Report

### Task 1: Build The Evidence Matrix

**Files:**
- Create: `docs/audits/codex_full_audit_continuation_gap_report.md`

- [ ] **Step 1: Extract branch claims**

Run:

```powershell
rg -n "Status:|Gate:|FIXED|COMPLETE|DEFERRED|in flight|scaffold|TODO|FIXME|ship gate|release gate" docs/v0.5_roadmap.md docs/audits -g "*.md"
git log --oneline origin/master..HEAD
```

Expected: list of claimed deliverables, WIP markers, and release gates.

- [ ] **Step 2: Map claims to code**

Run targeted searches for HALT/Risky Mode, wizard, cTrader, packaging, real-data fixtures, ignored tests, and hardcoded config.

- [ ] **Step 3: Write the initial gap report**

Record each item with columns:

```markdown
| Area | Claim | Evidence | Status | Next action |
| --- | --- | --- | --- | --- |
```

- [ ] **Step 4: Verify no code changed except report**

Run:

```powershell
git status --short
```

Expected: only the report plus planning docs are modified.

## Chunk 2: Baseline Mechanical Health

### Task 2: Establish Mergeability Baseline

**Files:**
- Modify only files reported by verification commands.

- [ ] **Step 1: Run compile baseline**

Run:

```powershell
cargo check --workspace --all-targets --locked
```

Expected: command exits 0. Warnings are captured in the gap report if they indicate disconnected implementation.

- [ ] **Step 2: Run diff hygiene check**

Run:

```powershell
git diff --check origin/claude/v0.4.1-full-audit...HEAD
```

Expected: fail currently if whitespace problems remain.

- [ ] **Step 3: Fix only mechanical diff hygiene**

This is exempt from TDD because it is formatting-only. Remove trailing whitespace and final blank-line issues reported by `git diff --check`.

- [ ] **Step 4: Verify hygiene**

Run the same `git diff --check` command. Expected: no output, exit 0.

## Chunk 3: First Verified Code Gap

### Task 3: Pick The Highest-Risk Proven Gap

**Files:**
- Determined by the gap report.

- [ ] **Step 1: Choose one finding**

Prefer the first item that is both high risk and locally testable: ignored `unimplemented!`, disconnected wizard path, duplicated hardcoded risk logic, or stale claim.

- [ ] **Step 2: Write or un-ignore a failing test**

Run a focused `cargo test` command and capture the expected failure.

- [ ] **Step 3: Implement the smallest fix**

Touch only the files needed for that finding.

- [ ] **Step 4: Verify focused tests pass**

Run the exact focused test command again.

- [ ] **Step 5: Run affected crate check**

Run:

```powershell
cargo check -p <crate> --all-targets --locked
```

Expected: exit 0.

## Chunk 4: Repeatable Audit Loop

### Task 4: Continue Findings One At A Time

**Files:**
- Determined per finding.

- [ ] **Step 1: Re-rank remaining findings**

Update `docs/audits/codex_full_audit_continuation_gap_report.md`.

- [ ] **Step 2: Apply the TDD loop to the next finding**

Repeat red, green, focused verification, and affected crate check.

- [ ] **Step 3: Stop when the next finding needs external data or user decision**

Document fixture requirements instead of faking real broker data.

## Final Verification

- [ ] Run `cargo check --workspace --all-targets --locked`.
- [ ] Run `git diff --check origin/claude/v0.4.1-full-audit...HEAD`.
- [ ] Run focused tests for every changed crate area.
- [ ] Update the gap report with remaining known limits.
