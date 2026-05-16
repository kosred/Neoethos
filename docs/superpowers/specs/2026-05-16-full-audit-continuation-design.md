# Full Audit Continuation Design

**Date:** 2026-05-16
**Base branch:** `origin/claude/v0.4.1-full-audit`
**Continuation branch:** `codex/full-audit-continuation`

## Goal

Continue from the Opus 4.7 `full-audit` branch without assuming that every claimed fix is complete. The work starts with evidence: compare the audit and roadmap claims against code, tests, and duplication patterns, then make only targeted changes where a concrete gap, bug, or duplicate implementation is proven.

## Scope

This pass covers the current `v0.4.1-full-audit` branch state:

- audit and roadmap claims in `docs/v0.5_roadmap.md` and `docs/audits/`
- Rust implementation under `crates/`
- packaging and workflow scaffolds where the audit claims shipping readiness
- TODO/FIXME/WIP markers, ignored tests, `unimplemented!`, warnings, and duplicate logic

It does not invent new features beyond the branch's own stated scope. If a feature has no implementation at all, the first deliverable is a narrow design/test slice, not a broad rewrite.

## Approach

1. Build an implementation-gap matrix from docs and code.
2. Classify each item as `implemented`, `partial`, `missing`, `wrong`, `duplicate`, or `stale-doc`.
3. Prioritize fixes by risk: compile correctness, tests that are ignored or fake, duplicated domain logic, then cleanup.
4. For each fix, write or un-ignore a focused failing test first, then implement the smallest change that makes the finding true.
5. Avoid unrelated refactors. Extract shared logic only where duplicate behavior is proven and local callers can use a small, stable interface.

## Evidence Rules

Every code change must point to one of:

- a failing or ignored test
- a documented audit claim that code does not satisfy
- duplicated logic found in two or more production files
- a compiler warning that reflects dead or disconnected implementation
- a release gate that cannot pass with current code

No code is written just because a TODO exists. A TODO becomes actionable only when it blocks a claimed branch goal or release gate.

## Verification

Baseline verification starts with:

- `cargo check --workspace --all-targets --locked`
- `git diff --check origin/claude/v0.4.1-full-audit...HEAD`
- focused `cargo test` commands for each touched area

For large or external-service gaps such as real cTrader fixtures, the expected outcome may be a clear test harness or fixture contract rather than fake data.
