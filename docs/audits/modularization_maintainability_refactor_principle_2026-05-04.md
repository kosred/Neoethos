# Modularization / Maintainability Refactor Principle

Created: 2026-05-04 Europe/Berlin
Repository: kosred/forex-ai
Scope: record the requirement that the refactor should make the codebase more modular, smaller per file, easier to maintain, and less duplicated.

## User idea

The user pointed out that the project should become more modular. This should reduce large files, reduce repeated logic, potentially reduce total lines of code, and make the system easier to maintain.

This is correct and should guide the refactor.

## Core principle

The refactor should not only move code around. It should reduce duplicated logic and make ownership clear.

Target:

```text
small focused modules
+ shared contracts
+ one owner per concept
+ backend implementations behind traits/interfaces
+ fewer scattered env reads
+ fewer duplicated helper functions
```

## Why modularization matters here

The audits found repeated or overlapping logic across:

- feature generation and alignment
- timestamp unit handling
- SMC feature derivation
- signal synthesis
- CPU vs GPU backtest behavior
- search configuration
- quality validation
- runtime hardware planning
- artifact metadata
- fallback behavior

When the same concept appears in several places, bugs become hard to find. CPU, GPU, training, search, validation, and live/runtime paths can slowly drift apart.

## Expected benefits

A proper modular refactor should improve:

- maintainability
- testability
- CPU/GPU parity
- reproducibility
- artifact trustworthiness
- hardware portability
- UI configuration
- onboarding and future development speed

It may also reduce total lines of code by deleting duplicate helpers after callers move to shared modules.

## Small file rule

Files should stay focused.

A file should not own unrelated responsibilities such as:

```text
hardware detection + config parsing + scheduling + kernel launching + artifact metadata
```

Instead, split by responsibility:

```text
hardware_profile.rs
hardware_probe.rs
device_assignment.rs
scheduler.rs
work_unit.rs
precision_policy.rs
fallback_policy.rs
```

The same rule applies to search, features, validation, and models.

## Module ownership rule

Each concept should have one owning module.

Examples:

- Feature logic belongs to the feature module.
- Signal synthesis belongs to the signal module.
- Backtest semantics belong to the evaluation module.
- Quality and challenge validation belong to the validation module.
- Hardware detection and work assignment belong to the runtime scheduler module.
- Artifact provenance belongs to the artifact/provenance module.

Other modules should call the owner module instead of duplicating the logic.

## Refactor strategy

1. Define shared contracts first.
2. Move callers one at a time.
3. Add parity tests before deleting old code.
4. Delete duplicated helpers only after all callers are migrated.
5. Keep files small and split when a file gains a second unrelated responsibility.

## Important caution

The goal is not to create hundreds of tiny files with no structure.

The goal is clear ownership:

```text
one concept -> one module -> small internal files -> reusable public contract
```

## Bottom line

The user's direction is correct.

The system should become more modular so that the same feature, signal, backtest, validation, runtime, and artifact contracts are reused everywhere. This should reduce duplicated code, reduce hidden mismatches, and make the bot much easier to maintain as it grows.
