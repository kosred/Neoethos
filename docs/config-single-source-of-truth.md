# One config. No env. No hidden fallback.

Decided and implemented 2026-08-10. This document is the scheme; the tests named
in it are the enforcement. If the two ever disagree, the tests are right.

---

## The defect, stated once

Four configuration surfaces existed, and exactly one was ever read.

| # | Surface | Keys | Read by a run? |
|---|---------|------|----------------|
| 1 | Rust `Default` impls, `crates/neoethos-core/src/config.rs` | 390 across 21 structs | only when a key is absent everywhere else |
| 2 | repo root `config.yaml` | 383 | **no** — only when no live store exists |
| 3 | `desktop/src-tauri/resources/config.yaml` | 233 | **no** |
| 4 | `%LOCALAPPDATA%\neoethos\config.yaml` | — | **YES. This one. Only this one.** |

`Settings::load()` resolves `$CONFIG_FILE` → the live store if it exists → the
literal relative path `"config.yaml"`. The live store exists. **Surfaces 2 and 3
were documentation that lied**, and the measurements say how badly:

- **150 keys** present in the repo root were absent from the desktop seed. Zero
  the other way. Every one of those 150 silently took a code default in the
  desktop bundle, with nothing recording that it had.
- **Five shared keys disagreed, and all five were on the money path**:
  `discovery_mode`, `daily_drawdown_limit`, `total_drawdown_limit`,
  `prop_firm_rules`, `account_currency`.
- Two of those five carried **f32 values widened to f64**
  (`0.10000000149011612`, `0.20000000298023224`). Nobody types that. It is the
  fingerprint of `server/risk.rs` persisting a prop firm's **raw** ceiling
  instead of the **buffered** one — and a UI preset click rewrites it, so the
  divergence regenerates itself.
- The test that was supposed to catch this, `shipped_config_matches_defaults.rs`,
  guarded **3 keys of 390**, **skipped absent keys by design**, never compared
  the two files against each other, and could not reach `%LOCALAPPDATA%` at all.
  Its own motivating example — `prefilter_top_k: 50` — is live in the operator's
  store right now, and the test was green.

The point that decided the design: **fixing those five values would have fixed
them once.** The mechanism that produced them — a second hand-editable source of
default values — would have been left completely intact.

---

## The scheme

### 1. Rust `Default` is the single source of default values

Nothing else declares a default. Not a YAML file, not a UI control, not a
constant next to a call site.

### 2. The desktop seed is GENERATED

`desktop/src-tauri/resources/config.yaml` is now a projection of
`Settings::default()`, produced by
`crates/neoethos-core/tests/generated_seed_is_current.rs`.

It cannot drift, because the test **rewrites the file and then fails**, printing
every key whose value moved. A `Default` change that is not mirrored into the
seed does not ship. The failure is deliberate: a shipped default moving is
something a human reads and commits, and `git diff` is the review surface.

The generated body opens with a marker line, `# neoethos-generated-defaults: v1`.
The migration tool refuses to treat the file as a defaults source without it, so
a stale hand-written seed can never be mistaken for the real defaults.

### 3. The repo root `config.yaml` is the developer experiment profile

Not installed, not shipped, not a source of defaults. It is allowed to tune. It
is **not** allowed to disagree with a `Default` without saying so: every
divergence on a pinned key must appear in the `ROOT_REGISTERED` table in
`shipped_config_matches_defaults.rs`, with the value and the reason. A
divergence then stops being drift and becomes a dated decision that neither side
can move without editing that table.

### 4. The operator's live store carries overrides only

It is the only file a run reads. It is collapsed to overrides by
`neoethos-cli config normalize --write`, never by hand and never by a startup
step.

---

## How a developer runs a local experiment with a different setting

`$CONFIG_FILE` is the **first** branch of the resolution order and is the only
supported way to point a run at a different file.

```powershell
Copy-Item config.yaml my-experiment.yaml
# edit my-experiment.yaml
$env:CONFIG_FILE = 'my-experiment.yaml'
cargo run -p neoethos-cli -- discover
```

Two properties make this safe rather than a fifth surface:

1. **The run logs, by name, which file it opened.** Nothing logged this before,
   which is how two subsystems in one process ended up reading different files.
2. **`$CONFIG_FILE` is explicit.** The failure mode being removed is the
   *implicit* one — the bare relative `"config.yaml"` fallback, which made the
   same binary trade differently depending on the directory it was started from.

For tests, an explicit test-only constructor is the supported route. A silent
cwd fallback is not.

---

## What enforces this

| Enforcement | Where | What it refuses |
|---|---|---|
| A second load path fails to **compile** | `config_single_load_path.rs` + a private loader module (shard A) | `Deserialize`-based construction, `from_str`, `from_reader`, `Settings { .. }` literals outside the loader |
| The seed cannot drift | `generated_seed_is_current.rs` | any `Default` change not mirrored into the shipped seed |
| No unknown keys | `shipped_config_matches_defaults.rs` | tombstones and typos in the repo config **and in the live store** |
| No unregistered divergence | same file, `ROOT_REGISTERED` / `LIVE_REGISTERED` | a pinned key differing from its `Default` without a written reason |
| Unknown keys become fatal | `deny_unknown_fields` on all 21 structs (shard A) | `trailing_enabeld:` parsing, saving, and reporting "saved (verbatim)" |
| No env reads | `env_surface_is_empty.rs` in `neoethos-search` and `neoethos-data` (shard D) | `env::var` outside `#[cfg(test)]` |

**Ordering constraint, and it is not optional:** `deny_unknown_fields` turns the
four tombstones in the operator's live store (`export_onnx`,
`news_kill_window_min`, `news_lookahead_minutes`, `perplexity_enabled`) into a
hard startup failure. **The migration must land and be run first.**

---

## The rules this wave did not break

1. **No money value changed silently.** Every money-path value whose effective
   result changes is named at startup with old, new and why.
2. **No limit was silently raised.** Where two numbers conflicted the safer one
   won and both were logged. `daily_drawdown_limit` 0.10 → 0.04 and
   `total_drawdown_limit` 0.20 → 0.07 in the repo profile are the instance;
   both tighten, neither permits anything new.
3. **`max_portfolio_risk: 0.0` was not reinterpreted.** On a knob named `max_`,
   `0` currently means **no cap at all**, not "no risk". Deciding it means
   something else is a behaviour change the operator sanctions, so it is
   implemented as a **loud startup error naming both readings**, never a silent
   correction.
4. **The operator's live file is sacred.** The migration backs it up first,
   prints the whole diff before writing, refuses to run unattended, has no
   `-Force` and no `-Yes`, asks per section, asks per money item, and defaults
   to leaving every one of them alone.
5. **Nothing was committed.**

---

## The five money values only the operator can decide

His live store is dated 2026-07-31. These are printed individually by the
migration tool, with what each means **today**:

| key | his value | what that means right now |
|---|---|---|
| `models.prop_search_min_payoff_ratio` | `0.0` | **the realized-payoff floor is OFF** — any win/loss shape is admitted |
| `models.discovery_runtime.prefilter_top_k` | `50` | base feature set collapses 217 → ~64 columns; SMC, session and footprint families die first ⚠ **the "217" is stale** — see note below |
| `models.require_walkforward_for_export` | `false` | **the out-of-sample export gate is OFF** — a portfolio reaches live money on the window gate alone |
| `risk.max_portfolio_risk` | `0.0` | **NO CAP AT ALL**, not "no risk" |
| `risk.trailing_enabled` | `true` | the **orphaned** copy; the search reads `models.exit_policy`, and live execution trails unconditionally with no config gate |

> ⚠ **The "217 columns" in the `prefilter_top_k` row is stale (2026-08-09).**
> The base cube is no longer 217 columns. With the vocabulary restored it offers
> **1,946 columns per timeframe** on this machine — 5,838 across an M5+H1+H4 run
> — and up to the memory budget's 4,096 on a larger box. The *direction* of the
> warning stands (a small `top_k` kills whole feature families first), but the
> arithmetic behind it does not: a **constant** `top_k` against a
> hardware-dependent cube width discards a hardware-dependent fraction. Measured
> in [`higher-timeframe-lane-2026-08-09.md`](higher-timeframe-lane-2026-08-09.md).
> Unrelated to — but easily confused with — the **void** `217/217, 40/217,
> 8/217` higher-timeframe keep rates retracted in that same document.

And the trailing values specifically: his file sets
`risk.trailing_atr_multiplier: 0.4` and `risk.trailing_be_trigger_r: 0.1` and has
**no `models.exit_policy` block at all**. The search reads that block
exclusively, so those hand-tuned numbers have moved nothing. The migration
shows all four side by side and **does not copy them across** — copying
`trailing_enabled: true` turns the trail on for every future search, and the
measured record says the trail is applied before the take-profit check on every
bar, which made the take-profit dead code and pinned realised payoff near 1.08
against a configured floor of 2.0. It also warns that
`trailing_atr_multiplier` → `trailing_stop_multiplier` is a rename where **the
old name lied**: it was never an ATR multiple, it is a multiple of the
position's own stop distance.

---

## Running the migration

```bash
# 1. Report only. Nothing is written. Read this first: one row per key you
#    override, beside the default it shadows, money keys marked.
neoethos-cli config normalize

# 2. Apply: backs the store up, rewrites it carrying ONLY what diverges, then
#    reloads it and RESTORES THE BACKUP unless the reload is byte-identical in
#    effect. Money keys are written even when they equal the default.
neoethos-cli config normalize --write
```

There is deliberately no `-Force` and no `-Yes`. The backup is timestamped and
the revert command is printed on completion.
