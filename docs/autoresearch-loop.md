# The autoresearch loop — design

**Status:** design, complete and decided. This is the document the three builders
implement. Where a choice was open it has been made here and the rejected option
is named, so no builder has to re-open it.

**Territory:** everything specified here is built in `crates/neoethos-autoresearch/**`
plus one workspace entry in the root `Cargo.toml`. Every change needed anywhere
else is in §14 REQUIRED ELSEWHERE and must be applied by whoever owns that file.

**What it is.** A loop that proposes `(search configuration, objective)` pairs,
runs them as sweeps, judges them against a frozen judge, records everything, and
stops with one of three verdicts. It optimises *toward* the operator's goal
constants; it can never redefine them.

**What it is not.** It never places an order, never contacts a broker, never
writes `live_portfolio.json`, never writes `config.yaml`. Its only output is a
proposal artifact and a verdict report. The operator promotes.

---

## 0. The one risk this design exists to contain

An optimiser pointed at *"turn 100 into 50,000 in six months"* will find
configurations that appear to do it. The measured base rate says so:
expectancy was **−4.15 pips per trade in every exit configuration tested** while
the payoff ratio moved from 0.91 to 2.53
(`crates/neoethos-search/src/run_identity.rs`, module doc). Exit geometry
redistributes between win rate and payoff; the product stays at minus the cost.

Everything below follows from that one fact:

* the objective variants in §7 are chosen precisely because they are *not*
  re-parameterisations of the (win-rate, payoff) split;
* the judge in §8 is the spine, not a filter on the end;
* the shuffle control in §9 is inside the loop, on a fixed cadence, spending 20%
  of the budget, because a session whose output is unfalsifiable is worth 0% of
  the budget;
* the loop can say *"no configuration in this space reaches the goal"* (§10) in
  exactly the same report shape as a success.

---

## 1. The goal constants — read-only, named by field path

These are already in the config. The loop **loads** them and **hashes** them. It
has no code path that writes them: `GoalSet`'s fields are private, it exposes
getters only, and `SearchConfigDelta` (§6) has no variant that names any of them.

### 1.1 Risky mode — capital multiplication

| Field path | Type | Default | Source |
|---|---|---|---|
| `system.risky_start_balance_usd` | `f64` | `100.0` | `crates/neoethos-core/src/config.rs:147` (default `:263`) |
| `system.risky_target_balance_usd` | `f64` | `50000.0` | `config.rs:148` (default `:264`) |
| `system.risky_horizon_days` | `u32` | `180` | `config.rs:149` (default `:265`) |
| `system.trading_mode` | `String` | `"prop_firm"`; operator's installed value is `risky` (`config.yaml:224`) | `config.rs:138` |

Mirrored into the search as `DiscoveryConfig::risky_start_balance`,
`risky_target_balance`, `risky_horizon_days`
(`crates/neoethos-search/src/discovery.rs:667-669`). The loop reads the
`system.*` fields as the authority and asserts the mirror agrees; a disagreement
is a startup abort naming both paths.

**The second scenario (1,000 → 200,000) is not in the config.** It exists only
as a demo constant, `crates/neoethos-search/examples/goal_frontier_demo.rs:36`.
The loop must not invent it and must not hardcode it. §14.1 specifies the exact
config addition that makes it a first-class read-only scenario. Until that lands,
`GoalSet::scenarios()` returns exactly one risky scenario and the session header
records `scenarios_source: "system.risky_* (single)"`.

### 1.2 Prop-firm mode — up to 4% each month

| Field path | Type | Default | Source |
|---|---|---|---|
| `risk.monthly_profit_target_pct` | `f64` | seeded from the active preset's `min_monthly_net_profit_pct` = **0.04** | `config.rs:388`, seeded at `:625`; preset constant `crates/neoethos-core/src/domain/prop_firm.rs:125` |
| `risk.preset` | `PropFirmPreset` | `Ftmo` | `config.rs:380` |
| `models.prop_firm_min_pass_rate` | `f64` | `0.40` (`config.rs:2523`) | `config.rs:1165` |
| `models.discovery_runtime.prop_firm_gate.max_daily_loss_pct` | `Option<f64>` | `None` → FTMO `0.05` | `config.rs:1559` |
| `models.discovery_runtime.prop_firm_gate.max_overall_drawdown_pct` | `Option<f64>` | `None` → FTMO `0.10` | `config.rs:1562` |
| `models.discovery_runtime.prop_firm_gate.profit_target_pct` | `Option<f64>` | `None` → FTMO `0.10` | `config.rs:1567` |
| `models.discovery_runtime.prop_firm_gate.min_trading_days` | `Option<usize>` | `None` → FTMO `10` | `config.rs:1570` |
| `models.discovery_runtime.prop_firm_gate.window_days` | `usize` | `60` | `config.rs:1575` |

### 1.3 The startup refusals — a zero goal is not a goal

The operator's installed `config.yaml` carries `risk.monthly_profit_target_pct: 0.0`
(`config.yaml:194`) and `models.prop_firm_min_pass_rate: 0.0` (`config.yaml:160`).
A loop that judged a prop-firm sweep against a 0% monthly target and a 0% pass
rate would promote anything at all. So:

```rust
// goals.rs — GoalSet::load(). Every one of these ABORTS the session, naming the
// field path and the resolved value. None of them is a warning.
G1  trading_mode == "risky"     && !(target > start > 0)               -> abort
G2  trading_mode == "risky"     && horizon_days == 0                   -> abort
G3  trading_mode == "prop_firm" && monthly_profit_target_pct <= 0.0    -> abort
G4  trading_mode == "prop_firm" && prop_firm_min_pass_rate  <= 0.0     -> abort
G5  trading_mode not in {"risky","prop_firm"}                          -> abort
G6  any goal field non-finite                                          -> abort
```

The loop does not fix these. It refuses to run and names the field, because
fixing them is writing the goals.

### 1.4 What else is frozen

Frozen for the life of a session, alongside the goals:

* **the cost model** — `evaluation_spread_pips`, `evaluation_commission_per_trade`,
  `swap_long_pips_per_day`, `swap_short_pips_per_day`, `session_spread_pips`,
  `cost_band_pips`, `pip_value_per_lot`;
* **the judge's own thresholds** (§8.4) and the OOS window definition (§8.2);
* **the validation geometry** — `enable_cpcv`, `cpcv_n_splits`,
  `cpcv_n_test_groups`, `cpcv_embargo_pct`, `cpcv_purge_pct`, `cpcv_max_rows`,
  `walkforward_splits`, `embargo_minutes`. These belong to the judge, not to the
  search. A loop that could widen its own folds could reach any number.

**The distinction that matters, stated once so it cannot drift:**

> The loop may vary **what the SEARCH refuses**. It may never vary **what the
> JUDGE refuses**.

Axis B (§7) moves `TargetProfile` floors freely — that is exactly what the
operator asked for. Every sweep, whatever its search-side floors were, is then
judged by the same frozen judge with the same frozen costs on the same frozen
OOS window. A loosened search-side floor can therefore buy more candidates, but
it can never buy a promotion. That is the whole safety argument and it is
mechanical, not procedural.

Enforcement: `space.rs` defines `const FROZEN_FIELDS: &[&str]` listing every path
above; a unit test enumerates every field `SearchConfigDelta::apply` writes and
asserts the two sets are disjoint. A builder who adds a delta variant touching a
frozen field breaks that test.

---

## 2. Crate layout

```
crates/neoethos-autoresearch/
  Cargo.toml
  src/lib.rs         invariants in doc form; re-exports; NO logic
  src/contracts.rs   fail-loud imports + startup self-check (§13)
  src/goals.rs       GoalSet: read-only load, the six refusals, goal_hash
  src/space.rs       axis-A factors, axis-B variants, SearchConfigDelta, FROZEN_FIELDS
  src/proposal.rs    Proposal, ProposalOrigin, trust region, dedupe by config_hash
  src/proposer.rs    Beta-Bernoulli marginal sampler + exploration floor (§5)
  src/objective.rs   ObjectiveVariant -> concrete search-side overrides (§7)
  src/judge.rs       JudgeThresholds, screen(), promote(), the inequality (§8)
  src/shuffle.rs     rotation control, block bookkeeping, the session null (§9)
  src/session.rs     Session = a pure fold over the journal (§11)
  src/journal.rs     JSONL records, crash-safe append, replay
  src/verdict.rs     the one report shape; both stopping rules (§10)
  src/runner.rs      the state machine (§3); the only module that calls the search
```

No `[[bin]]`. The CLI entry point lives in `neoethos-cli`, which is forbidden
territory; §14.2 specifies the subcommand.

Dependencies: `neoethos-core`, `neoethos-search`, `neoethos-data`, `serde`,
`serde_json`, `anyhow`, `rand`, `rand_chacha`, `tracing`. **Not**
`neoethos-app`, **not** `neoethos-trader` — the crate must be structurally
incapable of reaching the broker.

---

## 3. The state machine — one iteration

**An iteration is one SWEEP. A sweep is `SWEEP_SEARCHES = 100` searches**, which
is the operator's own unit: *"it does a SWEEP with 100 searches on the card, logs
the good ones, and passes them through validation ONCE."*

The 100 proposals of a sweep are **all drawn before any of them runs**, so a
sweep is one experiment with one `sweep_hash` (the ordered hash of its 100
`config_hash`es) and is reproducible from `(session_seed, sweep_index)` alone.

```
                    ┌──────────────────────────────────────────────┐
                    │  S0  Open / Resume                           │
                    │  fold the journal -> Session (§11)           │
                    │  verify goal_hash + judge_hash unchanged     │
                    └───────────────────┬──────────────────────────┘
                                        v
   ┌──────────────────────────> S1  PROPOSE  (proposer.rs)
   │                            draw 100 Proposals from the posterior
   │                            75 posterior-sampled + 25 uniform floor
   │                            journal: ProposalDrawn x100
   │                                    v
   │                            S2  GUARD  (proposal.rs)
   │                            per proposal: trust region, dedupe by
   │                            config_hash against the session hash set,
   │                            coverage debt. Duplicates -> ProposalDuplicate
   │                            + redraw (bounded). Exhaustion -> R2/U5.
   │                                    v
   │                            S3  STAMP  (run_identity.rs)
   │                            ResolvedConfigStamp per proposal ->
   │                            config_hash; assert_payoff_floor_reachable.
   │                            A proposal whose floor is unreachable under its
   │                            own geometry is REFUSED HERE, named and counted
   │                            (reject_unreachable_floor) — never run.
   │                                    v
   │                            S4  RUN  (runner.rs -> neoethos-search)
   │                            journal: SweepStarted (INTENT, fsynced) BEFORE
   │                            any work. 100 searches on the card, each through
   │                            the streaming inner loop with the early-reject
   │                            predicate. Per-search failures are named and
   │                            counted; the sweep continues.
   │                                    v
   │                            S5  COLLECT
   │                            per search: TrialStatisticsReport, the ten
   │                            rejection counters, CostBandCensus, the batch
   │                            ledger, the canonical survivors.
   │                            N_session += trials_offered (all of them).
   │                                    v
   │                            S6  SCREEN  (judge.rs::screen)
   │                            in-sample only. Never touches OOS.
   │                            journal: Screened
   │                                    v
   │                            S7  RECORD
   │                            journal: SweepCompleted (OUTCOME, fsynced)
   │                            + PosteriorUpdated
   │                                    v
   │                            S8  BLOCK BOUNDARY?  every SHUFFLE_PERIOD = 4
   │                            live sweeps -> S8a run the block's BEST sweep's
   │                            exact 100 proposals against rotated features
   │                            (§9); journal: ShuffleControlCompleted,
   │                            BlockJudged; update the session null.
   │                                    v
   │                            S9  DECIDE  (verdict.rs)
   │                            R1 goal reached?      -> S10 PROMOTE (one OOS touch)
   │                            R2 provably not?      -> STOP, verdict
   │                            budget out?           -> STOP, Inconclusive
   └────────────────────────────  else                -> S1
                                        v
                                S10 PROMOTE
                                evaluate ONLY (no search) on the OOS window;
                                journal: OosTouchSpent, then Promoted or
                                PromotionRefused with the failing conjunct named.
                                STOP either way — the OOS window is spent.
```

Two-phase journalling (intent before work, outcome after) is what makes *"a
crash loses one sweep, not the session"* true rather than hoped. On resume a
`SweepStarted` with no matching outcome becomes `SweepAbandoned { reason:
"process died" }` — counted, named, and if a partial trial-returns matrix exists
its `trials_offered` **still counts into `N_session`**. A trial that ran is a
trial that ran; forgetting it would be a loop lying to itself with extra steps.

### 3.1 What carries between iterations

| Carried | Why |
|---|---|
| the journal (everything) | the only state; §11 |
| `N_session` — cumulative `trials_offered` over **live + control + failed + abandoned** sweeps | the honest N for the DSR |
| the factor-level Beta posteriors | the proposer's memory |
| the `config_hash` set | duplicate refusal; two sweeps with the same hash are the same experiment |
| the OOS touch budget (per window) | one touch, ever |
| block bookkeeping + the session shuffle null | §9 |
| best-ever screened sweep (id, hash, statistics) | R1/R2 evidence |
| coverage counters per axis-A level and axis-B variant | U4: a space is not refuted if it was not searched |
| `goal_hash`, `judge_hash`, `cost_hash` | resume invalidation |
| RNG stream position `(session_seed, draw_counter)` | proposals replayable |
| the session champion matrix (one row per sweep) | `pbo_session`, §8.3 |

### 3.2 What is discarded

The feature cube; the GA populations; every non-surviving gene; per-sweep frames;
the streaming batch cursor *inside* a sweep (each sweep's streaming run is
self-contained — its `next_cursor` is recorded as provenance, but a new sweep
starts from its own proposal's cursor); everything in `DiscoveryResult` except
the stamp, the statistics, the censuses, the canonical survivors and the
trial-returns manifest hash.

**The trial-returns binary is ~14 MiB per run** and there are 100 per sweep. The
manifest and the `TrialStatisticsReport` are kept always. The binary is kept only
for (a) the session's best-ever sweep, (b) every promotion candidate, (c) every
shuffle control. The rest are deleted **under a named census** —
`sweeps_gc: {kept, deleted, bytes_reclaimed, rule}` — never silently, and the
disk budget is probed from the filesystem the way `trial_returns.rs` already
does, never a constant.

---

## 4. Karpathy's autoresearch — what is taken, what is rejected

**Taken.**

* *The journal is the memory.* Every trial with its full configuration and its
  result, appended, and the next proposal is a function of that history. This is
  the single most valuable idea and it is adopted whole, including the property
  that the history is the artifact — the loop's claims are re-derivable from it.
* *Proposals are deltas against a named baseline*, not configurations built from
  nothing. So an improvement is attributable to a change.
* *The outer budget with an explicit propose → run → judge → record → decide
  cycle*, and the loop owning the decision to stop.

**Rejected, with reasons.**

* **Free-form code/config mutation by an LLM.** Karpathy's loop lets the proposer
  rewrite the thing being optimised. Here the objective is adversarially
  exploitable and the operator trades the output with real money. A proposer that
  can emit arbitrary edits can edit the gates, and *a loop that can edit its own
  objective will reach any goal you like and mean nothing*. Replaced by a typed,
  bounded `SearchConfigDelta` over a declared knob set, with `FROZEN_FIELDS`
  enforced by a test.
* **Hill-climbing on the headline in-sample metric.** This is the core of
  Karpathy's loop and it is exactly the machine that manufactures overfits here,
  because on this data the in-sample metric is mostly selection noise. Replaced
  by: the proposer's reward channel is **binary and shuffle-corrected** (§5.3),
  and the headline metric never enters the proposer at all.
* **The single-metric leaderboard.** Replaced by the two-stage judge (§8), where
  a refusal is a failure and never a pass.
* **An LLM in the inner loop.** A proposal must be replayable from
  `(session_seed, sweep_index)`; a model in the loop destroys that and makes the
  session unauditable. An LLM may act only as an out-of-band suggestion channel:
  a human-or-model-authored proposal is admitted as
  `ProposalOrigin::External { note }`, must be expressible in the typed space,
  is stamped and deduplicated like any other, is counted separately in the
  census, and has **no** privileged path through the judge.
* **Restarting the trial count per experiment.** Karpathy's loop has no notion of
  a session-wide N because it is not doing statistical selection. Here N is the
  spine: it is a fold over the journal (§11.3), never a carried counter, so it
  cannot be reset.

---

## 5. The proposer

### 5.1 Shape of the space

A proposal is:

```rust
pub struct Proposal {
    pub parent: Option<SweepId>,        // the baseline this is a delta from
    pub axis_a: SearchConfigDelta,      // §6 — typed, bounded, frozen-field-free
    pub axis_b: ObjectiveVariant,       // §7 — a small enum, enumerable
    pub refusals: RefusalLevels,        // §7.9 — search-side floors, a vector
    pub replicate_seed: u64,            // GA seed; a replicate dimension, not a factor
    pub origin: ProposalOrigin,         // Posterior | ExplorationFloor | External
    pub config_hash: String,            // from ResolvedConfigStamp, filled at S3
}
```

Axis A is a vector of ~11 categorical **factors** with 3–5 **levels** each (§6).
Axis B has 8 variants plus a 4-dimensional refusal vector. The full product is
order 10^6–10^7 configurations against a budget of order 10^4 searches. That
ratio decides the algorithm: **only marginal (per-level) effects are estimable**;
anything that tries to estimate interactions would be fitting cells of size 1.
This is stated as a design assumption rather than discovered later.

### 5.2 The reward, and why it is binary

Reward per *proposal* (per search), credited at S6:

```
reward(p) = 1  if screen(p) passed  (§8.1, all of S1..S6)
            0  otherwise
```

Binary, not the screen's continuous score, for one reason: on this data the
continuous in-sample score is dominated by selection noise, so a proposer
hill-climbing on it converges on the first lucky draw. `"did anything survive a
deflated, shuffle-corrected screen"` is a far more robust signal, and
Beta–Bernoulli's posterior width gives exploration proportional to ignorance for
free — no hand-tuned temperature to drift.

### 5.3 The anti-lucky-overfit rule, concretely

Three mechanisms, all mechanical:

1. **The reward channel is shuffle-corrected before it reaches the proposer.**
   `screen()` includes conjunct **S6** — the candidate must beat the accumulated
   shuffle null (§9). Until the null has `MIN_NULL_OBS = 3` observations, S6 is
   *unavailable*, and while it is unavailable **every** reward is recorded as 0
   and the posterior is not updated at all. So the first three blocks (12 sweeps,
   1,200 searches) are pure exploration by construction, not by a tuned schedule.
2. **A hard exploration floor that never decays.** Exactly
   `EXPLORATION_FLOOR = 25` of each sweep's 100 proposals are drawn uniformly at
   random over the whole space, independent of the posterior, forever. It is not
   annealed. A proposer whose exploration decays converges on its own first
   mistake, and this loop runs for days on a noisy signal.
3. **One success per level per sweep, and a block veto.** A factor level gets at
   most **one** success credited per sweep, however many of that sweep's 100
   proposals carried it — otherwise a single lucky region floods the posterior.
   And if the block containing the sweep is later judged `INDISTINGUISHABLE`
   from its shuffle control (§9.4), every success credited from that block is
   **retracted and re-credited as a failure**, by replaying the journal. The
   posterior is a fold, so this is exact, not an approximation.

### 5.4 The algorithm

```
for each of the 100 slots in a sweep:
    if slot < EXPLORATION_FLOOR (25):
        draw every factor level uniformly; parent = None
    else:
        # Thompson sampling on marginals
        for each axis-A factor f:
            for each level l of f: theta[f][l] ~ Beta(alpha[f][l], beta[f][l])
            pick argmax_l theta[f][l]
        for axis-B: theta[b] ~ Beta(alpha[b], beta[b]); pick argmax_b
        parent = the session's best-ever screened sweep
        # TRUST REGION: revert factors until at most MAX_DELTA_FACTORS = 3
        # differ from parent, choosing which to revert by smallest posterior gap
    apply COVERAGE DEBT: if any axis-B variant has fewer than B_MIN_DRAWS = 3
        sweeps' worth of draws, force this slot to that variant (round-robin).
        Axis B has 8 levels and is the half usually skipped; it is small enough
        to enumerate rather than sample, and U4 requires the coverage anyway.
    stamp -> config_hash; if duplicate, redraw (up to PROPOSER_RETRIES = 64);
        after that force a uniform draw; after PROPOSER_RETRIES uniform
        duplicates in a row -> ProposerExhausted (§10, U5).
```

Priors: `Beta(1, 1)` on every level — uniform, no built-in preference. The
priors are written into the session header so a reader can see the loop started
with no opinion.

Rejected alternatives, named: **ε-greedy** (needs a tuned ε and an annealing
schedule, both of which are exactly the knobs that let a design drift toward
early exploitation); **UCB** (its confidence term needs a bounded reward scale
and behaves badly when most arms have reward ≈ 0, which is the expected regime
here); **full-configuration bandits** (the space is 10^6 wide against a 10^4
budget — every arm would be pulled at most once); **Bayesian optimisation with a
GP** (the response surface is categorical and the observation noise dominates the
signal; a GP would fit the noise and would add a dependency for the privilege).

---

## 6. Axis A — search configuration variants

The typed delta. Every factor here is a *search* knob; none of them is in
`FROZEN_FIELDS`.

| Factor | Levels | Target field |
|---|---|---|
| `population` | 512 / 2048 / 4096 / 8192 (with `population_auto = false`) | `DiscoveryConfig::population` |
| `generations` | 20 / 50 / 100 / 200 | `::generations` |
| `gene_width` | 4 / 8 / 12 | `::max_indicators` |
| `prefilter_width` | 120 / 240 / 480 / 960 | `runtime.prefilter_top_k` |
| `prefilter_insample_frac` | 0.60 / 0.70 / 0.80 | `runtime.prefilter_insample_frac` |
| `base_timeframe` | M5 / M15 / M30 / H1 | `::timeframe_label` |
| `higher_lanes` | none / {H1} / {H1,H4} / {H1,D1} | `::higher_timeframes` |
| `streaming_batches` | 0 (single pass) / 4 / 12 / 32 | `StreamingPlan::max_batches` |
| `working_set_cursor` | 0 / resumed / random offset | the streaming cursor |
| `candidate_count` | 64 / 256 / 1024 | `::candidate_count` |
| `portfolio_shape` | `(portfolio_size, corr_threshold)` ∈ {(1, 1.0), (5, 0.7), (12, 0.5)} | `::portfolio_size`, `::corr_threshold` |

**The batch WIDTH is never a factor.** It comes from
`hpc_ta::streaming_batch_columns` / `VocabularyBudget`, i.e. from free RAM,
exactly as `docs/streaming-parameter-search.md` §3.1 and the `StreamingPlan` doc
require. What the loop chooses is *whether* to sweep and *for how many batches*,
never how much memory to risk. The never-OOM invariant is not negotiable by a
proposer.

**H4 is deliberately absent from `higher_lanes` as a lane on its own.**
`docs/higher-timeframe-lane-2026-08-09.md` §3–4 measured that **zero of 1,855 H4
columns clear a Bonferroni bar on their own effective sample size**, and that the
H4 columns that currently take top-240 places do so on 12–18 real observations
dressed as 568–840 forward-filled rows. Admitting H4 unchanged replaces a false
negative with a false positive, and this loop would find it and promote it. H4
becomes reachable through **axis B, variant B4** (label horizon) — which is the
mechanism that document names: *"if H4 is to be tested properly it needs a longer
label horizon and/or an H4-native base, not a quota."* That is the honest way to
give the lane a chance, and it is the only way this design allows.

**H1 is present and prominent**, because the same document overturned the
standing belief for H1: 0 columns earned on rank before the correlation repair,
**79 after**, 106 clearing the Bonferroni bar, topping out at `t = 15.4`.

`replicate_seed` is a **replicate dimension, not a factor**: repeated draws of
the same `(axis_a, axis_b, refusals)` cell with different seeds are how
within-cell variance is estimated, and the judge needs that. Two proposals
differing only in `replicate_seed` are **not** duplicates (they produce different
`config_hash`es only if the seed is in the stamp — see §14.3).

---

## 7. Axis B — the objective variants

This is the half that is usually skipped, and it is the half where the measured
base rate says the answer must live.

### 7.0 What does NOT count as a variant

Named explicitly so no builder adds them and no reader mistakes them for
content. Each of these only reshapes the (win-rate, payoff) split of a fixed
population of trades, and the product of that split is pinned at minus the cost:

* **B-null-1: exit geometry** — trailing on/off, `be_trigger_r`, give-back,
  min-lock, SL/TP clamps. *Measured*: expectancy −4.15 pips in every one of them
  while payoff moved 0.91 → 2.53.
* **B-null-2: the RR ladder / `min_payoff_ratio`** — the same thing under
  another name. Payoff 2.53 at expectancy −4.18 pips is a gate-passing
  money-loser (`discovery.rs:2516`).
* **B-null-3: re-weighting the four scoring tables** (`scoring/named.rs`) —
  Sharpe vs PF vs win rate vs net. These are different monotone summaries of the
  *same* in-sample per-trade distribution; they reorder candidates without
  changing what the population of trades is or what is being estimated. Worth
  **one** sweep as a control, recorded as `B0_ScoringTable`, never a family.

A genuinely different objective must change at least one of:
**(i) which trades are eligible at all**, **(ii) what quantity is being
estimated**, **(iii) what the unit of evaluation is** (trade vs. path vs. month),
or **(iv) what event is being predicted**.

### 7.1 B1 — Selectivity / abstention

*Changes (i): the eligible population.*

The average trade loses 4.15 pips. The only exit from that is **fewer,
different** trades. So invert the trade-count constraint: instead of a
`min_trades_per_day` floor plus a fitness that rewards volume, impose a **hard
trade budget** and maximise net over it.

```
min_trades_per_day      -> 0.0            (floor removed)
max_in_market           -> 0.10 / 0.05 / 0.02   (a level of the refusal vector)
fitness                 -> net per BAR IN MARKET, not net per trade
                           (a strategy that is flat is not penalised)
```

Rationale in the code's own words: *"A strategy in the market almost always is
not selecting entries, and its win rate converges on the market's base rate
however the entry rule is written"* (`discovery.rs:2522`).

### 7.2 B2 — Conditional / regime-restricted expectancy

*Changes (i).*

Maximise expectancy **within a conditioning set declared before the run**, from a
fixed list: `{session ∈ (Asia, London, NY)}`, `{ATR percentile bucket}`,
`{day of week}`, `{regime label}`, `{post-high-impact-news window}`. The honest
version is not "find the bucket that worked" — it is: the family is declared, and

* the **complement's** expectancy is reported alongside, always; and
* the **number of buckets in the family multiplies into the trial count `N`**
  the DSR deflates against.

Without those two rules this is the classic subgroup overfit. With them it is the
one variant that can find a real edge that lives only in part of the day, which
is what a cost-dominated instrument would look like if it had one.

### 7.3 B3 — Cost-elastic objective

*Changes (ii): what is being estimated.*

Score at the **pessimistic edge of `cost_band_pips`**, not at the point estimate.
This is not a monotone transform of the point-estimate ranking, because cost
scales with trade *count*: it systematically re-ranks a high-frequency candidate
below a low-frequency one with the same point-estimate net. It attacks the
measured failure directly — −4.15 pips is a cost-dominated number.

Prerequisite, and it is a real one: `cost_band_discriminates(band, baseline)`
must be **true**. At the shipped configuration it is **false** — the baseline is
3.4 pips against band edges 1.6 / 2.4, both *cheaper* than the run the candidate
already survived (`discovery.rs`, `cost_band_discriminates` doc). A band that
cannot discriminate makes every census read clean, *"which is worse than no
census, because a reader takes it as evidence."* So: **the loop aborts at startup
if `cost_band_discriminates` is false**, naming the three numbers. §14.4 carries
the config change; the loop does not make it.

### 7.4 B4 — Label / holding horizon

*Changes (iv): the event being predicted.*

The triple-barrier first-passage label's horizon (`max_hold_bars`, and the
`sl_atr_mult` / `rr` pair that define the barriers) decides what "an edge" even
means. This is not exit geometry: it changes the *target variable* the prefilter
ranks against and the GA is scored on.

The higher-timeframe document makes this concrete: a 35-bar (~3 hour) label
**resolves inside a single H4 candle** — the feature is being asked to predict an
outcome shorter than its own bar. Levels: `35 / 120 / 480 / 1440` base bars,
paired with `higher_lanes` so an H4 lane is only ever proposed with a horizon
that can express it.

This is the variant that makes the H4 lane askable at all, and the only one.

### 7.5 B5 — Path-level / terminal-wealth objective

*Changes (iii): the unit is a PATH, not a trade.*

Maximise the goal itself:
`goal_report::build_report(...).frontier[argmax].p_reach_target`, subject to
`p_ruin ≤ P_RUIN_MAX` at that same risk level. Two candidates with identical
per-trade expectancy can have very different `P(reach)` through variance,
compounding and serial structure, so this is a genuinely different functional.

**Why it is safe to include even though it is the goal.** `goal_report`'s own
test proves the property: a negative-edge system reaches a 500× target with
probability `< 0.05` at *every* risk level
(`goal_report.rs::a_negative_edge_basically_never_reaches_and_mostly_ruins`). The
objective can *rank* among positive-edge candidates; it cannot manufacture edge
from a negative one. And it is scored on OOS R-multiples at the pessimistic cost
edge when it matters (§8.3), never on in-sample ones.

### 7.6 B6 — Monthly consistency under path constraints (the prop-firm half)

*Changes (iii): the unit is a MONTH, with path constraints.*

Maximise the fraction of months clearing `risk.monthly_profit_target_pct` (4%),
subject to no window ever violating `max_daily_loss_pct` (5%) or
`max_overall_drawdown_pct` (10%). Drawdown-path constraints are not a function
of the per-trade mean — a candidate can have identical expectancy and be either
inside or outside the rule — so this is a different objective, not a re-weighting.

This is the axis-B variant that serves `system.trading_mode = "prop_firm"`, and
it is the one that must dominate when the goal set is the 4%/month one.

### 7.7 B7 — Significance-first

*Changes (ii).*

Maximise the expectancy **t-statistic** rather than the expectancy. This
penalises the high-variance, low-count candidates that dominate an expectancy
ranking, and it targets selection bias at the candidate level, where it is
cheapest to fight. The field exists and ships at 0.0 with a note saying `t ≥ 2.0`
is the value to set once a baseline is read (`discovery.rs:2492-2506`) — this
variant is that experiment, run properly and judged.

### 7.8 B8 — Portfolio-level objective

*Changes (iii): the selection unit.*

Score the **portfolio** after correlation pruning, not the best single gene:
maximise portfolio net expectancy subject to `corr_threshold` and a
portfolio-level drawdown. Diversification is a real, non-redistributive effect on
path statistics, and the live artifact is a portfolio anyway. Cross-batch
portfolios are already sound: correlation pruning operates on `Vec<i8>` signal
vectors over the same bars, and Option C's canonical index makes the gene indices
self-describing (`docs/streaming-parameter-search.md`, settled 2026-08-10).

### 7.9 The refusal vector — what the SEARCH refuses

Not separate objectives; a 4-dimensional vector carried alongside the variant, so
"what is maximised" and "what is refused" are independent as the operator asked.

| Dimension | Levels | Field |
|---|---|---|
| `expectancy_significance` | 0.0 / 2.0 / 3.0 | `TargetProfile::min_expectancy_t_stat` |
| `win_rate_floor` | 0.0 / 0.35 / 0.45 | `TargetProfile::min_win_rate` |
| `payoff_floor` | 0.0 / 1.0 / 2.0 | `TargetProfile::min_payoff_ratio` |
| `candidate_pbo_cap` | 1.0 / 0.7 / 0.5 | `DiscoveryConfig::max_pbo` (the per-candidate CPCV number — a *different object* from the judge's session CSCV PBO; `trial_returns.rs` module doc says so) |

`TargetProfile::min_net_expectancy_per_trade` is **not** in the vector. It is
unconditional in `TargetProfile::evaluate` and it is the floor under everything
else; a loop that could raise or lower it would be varying the one gate that says
"the average trade must make money."

### 7.10 The honest statement that belongs in the report

B1–B8 are the only families that are not re-parameterisations, **and even they
cannot create edge.** They can only find edge if it exists in a sub-population
(B1, B2), at a different horizon (B4), in path structure (B5, B6, B8), or if it
was being hidden by cost mis-scoring (B3) or by noise (B7). If all eight come
back non-positive after the shuffle control, that **is** the answer, and §10's R2
is how the loop says so.

---

## 8. The judge

Two stages. Stage 1 runs on every sweep and never touches the OOS window.
Stage 2 runs at most once per session per OOS window.

### 8.1 Stage 1 — SCREEN (in-sample)

Inputs per search `p` within a sweep: `TrialStatisticsReport` from
`deflated::analyse_matrix(&matrix, trials_offered = N_session)`, the ten
rejection counters, the `CostBandCensus`, and the best survivor's metrics.

```
SCREEN(p) ⟺
  S1  pbo_sweep(p)                     is Some(x)  ∧  x ≤ PBO_MAX = 0.50
  S2  dsr(p ; N_session)               is Some(d)  ∧  d ≥ DSR_MIN = 0.95
  S2b excess_over_expected_max_per_period(p)       >  0
  S3  cost_band_discriminates(band, baseline)      =  true
      ∧ census.optimistic_edge_only    =  0
      ∧ census.survives                ≥  1
  S4  E_screen_pess(p)                 >  0
  S5  n_trades(p)                      ≥  N_MIN_SCREEN
  S6  E_screen_pess(p)                 ≥  Q_shuffle_0.95(session)    [§9]
```

* `E_screen_pess` = per-trade net expectancy of the best survivor, charged at the
  **pessimistic** edge of `cost_band_pips`.
* **A refusal is a FAIL.** `dsr_refusal.is_some()` ⇒ S2 fails.
  `pbo_refusal.is_some()` ⇒ S1 fails. `TrialStatisticsReport::unreadable` ⇒ the
  whole screen fails. "No number" is never good news — that is the module's own
  doctrine and this judge enforces it.
* **S2b** exists because the DSR's cross-trial Sharpe variance is estimated from
  *this sweep's* rows while `trials_n` is the session-wide count. That mismatch
  understates the variance and therefore understates `SR*`, which **flatters**
  the DSR. `excess_over_expected_max_per_period > 0` is the term that does not
  depend on the variance estimate, and it is what hardens the conjunction. The
  direction of the bias is written into the sweep record's `assumptions`.
* **S6 is unavailable** until the session null has `MIN_NULL_OBS = 3`
  observations. While unavailable, `SCREEN` returns `Unavailable`, which counts
  as a failure for the proposer (§5.3) and blocks promotion.

`N_session` is defined once, in `session.rs`, as a fold:

```rust
fn n_session(journal: &[Record]) -> usize {
    journal.iter().filter_map(|r| match r {
        Record::SweepCompleted { trials_offered, .. }
      | Record::SweepFailed    { trials_offered, .. }
      | Record::SweepAbandoned { trials_offered, .. }
      | Record::ShuffleControlCompleted { trials_offered, .. } => Some(*trials_offered),
        _ => None,
    }).sum()
}
```

Live, control, failed and abandoned all count. There is no code path that
decrements it and no field that stores it — it is recomputed from the journal on
every use, which is what makes *"a loop that resets its own N between sweeps"*
structurally impossible rather than forbidden by policy.

### 8.2 The OOS window — defined once, touched once

* Defined in the session header: the final `OOS_FRACTION = 0.20` of the symbol's
  bar span, by **time**, after the frozen purge and embargo.
* **No sweep may load a bar inside it.** Enforced twice: the runner bounds the
  loaded span, and a hard assertion at S4 compares the loaded span's last
  timestamp against `oos_start_ms`. An overlap is a **hard error that aborts the
  session**, not a warning — a leaked OOS window cannot be un-leaked.
* **No search ever runs on OOS data. Only evaluation.** Stage 2 takes the
  promotion candidate's ordered batch-bound genes. Every gene retains its source
  ordinal/cursor and the batch-local post-prefilter feature-name table that gives
  its indices meaning; OOS projects by those names and backtests the result.
  Nothing is fitted, nothing is selected.
* The touch budget is `OOS_TOUCHES_TOTAL = 1` per `(scenario, window)`. Once
  spent, the session stops — whether the promotion passed or failed. A
  configuration that touches the OOS window more than once has spent it.

### 8.3 Stage 2 — the promotion inequality

One expression. It cannot drift into prose.

```
PROMOTE(s) ⟺
      SCREEN(s)                              = true
   ∧  pbo_sweep(s)                           ≤ 0.50
   ∧  pbo_session                            ≤ 0.50
   ∧  dsr(s ; N_session)                     ≥ 0.95
   ∧  excess_over_expected_max_per_period(s) > 0
   ∧  E_oos_pess(s)                          > 0
   ∧  E_oos_pess(s) − 2 · SE_oos(s)          > 0
   ∧  band_verdict_oos(s)                    = SurvivesBand
   ∧  n_oos(s)                               ≥ N_MIN_OOS
   ∧  oos_touches_spent(window)              = 0        (before this evaluation)
   ∧  goal_metric(s)                         ≥ goal_bar(scenario)
```

with

```
E_oos_pess(s)  = mean per-trade net expectancy of s's portfolio on the OOS
                 window, charged at the PESSIMISTIC edge of cost_band_pips
SE_oos(s)      = stderr of that mean over the OOS trades
                 (the "− 2·SE" term is min_expectancy_t_stat ≥ 2 applied OUT of sample)
pbo_session    = deflated::pbo_cscv(session_champion_matrix)
                 where session_champion_matrix is a DecodedTrialMatrix built in
                 memory with ONE ROW PER SWEEP — the sweep champion's monthly
                 return series. Its fields are all pub, so it is constructed, not
                 encoded. This is PBO of the LOOP'S OWN selection procedure, which
                 is the object the operator's instruction names; pbo_sweep is PBO
                 within one sweep. Both must clear 0.50.
```

and, per scenario,

```
scenario = Risky:
    r        = OOS per-trade R-multiples, pessimistic cost edge
    report   = goal_report::build_report(r,
                   start   = system.risky_start_balance_usd,
                   target  = system.risky_target_balance_usd,
                   horizon = system.risky_horizon_days as f64,
                   trades_per_day = measured on the OOS window,
                   DEFAULT_RISK_LEVELS, seed = session_seed)
    f*       = report.best_risk_fraction
    goal_metric(s) = report.frontier[f*].p_reach_target
    goal_bar       = P_REACH_MIN = 0.50
    AND additionally  report.frontier[f*].p_ruin ≤ P_RUIN_MAX = 0.50

scenario = PropFirm:
    goal_metric(s) = fraction of OOS months whose net return
                     ≥ risk.monthly_profit_target_pct
    goal_bar       = models.prop_firm_min_pass_rate
    AND additionally  no OOS window violates
                     prop_firm_gate.max_daily_loss_pct
                  or prop_firm_gate.max_overall_drawdown_pct
                  and min_trading_days is met
```

`P_REACH_MIN = 0.50` is chosen and stated: below one half, the **median path does
not reach the target**, and a "goal reached" claim backed by `P(reach) = 0.2` is a
lottery ticket sold as a result. `P_RUIN_MAX = 0.50` likewise — the frontier is
required to be reported in full in either case, so the operator sees the whole
curve and not just the conjunct that passed.

### 8.4 The thresholds, and their immutability

```rust
pub struct JudgeThresholds {
    pub pbo_max: f64,            // 0.50
    pub dsr_min: f64,            // 0.95
    pub p_reach_min: f64,        // 0.50
    pub p_ruin_max: f64,         // 0.50
    pub oos_fraction: f64,       // 0.20
    pub oos_touches_total: u32,  // 1
    pub n_min_screen: usize,
    pub n_min_oos: usize,
    pub min_null_obs: usize,     // 3
    pub oos_t_stat_min: f64,     // 2.0
}
```

These are **judge** thresholds, not goals — but they are equally read-only to the
loop. They are hashed into `judge_hash`, written into the session header, and
verified on every resume. Changing one starts a **new session id**: results
judged by two different judges are not comparable and must never share a history.
`PBO > 0.5` means the selection procedure is worse than random, which is why that
threshold in particular is not a tuning knob.

---

## 9. The shuffle control

### 9.1 Cadence

`SHUFFLE_PERIOD = 4` live sweeps form a **block**. At each block boundary the
loop re-runs **the block's best-screening sweep's exact 100 proposals**, byte for
byte, against rotated features. `SHUFFLE_REPLICATES = 1` per block, with a fresh
rotation offset each time; the null accumulates across blocks.

Cost: 1 control sweep per 5 = **20% of the budget**. Stated plainly and defended:
the alternative is a session whose entire output is unfalsifiable, which is worth
0% of the budget. Comparing against the block's *best* is the right comparison
because the loop reports maxima, and a null built from average runs would be too
easy to beat.

### 9.2 The permutation, exactly

**The control is a joint circular rotation of the entire feature block by a
uniformly random offset `τ ∈ [0.05·T, 0.95·T]`, wrapping. Prices, labels, costs,
exit geometry, gene encoding and the GA seed are untouched.**

Why rotation and not an i.i.d. row shuffle — this was a real choice:

| | preserves marginals | preserves cross-feature structure | preserves each feature's autocorrelation | destroys feature→future alignment |
|---|---|---|---|---|
| i.i.d. row permutation | yes | no (if per-column) / yes (if joint) | **no** | yes |
| **circular rotation** | **yes** | **yes** | **yes** | **yes** |

An i.i.d. permutation destroys autocorrelation, so signals flicker bar to bar,
trade counts explode, and the control pays a completely different cost per trade
than the live sweep. The comparison then measures trade frequency, not
predictability — the control becomes trivially easy to beat and the falsification
is worthless. Rotation is the **conservative** control: it leaves everything
intact except the one thing being tested.

A **second, weaker** control is run once per session as a diagnostic: an
independent per-column permutation, labelled `ControlKind::ColumnPermutation`.
It is reported and never enters any inequality. Rotation
(`ControlKind::CircularRotation`) is the one in S6 and in R2.

### 9.3 What is compared, exactly

For each control sweep `c` the loop computes **the same screen statistics on the
same in-sample window** as the live sweeps. The control **never touches the OOS
window** — spending OOS touches on the null would burn the one thing the design
protects. So the shuffle term lives on the screen side (S6), and the OOS
conjuncts of §8.3 stand on their own.

Per block, three recorded numbers:

```
delta_expectancy(block) = best_live_E_screen_pess(block) − control_E_screen_pess(block)
p_block(block)          = 1 if control_E_screen_pess ≥ best_live_E_screen_pess else 0
                          (one observation; the session-level p-value is the mean
                           over blocks, resolution 1/blocks)
dsr_gap(block)          = dsr_live_block_best − dsr_control
```

The session null is the multiset
`{ control_E_screen_pess(b) : b ∈ blocks so far }`, and

```
Q_shuffle_0.95(session) = the 95th percentile of that multiset,
                          or UNAVAILABLE while |blocks| < MIN_NULL_OBS = 3
```

A block is **INDISTINGUISHABLE** iff `delta_expectancy ≤ 0` **or**
`p_block = 1` **or** `dsr_gap ≤ 0`.

### 9.4 The consequence, in the same breath as the best result

Every report the loop emits — success, failure, or interim — carries the
`ShuffleSummary` block immediately under the headline number:

```
blocks: 11   indistinguishable: 3 (27%)   session p: 0.27
best live  E_screen_pess: +0.42 pips
best shuffle E_screen_pess: +0.31 pips     Q_0.95(null): +0.36 pips
```

*"If the loop's winners score like the shuffle's winners, the loop is mining
noise and must say so in the same breath as its best result"* — this block is
that sentence, rendered mechanically so it cannot be omitted.

And it feeds back: an `INDISTINGUISHABLE` block retracts every posterior success
credited from it (§5.3, mechanism 3).

---

## 10. The stopping rules

Both produce a `SessionVerdict` — **the same struct, the same renderer, the same
fields.** A failure report that looks different from a success report is a
failure report nobody reads.

```rust
pub struct SessionVerdict {
    pub schema: &'static str,          // "neoethos.autoresearch.verdict.v1"
    pub verdict: Verdict,              // the tag below
    pub session_id: String,
    pub goal_hash: String,
    pub judge_hash: String,
    pub cost_hash: String,
    pub scenario: ScenarioSummary,     // start/target/horizon or monthly bar, by field path
    pub sweeps_run: usize,
    pub searches_run: usize,
    pub n_trials_session: usize,       // the honest N
    pub best_ever: BestEver,           // sweep id, config_hash, all screen statistics
    pub shuffle: ShuffleSummary,       // §9.4, always present
    pub coverage: CoverageSummary,     // per axis-A level, per axis-B variant
    pub census: AbandonCensus,         // §12 — every dropped configuration, named
    pub oos: Option<OosEvaluation>,    // present iff a touch was spent
    pub reproduction: String,          // the exact command that re-derives this
}

pub enum Verdict {
    GoalReached { sweep: SweepId, goal_metric: f64, goal_bar: f64,
                  frontier: Vec<RiskOutcome> },
    GoalUnreachableInSearchedSpace { space: SpaceDescription },
    Inconclusive { reason: InconclusiveReason },   // budget / operator stop
}
```

### 10.1 R1 — the goal is reached

```
R1 ⟺ ∃ s : PROMOTE(s)
```

The session **stops**. It does not keep looking for a better one: a second
promotion needs a second OOS window, and the first result is already the honest
answer. The loop writes `proposal.json` (the portfolio, its canonical feature
names, its stamp, its full statistics) and the verdict. It does **not** write
`live_portfolio.json`, does not contact a broker, does not size a position. The
operator promotes.

### 10.2 R2 — the goal is provably unreachable in this space

All of the following, each checkable in code:

```
U1  sweeps_run ≥ K_MIN = 40                    (= 4,000 searches, ≥ 8 blocks,
                                                 ≥ 8 null observations)
U2  max over all sweeps of E_screen_pess(best) ≤ Q_shuffle_0.95(session)
        — the loop never once beat its own noise
U3  indistinguishable_blocks / blocks ≥ 0.75
U4  every axis-B variant drawn ≥ B_MIN_DRAWS = 3 sweeps' worth
        ∧ every axis-A factor level drawn ≥ 1 sweep's worth
        — a space is not refuted if it was not searched
```

Plus one independent terminator that is also a "this space was searched" result:

```
U5  ProposerExhausted: PROPOSER_RETRIES = 64 consecutive uniform draws all
    collided with an existing config_hash — the reachable space is enumerated
```

Verdict: `GoalUnreachableInSearchedSpace { space }`, where `space` is the
**explicit list of factor levels and objective variants covered, with their draw
counts.** "Unreachable" is only ever a claim about a searched space and the
report must say which one. This is the result the operator has spent sixteen
months not being able to get.

### 10.3 The third stop, which is not a verdict about the goal

`Inconclusive { BudgetExhausted | WallClock | OperatorStop | InfrastructureAbort }`
— `max_sweeps` or wall clock ran out with neither R1 nor R2 satisfied, or a hard
abort fired (§13). Same report shape. It says **explicitly**: *"this is not
evidence that no edge exists; U1–U4 were not all met, and here is which ones were
not."*

---

## 11. Persistence

### 11.1 Layout

Root: resolved through the existing store resolver
(`%LOCALAPPDATA%\neoethos\autoresearch\`), **never a hardcoded path**.

```
<root>/<session_id>/
  session.json                 header, written ONCE, immutable, fsynced
  journal.jsonl                append-only, one JSON object per line, fsynced per record
  sweeps/<sweep_id>/
      proposals.json           the 100 stamped proposals, in draw order
      statistics.json          per-search TrialStatisticsReport
      censuses.json            ten rejection counters, CostBandCensus, batch ledger
      promotion/slot_NNN.json  v4 ordered batch bindings: canonical input receipt +
                               receipt SHA-256 + exact evaluated window + effective
                               search-config hash + batch-local feature names +
                               ordinal/cursor-tagged local genes. There is no flat or
                               canonical gene list. The slot's proposal config_hash
                               remains a separate top-level stamp.
                               ONE PER SEARCH, not one per sweep: a sweep runs 100
                               searches and each selects its own portfolio, so a
                               single file per sweep could only describe one of them.
                               Written at S5 by the search that selected the genes;
                               a search that selected nothing writes NOTHING, so
                               "no evidence" has exactly one shape. Read once, by
                               the promotion path, BEFORE the out-of-sample touch is
                               journalled as spent — and strict-refused unless every
                               batch binding and the proposal stamp validate exactly.
      trial_returns.bin        KEPT ONLY for best-ever / promotion / control (§3.2)
  session_champions.json       one champion row per sweep (the pbo_session input)
  verdict.json                 written ONCE, at stop
```

### 11.2 Format decision

**Append-only JSONL journal + a small immutable JSON header + per-sweep artifact
directories.** Rejected: one big JSON rewritten each sweep — this repository has
`exit 137` OOM kills and multi-hour runs that ended with no artifact, and a
whole-file rewrite loses the entire session in exactly the failure mode that
actually happens. Rejected: a database — a dependency, a lock, and a file a
human cannot read with `tail`.

Crash safety: each record is serialised to one line, written, and fsynced before
the work it describes begins (intent) or after it completes (outcome). On replay,
a **trailing partial line is truncated and recorded** as
`TruncatedTail { bytes }` in the resumed header — counted, never silently
absorbed, exactly as `trial_returns.rs` handles its own truncated tail.

### 11.3 The journal records

`schema: "neoethos.autoresearch.journal.v1"`, one tagged enum:

```
SessionOpened { session_id, session_seed, goal_hash, judge_hash, cost_hash,
                goals (all field paths + resolved values), judge thresholds,
                oos_window { start_ms, end_ms }, priors, budget }
ProposalDrawn { sweep, slot, proposal, origin }
ProposalDuplicate { sweep, slot, config_hash, first_seen_sweep }
ProposalRefused { sweep, slot, config_hash, reason }      // e.g. unreachable payoff floor
SweepStarted { sweep, sweep_hash, started_ms }            // INTENT
SweepCompleted { sweep, trials_offered, per_search: [...], wall_ms }
SweepFailed { sweep, trials_offered, error }
SweepAbandoned { sweep, trials_offered, reason }          // written on resume
Screened { sweep, slot, screen_result, failing_conjunct }
PosteriorUpdated { sweep, credits: [(factor, level, success)] }
PosteriorRetracted { block, sweep_ids, reason: "block indistinguishable" }
ShuffleControlCompleted { block, kind, tau, source_sweep, trials_offered, statistics }
BlockJudged { block, delta_expectancy, p_block, dsr_gap, indistinguishable }
OosTouchSpent { window, sweep, candidate }
Promoted { sweep, goal_metric, goal_bar, frontier }
PromotionRefused { sweep, failing_conjunct }
SessionStopped { verdict }
```

### 11.4 The resumability invariant

> **The in-memory `Session` is a pure fold over the journal. No state exists
> anywhere else.**

```rust
pub fn fold(records: &[Record]) -> Result<Session>;
```

Enforced by a test that, after **every** appended record, asserts
`fold(journal) == session_in_memory`. This is what makes "a crash loses one
sweep, not the session" a property rather than an intention, and it is also what
makes `N_session` un-resettable (§8.1).

Resume refuses, naming the field, when `goal_hash`, `judge_hash` or `cost_hash`
differ from the header. It does not migrate and does not adapt. Two judges, two
sessions.

---

## 12. No silent drops

Every abandoned configuration is counted **and named**, in one census carried in
every report:

```rust
pub struct AbandonCensus {
    pub proposals_drawn: usize,
    pub duplicates_refused: usize,           // with first_seen_sweep for each
    pub payoff_floor_unreachable: usize,     // refused at S3 by assert_payoff_floor_reachable
    pub trust_region_reverted: usize,
    pub searches_errored: usize,
    pub sweeps_failed: usize,
    pub sweeps_abandoned: usize,             // process death, found on resume
    pub screen_failed_by_conjunct: [usize; 7],   // S1..S6 + Unavailable
    pub gc_deleted_matrices: usize,          // with bytes reclaimed and the rule
    pub examples: Vec<(String, String)>,     // (reason, config_hash), bounded, named
}
```

Shaped after `IndicatorLedger` and the batch ledger — reason, count, named
examples, one census line — because that is the shape this project has already
settled on, and because *a batch rejected with no record is the silent drop
again, one level up.*

---

## 13. Contracts consumed, and how absence fails

`contracts.rs` exists for one purpose: **an absent symbol must fail at compile
time, and an absent behaviour must fail at startup, naming itself.** Never a
silent no-op that makes the loop appear to run while doing nothing.

### 13.1 Compile-time (a `use` of every symbol the loop depends on)

| Symbol | Path |
|---|---|
| `run_streaming_working_set`, `StreamingPlan`, `StreamingRunOutcome`, `BatchSearchResult`, `CanonicalSurvivor` | `neoethos_search::orchestration` |
| `CanonicalFeatureIndex`, `BatchLedgerEntry`, `BatchOutcome`, `StreamingRunLedger` | `neoethos_search::batch_ledger` |
| `batch_rejection_ledger`, `BatchRejectionLedger` | `neoethos_search::discovery` |
| `DiscoveryConfig`, `TargetProfile`, `TargetProfileRejection`, `DiscoveryMode`, `CostBandCensus`, `CostBandVerdict`, `cost_band_discriminates` | `neoethos_search::discovery` |
| `ResolvedConfigStamp`, `assert_payoff_floor_reachable`, `max_achievable_payoff`, `PayoffCeilingInputs`, `MEASURED_TRAILING_PAYOFF_CEILING` | `neoethos_search::run_identity` |
| `analyse_matrix`, `analyse_run`, `deflated_sharpe`, `pbo_cscv`, `DecodedTrialMatrix`, `DecodedTrialRow`, `TrialStatisticsReport`, `DeflatedSharpeReport`, `PboReport` | `neoethos_search::deflated` |
| `TrialReturnsManifest`, `load_manifest`, `month_keys_spanning`, `period_returns`, `TRIAL_RETURNS_MAGIC` | `neoethos_search::trial_returns` |
| `build_report`, `GoalReport`, `RiskOutcome`, `DEFAULT_RISK_LEVELS` | `neoethos_search::goal_report` |
| `Gene` | `neoethos_search::genetic` |
| `Settings`, `PropFirmPreset`, `PropFirmConstraints`, `PropFirmGateConfig` | `neoethos_core::config`, `neoethos_core::domain::prop_firm` |

The two APIs landing in parallel — **the streaming inner loop with the
`Gene.expectancy` early-reject predicate** (`crates/neoethos-search/src/genetic/strategy_gene.rs:17`,
booked after spread, commission and swap) and **the DSR/PBO reader** — are the
first two rows and the sixth. If either is absent the crate does not compile, and
the error names the symbol. That is the required behaviour.

`DecodedTrialMatrix`'s fields are all `pub`, so the session champion matrix
(§8.3) is **constructed in memory** and handed to `pbo_cscv` / `deflated_sharpe`
directly. No encoder is needed and no change to `trial_returns.rs` is required.

### 13.2 Startup self-check (behavioural absence)

```rust
pub fn startup_selfcheck(cfg: &ResolvedInputs) -> Result<(), MissingCapability>;
```

Each returns a named error and **aborts the session**:

1. `GoalSet::load` refusals G1–G6 (§1.3).
2. `cost_band_discriminates(band, baseline) == false` → abort naming baseline,
   optimistic edge, pessimistic edge (§7.3).
3. The OOS window is empty, shorter than `N_MIN_OOS` trades' worth of bars, or
   overlaps the loaded search span.
4. `assert_payoff_floor_reachable` fails on the **baseline** configuration → the
   space cannot express the question; abort naming the binding constraint.
5. **Sweep 1 produced no readable trial-returns matrix** → abort. `unreadable` on
   the first live sweep means the wiring is absent, not that the configuration
   was bad. On any later sweep it is that sweep's failure, counted and named.
6. The streaming plan was requested but `batch_columns == 0` on every sweep of
   block 1 → abort: the loop is not sweeping and would silently be running a
   single-pass search under a streaming label.

---

## 14. REQUIRED ELSEWHERE

Edits needed in forbidden territory, written out exactly. Whoever owns the file
applies them; this workflow does not.

### 14.1 `crates/neoethos-core/src/config.rs` — the second risky scenario

The 1,000 → 200,000 scenario exists only in
`crates/neoethos-search/examples/goal_frontier_demo.rs:36`. Make it a read-only
config citizen so the loop can enumerate it without inventing it.

Add to `SystemConfig`, directly under `risky_horizon_days` (currently `:149`):

```rust
    /// ADDITIONAL read-only Risky scenarios, beyond the primary
    /// `risky_start_balance_usd` / `risky_target_balance_usd` /
    /// `risky_horizon_days` triple. The operator's second stated scenario is
    /// 1,000 -> 200,000 in 180 days (x200), which until now existed only in
    /// `neoethos-search/examples/goal_frontier_demo.rs:36`.
    ///
    /// GOALS ARE READ-ONLY. Nothing in the codebase may write this field; the
    /// autoresearch loop enumerates it and hashes it (docs/autoresearch-loop.md
    /// §1.1). Empty by default so existing configs are untouched.
    #[serde(default)]
    pub risky_scenarios: Vec<RiskyScenario>,
```

and the type, beside `SystemConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RiskyScenario {
    pub label: String,
    pub start_balance_usd: f64,
    pub target_balance_usd: f64,
    pub horizon_days: u32,
}
```

and in `config.yaml`, under `system:`:

```yaml
  risky_scenarios:
    - label: "1000 -> 200k (x200)"
      start_balance_usd: 1000.0
      target_balance_usd: 200000.0
      horizon_days: 180
```

Also add `"system.risky_scenarios"` to the read-only / not-a-limit list beside
`"risk.monthly_profit_target_pct"` (`config.rs:3485`).

### 14.2 `crates/neoethos-cli/src/main.rs` — the entry point

```
neoethos autoresearch \
    --symbol <SYM> \
    --max-sweeps <N>            # default 200 (= 20,000 searches)
    --max-hours <H>             # wall-clock budget
    [--session <ID>]            # resume; without it, a new session id
    [--scenario <LABEL>]        # default: system.trading_mode's primary scenario
    [--dry-run]                 # draw and stamp proposals, run nothing
```

It calls exactly one function and does nothing else:

```rust
neoethos_autoresearch::runner::run(
    neoethos_autoresearch::RunArgs { symbol, max_sweeps, max_hours, session, scenario, dry_run },
    &settings,
)?;
```

The subcommand **must not** accept any flag that names a goal constant, a cost
field, or a judge threshold. If it did, the loop's goals would be writable from
the command line, which is the same defect wearing a different hat.

### 14.3 `crates/neoethos-search/src/run_identity.rs` — the GA seed in the stamp

`ResolvedConfigStamp` must include the GA `seed` so that two proposals differing
only in `replicate_seed` (§6) produce **different** `config_hash`es and are not
refused as duplicates — while two proposals identical in everything including the
seed still collide, which is the property the dedupe needs.

Add to `ResolvedConfigStamp`, beside the other resolved values, and include it in
the hash:

```rust
    /// GA seed. Two runs that differ ONLY in the seed are DIFFERENT EXPERIMENTS
    /// (they are replicates, and replicate variance is what the judge needs to
    /// estimate), so the seed is part of the identity. Before this field, a
    /// replicate was indistinguishable from a re-run.
    pub ga_seed: u64,
```

If this is not applied, `proposer.rs` must treat `(config_hash, replicate_seed)`
as the identity pair and say so loudly in the session header
(`identity_source: "config_hash + external replicate_seed (run_identity.rs lacks ga_seed)"`).
Do **not** let it silently dedupe replicates away.

### 14.4 `config.yaml` — a cost band that can discriminate

`cost_band_discriminates` is currently **false** at the shipped configuration:
baseline `1.5 + 0.5 + 14/10 = 3.4` pips against band edges `1.6 / 2.4`, both
cheaper than the cost already charged (`discovery.rs`, `cost_band_discriminates`
doc). §7.3/§13.2 abort on this. The band's pessimistic edge must exceed the
baseline round-trip cost — set the edges from the broker's measured worst-case
spread plus commission, not from a guess.

### 14.5 `config.yaml` — the prop-firm goal bars are zero

`risk.monthly_profit_target_pct: 0.0` (`config.yaml:194`) and
`models.prop_firm_min_pass_rate: 0.0` (`config.yaml:160`) contradict the FTMO
preset's `min_monthly_net_profit_pct = 0.04` and the shipped default `0.40`. The
loop aborts on both (G3, G4) when `trading_mode = "prop_firm"`. Restore them to
`0.04` and `0.40`, or state on the record that prop-firm mode is out of scope for
autoresearch. **The loop will not choose for you** — that would be writing the
goals.

### 14.6 `crates/neoethos-search/src/orchestration.rs` — the working-set cursor

Axis A's `working_set_cursor` factor (§6) has three levels — `0 / resumed /
random offset` — and **only the first is expressible today**.
`run_streaming_working_set` takes no cursor, and `StreamingSearch::new` always
starts at 0; `StreamingSearch::cursor()` is read-only. So a sweep cannot resume
where the last one stopped, and every sweep re-searches the prefix of the
(indicator, period) space.

Until this lands, `proposer.rs` **excludes the two unreachable levels from the
drawable space** and names them in the space report (with the symbol, the cargo
feature, and this section), rather than drawing them and silently starting at 0.
U4 is then evaluated over the reachable space, so `GoalUnreachableInSearchedSpace`
can never claim coverage of a level that was never searchable.

```rust
// StreamingPlan — add beside `max_batches`:
    /// Where in the (indicator, period) space this run's working set STARTS.
    /// `0` reproduces today's behaviour exactly. A sweep that resumes where the
    /// previous one stopped is the only way an outer loop can cover a space
    /// wider than one sweep's batch budget; without it every sweep re-searches
    /// the prefix and reports the repeat as new coverage.
    pub start_cursor: usize,
```

```rust
// StreamingSearch — add beside `new`:
    pub fn resume_at(budget_rows: usize, cursor: usize) -> Self {
        let mut me = Self::new(budget_rows);
        me.cursor = cursor.min(me.space_len);
        me
    }
```

and `run_streaming_working_set` builds its `StreamingSearch` with
`StreamingSearch::resume_at(budget_rows, plan.start_cursor)`. **It must still
never wrap** — running off the end returns `None` exactly as today.

Turn the loop's side on with `--features neoethos-autoresearch/search-streaming-start-cursor`;
the feature compiles a probe that reads `StreamingPlan::start_cursor`, so it
cannot be enabled before the field exists.

### 14.7 `crates/neoethos-search/src/discovery.rs` — the objective knobs

**Seven of the nine axis-B variants are not expressible against the compiled
search**, because `DiscoveryConfig` has no objective selector: the GA's fitness
is chosen by one `bool` (`EvaluationConfig::growth_objective`, set from
`DiscoveryMode`), the triple-barrier horizon is read from
`EvaluationConfig::max_hold_bars` which **ships at 0 and has no config recipient**
(`discovery.rs:5483` then substitutes the documented 35), and there is no
conditioning set, no cost-edge scoring flag and no portfolio-level objective.

Today, therefore, exactly one axis-B variant is drawable per scenario — B5 under
Risky (`ga_fitness_growth`, the mode's own Kelly log-growth objective) and B6
under PropFirm (the window-pass gate) — plus the whole refusal vector (§7.9),
whose four dimensions all have recipients already. `proposer.rs` reports the
other seven as EXCLUDED, by name, with the symbol each needs; it does **not**
draw them and quietly run the default objective, which would make the loop
appear to explore the objective space while sweeping axis A under one implicit
objective.

Add to `DiscoveryConfig`, each read-only to the loop and each carrying its own
cargo feature on the autoresearch side:

```rust
    /// Which named scoring table the GA maximises. Today the choice is a single
    /// `bool` derived from the mode, so the four tables in `scoring/named.rs`
    /// cannot be compared and B0/B1/B7 cannot be expressed.
    pub fitness_table: FitnessTable,          // feature: search-fitness-table
                                              //          search-in-market-fitness (NetPerBarInMarket)
                                              //          search-t-stat-objective  (ExpectancyTStat)
    /// Triple-barrier vertical barrier, in base bars, for BOTH the prefilter's
    /// label and the evaluator. `0` keeps today's behaviour (the documented 35).
    pub label_max_hold_bars: usize,           // feature: search-label-horizon
    /// Score candidates at the PESSIMISTIC edge of `cost_band_pips` rather than
    /// at the point estimate. Not a monotone re-ranking: cost scales with trade
    /// count.
    pub score_at_cost_band_edge: bool,        // feature: search-cost-edge-scoring
    /// A conditioning set declared BEFORE the run, from the fixed family in
    /// `objective.rs::B2_CONDITIONING_LEVELS`. The complement's expectancy is
    /// reported alongside, always, and the family's bucket count multiplies into
    /// the trial count the DSR deflates against.
    pub conditioning_set: Option<String>,     // feature: search-conditioning-set
    /// Score the PORTFOLIO after correlation pruning, not the best single gene.
    pub score_portfolio_after_pruning: bool,  // feature: search-portfolio-objective
```

with

```rust
pub enum FitnessTable { GaFitness, ArchiveScore, WindowScore, QualityScore,
                        NetPerBarInMarket, ExpectancyTStat }
```

`FitnessTable::NetPerBarInMarket` is the one new function: net divided by bars
in market rather than by trades, so **a strategy that is flat is not penalised**
— which is the whole content of B1, and the only objective shape that can act on
"the average trade loses 4.15 pips, so take fewer, different trades."

### 14.8 `crates/neoethos-search/src/run_identity.rs` — six factors are invisible to the stamp

Beyond §14.3's `ga_seed`: `ResolvedConfigStamp` does not carry
`max_indicators`, `higher_timeframes`, `corr_threshold`, `min_trades_per_day`,
the streaming plan, or the working-set cursor. Six of the eleven axis-A factors
and the whole streaming dimension therefore **produce identical `config_hash`es
for genuinely different experiments**, which breaks the property the stamp
exists for: *two runs with the same hash searched under the same decisions.*

The proposer does not paper over it. It dedupes on its own complete semantic key
(`proposal::ProposalKey`), so no part of the space is lost, and it counts and
names every collision (`stamp_collisions`, with the differing factors listed and
a `WARN` naming this section) rather than absorbing it. Add to the stamp body,
in declaration order, and include each in the hash:

```rust
    pub max_indicators: usize,
    pub higher_timeframes: Vec<String>,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
    pub max_pbo: f64,
    pub streaming_max_batches: usize,
    pub streaming_start_cursor: usize,
```

`max_pbo` matters as much as the rest: it is the per-candidate CPCV ceiling the
search refuses on (§7.9), so two runs that differ only there refused different
candidates and are not the same experiment.

---

## 15. Constants, in one place

```rust
// space / proposer
SWEEP_SEARCHES        = 100     // the operator's own unit
EXPLORATION_FLOOR     = 25      // per sweep, uniform, never annealed
MAX_DELTA_FACTORS     = 3       // trust region vs. parent
B_MIN_DRAWS           = 3       // per axis-B variant, before U4 can be claimed
PROPOSER_RETRIES      = 64
PRIOR                 = Beta(1, 1)

// shuffle
SHUFFLE_PERIOD        = 4       // live sweeps per block
SHUFFLE_REPLICATES    = 1       // control sweeps per block  -> 20% of budget
MIN_NULL_OBS          = 3       // blocks before S6 is available
TAU_RANGE             = [0.05·T, 0.95·T]

// judge (frozen, hashed into judge_hash)
PBO_MAX               = 0.50
DSR_MIN               = 0.95
OOS_FRACTION          = 0.20
OOS_TOUCHES_TOTAL     = 1
OOS_T_STAT_MIN        = 2.0
P_REACH_MIN           = 0.50
P_RUIN_MAX            = 0.50

// stopping
K_MIN                 = 40      // sweeps before R2 may be claimed
U3_THRESHOLD          = 0.75    // indistinguishable block fraction
DEFAULT_MAX_SWEEPS    = 200
```

`N_MIN_SCREEN` and `N_MIN_OOS` are **derived**, not constants: from
`min_trades_per_month × months_in_window`, so they scale with the window rather
than degrading differently on every dataset.

---

## 16. The five non-negotiables, and where each is enforced

| Non-negotiable | Enforcement, mechanical |
|---|---|
| 1. The loop may never write the goals, the cost model, or the gates | `GoalSet` has getters only; `SearchConfigDelta` has no variant naming a frozen field; `FROZEN_FIELDS` + a test asserting disjointness with everything `apply()` writes; the CLI has no flag for any of them (§14.2) |
| 2. Every sweep recorded with its `config_hash`, result and verdict | `ResolvedConfigStamp` per proposal at S3; `journal.jsonl` two-phase; `reproduction` string in every verdict |
| 3. Resumable — a crash loses one sweep, not the session | append-only JSONL, fsync per record, intent-before-work, `fold(journal) == session` asserted after every record, `SweepAbandoned` on resume |
| 4. No silent drops | `AbandonCensus` (§12) in every report, shaped after `IndicatorLedger`; `ProposalRefused` / `ProposalDuplicate` / `PosteriorRetracted` records |
| 5. Never places an order, never touches the broker, never writes `live_portfolio.json` | the crate does not depend on `neoethos-app` or `neoethos-trader`; its only writes are under `<root>/<session_id>/`; R1 emits `proposal.json` and stops |

---

## 17. REQUIRED ELSEWHERE — found while building the core loop

Added 2026-08-10 by the core-loop builder (session, state machine, resume). §14
is the design's own list; these five surfaced only once the loop was written
against the real APIs. Each is a change in **forbidden territory**; whoever owns
the file applies it. None of them is applied here.

Until they land, the loop's behaviour is stated beside each item — in every case
it is *fail loud* or *carry a mirror*, never a silent substitution.

### 17.1 `crates/neoethos-search/src/discovery.rs` — `CostBandCensus` is not `Serialize`

`CostBandCensus` (`discovery.rs:4653`) derives `Debug, Default, Clone, Copy,
PartialEq, Eq` but not `Serialize`/`Deserialize`, so it cannot be written into a
journal record.

```rust
-#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
+#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
 pub struct CostBandCensus {
```

**Until then:** `journal::CostBandCounts` is a field-for-field mirror with a
`From<&CostBandCensus>`. Delete the mirror when the derive lands.

### 17.2 `crates/neoethos-search/src/goal_report.rs` — `RiskOutcome` is not `Serialize`

Same problem, and it matters more: §8.3 requires the **whole frontier** to be
reported on a promotion, not just the risk level that passed.

```rust
-#[derive(Debug, Clone)]
+#[derive(Debug, Clone, Serialize, Deserialize)]
 pub struct RiskOutcome {
```

**Until then:** `journal::RiskOutcomeRecord` mirrors it.

### 17.3 `crates/neoethos-search/src/discovery.rs` — the ten rejection counters are private

`QualityScreenRejects` (`discovery.rs:4615`) and `BaseQualityReject`
(`:4458`) are both private, and `DiscoveryResult` carries no field holding them.
The design's S5 requires *"the ten rejection counters"* per search; the only
public route today is `FunnelProfile::stages[].top_reasons`, which is a
`Vec<(String, usize)>` of the top reasons per stage and is **not** guaranteed to
carry all ten.

```rust
-struct QualityScreenRejects {
+#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
+pub struct QualityScreenRejects {
```

plus a field on `DiscoveryResult`:

```rust
    /// The ten named base-quality rejection counters for this run. A single
    /// `rejected_base_quality` number cannot say WHICH gate did it, and the
    /// answer in the 0-of-174 run was one gate: the payoff floor.
    pub quality_rejects: QualityScreenRejects,
```

**Until then:** `SearchOutcome::rejections` is built from the funnel profile's
`top_reasons`, so the census is a superset-of-nothing rather than a guaranteed
partition. It is named `rejections: Vec<(String, usize)>` — named, not
positional — so widening it later does not shift any column.

### 17.4 `crates/neoethos-search/src/discovery.rs` — no public per-window portfolio evaluation

Stage 2 (§8.3) must **evaluate, never search**: take the promotion candidate's
already-selected genes and backtest them on the OOS window. The only sanctioned
per-gene settings resolvers (`GeneEvalSettingsResolver`,
`PopulationTemplateResolver`) are `pub(crate)`, and `discovery.rs` carries a test
that fails on any new call site of the raw builder — for a good reason: with
adaptive stops installed, a gene's effective SL is `stop_vol_mult ×` the
dataset's per-bar volatility, and screening with the raw builder evaluates a
**different strategy** from the one that was scored (measured: 30,331 trades
against 1,727 on one signal).

Needed:

```rust
/// Backtest an ALREADY-SELECTED portfolio on a bounded window. Nothing is
/// fitted and nothing is selected: this is the out-of-sample evaluation, and it
/// resolves each gene's effective stop through the SAME resolver the GA scored
/// with, so it cannot screen a different strategy from the one that was chosen.
pub fn evaluate_portfolio_on_window(
    genes: &[Gene],
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    from_ms: i64,
    to_ms: i64,
) -> Result<Vec<Trade>>;
```

**Until then:** `StreamingSweepExecutor::evaluate_oos` refuses — by name, with
the measured divergence quoted — whenever
`neoethos_search::stop_target::adaptive_stops_enabled()` is true. With adaptive
stops off it evaluates through `signals_for_gene` + `simulate_trades_core` using
the gene's own `sl_pips`/`tp_pips`, which are the effective stops in that
configuration. **A promotion is refused rather than approximated**, which is the
correct bias for a loop whose one failure mode is a confident overfit.

### 17.5 `crates/neoethos-search/src/discovery.rs` — the cost-band census does not reach the caller

`CostBandCensus` is computed inside the search but `DiscoveryResult` has no field
carrying it, so screen conjunct **S3** (`census.optimistic_edge_only == 0 ∧
census.survives ≥ 1`) has no input.

```rust
    /// What the candidates did across the round-trip cost band. Without it a
    /// caller cannot tell "profitable at the pessimistic edge" from "profitable
    /// only at the cheap end of a cost nobody can pin down".
    pub cost_band_census: CostBandCensus,
```

**Until then:** the loop reports every survivor as `unmeasured`, which **fails**
S3 rather than passing it silently. That is deliberate: an unmeasured band that
read as `survives` would make every census look clean, *"which is worse than no
census, because a reader takes it as evidence."*

### 17.6 Note on where the cost model is charged

Not an edit — a decision the core loop made and is recording so it is not
rediscovered as a surprise.

**Every search the loop runs is charged at the PESSIMISTIC edge of the frozen
`cost_band_pips`**: `evaluation_spread_pips = band.1`,
`evaluation_commission_per_trade = 0.0`. Consequences, all intended:

* `StrategyMetrics::profit_per_trade` **is** `E_screen_pess` — same number, same
  unit, no conversion, no assumed lot size and no second cost model to drift;
* the loop never computes an optimistic-edge number at all, so it cannot
  accidentally report one;
* it is not the loop varying a frozen field: the value is fully determined by
  `cost_band_pips`, which no `SearchConfigDelta` can name
  (`space::FROZEN_FIELDS`).

`E_screen_pess` is therefore in **account currency per trade**, not pips. §9.4's
worked example renders pips; the quantity is the same, the unit is the one the
backtest already books in.
