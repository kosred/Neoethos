# Search objective audit — is the search optimising for what the operator wants?

Date: 2026-08-08. Slice 4 of the search-correctness campaign.
Verdict up front: **no stage of the search rewards payoff ratio, the one
payoff-aware control that exists is switched off, and the Risky GA objective
is linear in trade count** — so the search selects volume × small-edge over
per-trade edge, on a cost model that undercharges the demo broker's costs.
That combination is the shape of the operator's measured live outcome
(win rate 44.3 %, payoff 0.803, costs 152 % of gross profit).

Nothing in this document changes selection. Every change below is a
DECISION for the operator, with its measured consequence attached.

---

## 1. What the system literally maximises (file:line)

Every recorded run on this machine is **Risky mode** (all 11 funnels in
`%LOCALAPPDATA%\neoethos\cache\discovery\*_funnel.json` say `"mode": "Risky"`),
so the left column is the objective that produced every artifact on disk.

### 1a. GA fitness — Risky lane (`ga_fitness_growth`, scoring_version 5)

`crates/neoethos-search/src/scoring/named.rs:218-261`

```
f*      = p·(pf−1)/pf, half-Kelly, capped 0.25          (named.rs:240-245)
rr      = pf·(1−p)/p          ← this IS the payoff ratio (named.rs:246)
g_trade = p·ln(1+rr·f) + (1−p)·ln(1−f)                  (named.rs:247-251)
growth  = g_trade × trades                               (named.rs:252)
fitness = growth × 10 + edge_gradient                    (named.rs:260)
```

- Payoff enters `g_trade` correctly — but `growth = g_trade × trades` is
  **LINEAR in trade count with no cap** (`named.rs:252`). Doubling the
  trade frequency at half the per-trade growth leaves fitness unchanged;
  at 60 % of the per-trade growth it WINS. The GA can always buy fitness
  with volume as long as the modeled per-trade edge stays ≥ 0.
- Deliberately no drawdown or worst-day penalty in this lane (`named.rs:214-217`).

### 1b. GA fitness — PropFirm/Strict lane (`ga_fitness`, v4 math)

`crates/neoethos-search/src/scoring/named.rs:89-188`

| term | weight | line |
|---|---|---|
| monthly ≥4 % hit-rate (slot 7) | ×0.45 — dominant | named.rs:161 |
| net / 20 000 (clamped ±2) | ×0.15 | named.rs:162 |
| Sharpe (clamped, conf-scaled) | ×0.10 | named.rs:163 |
| consistency | ×0.10 | named.rs:164 |
| profit factor (GA shape) | ×0.15 (×0.25 if pf<1) | named.rs:165-166 |
| **win rate: (wr−0.45)×2, cap 0.5** | ×0.10 | named.rs:167, ingredients.rs:176-179 |
| max-DD penalty ×15 (cap 5), unscaled | −dd | named.rs:168, ingredients.rs:120-123 |
| worst-day penalty ×10 | −daily_dd | named.rs:181-182 |
| activity multiplier 0.3+0.7·min(trades/30,1) | × positives | named.rs:142-143 |

**Payoff ratio appears in NO term.** Win rate has its own explicit reward;
payoff does not. Profit factor folds the two together (`pf = payoff·p/(1−p)`),
which is exactly the confounding the codebase itself documents at
`quality.rs:139-144` ("30 % at 5:1 and 70 % at 0.6:1 both give 2.1").
Expectancy is deliberately excluded from GA fitness
(`ingredients.rs:203-206` — "operator directive 2026-05-17").

### 1c. Post-GA ranking (income score)

`crates/neoethos-search/src/discovery.rs:4036-4094`

- Risky: `min(achievable/required, 3.0) × (0.7 + 0.3·min(fitness,1))`
  where `achievable = g_trade × trades_in_horizon` (discovery.rs:4068).
  Measured on the July-15 EURUSD M15 run: **both** the selected genes and
  the high-payoff genes saturate the 3.0 cap, so ordering falls to the
  tie-breakers — consistency, then `fitness` (discovery.rs:4102-4111),
  which is again `g_trade × trades` — volume decides.
- PropFirm/Strict: `fitness × (0.4·consistency + 0.3·win_rate + 0.2·safety
  + 0.1·pf_capped)` (discovery.rs:4079-4092). Payoff absent; win rate ×0.3.

### 1d. Quality screen ordering and score

`crates/neoethos-search/src/quality.rs:837-913` (`score_strategy`, 0-100):
sortino 30 + PF 20 + **win rate 15** (linear 0.45→0.70, quality.rs:848) +
calmar 20 + DD 15 + p-value 10 + monthly-WR 10 + monthly-return 10.
**Payoff ratio: 0 points.** Survivors are sorted into the correlation
pruner by this score (discovery.rs:4684-4701), so the final 50 slots go to
high-WR, high-volume, high-compounded-net genes first.

### 1e. Gates and floors (selection-shaping constants)

| gate | value | where |
|---|---|---|
| base filter (Risky floors) | max_dd 0.60, everything else ~0 | discovery.rs:755-761 |
| base filter (PropFirm floors) | permissive, judged by window gate | discovery.rs:685-703 |
| min_trades_per_day | live config **1.0**, floor .max(0.2) | live config:208, discovery.rs:546 |
| min_trades_per_month (strict lane) | live config **15** | live config:207, discovery.rs:1625 |
| prop-firm window bar | 8 %/60-day window | discovery.rs:806-807 |
| prop-firm pass-rate floor | default 0.40; live config **0.0** | discovery.rs:449, live config:330 |
| promotion gate | Sharpe ≥1, **WR ≥0.45**, PF ≥1.2, DD ≤25 %, trades ≥30 — no payoff floor | config.yaml:540-548 |
| **TargetProfile payoff gate** | `prop_search_min_payoff_ratio` — EXISTS, default **0.0 = OFF**, off in live config too | discovery.rs:1569-1607, core config.rs:589+1643, live config:209 |

The only payoff-aware control in the entire funnel is the TargetProfile
gate (`discovery.rs:1593`), built for exactly this purpose, and it has
never been on.

---

## 2. Against the operator's measured reality

Measured live: **win rate 44.3 %, payoff 0.803** → live PF = 0.639.
Break-even needs **55.4 % WR at payoff 0.803**, or **payoff 1.257 at WR 44.3 %**.

The July-15 EURUSD M15 run (the newest full artifact set) selected genes at
in-sample **payoff 1.31 / WR 0.649**. Break-even for that shape is WR 43.3 %
— the selected shape carries **one percentage point of win-rate margin**
before it loses money. The live measurement (44.3 %) sits exactly at that
cliff edge. A payoff-2.0 shape breaks even at WR 33 % — twenty points of
margin for the same in-sample story. The search systematically picks the
shape with no room for decay because nothing in it prices decay margin,
and payoff IS the decay margin.

## 3. Volume gaming — confirmed, with the mechanism

1. `ga_fitness_growth` is linear in n (`named.rs:252`) — see §1a.
2. The income-score cap (3.0) is saturated by everyone, handing the
   decision back to the volume-linear fitness (§1c).
3. Activity floors REQUIRE volume: live config demands ≥1 trade/day and
   ≥15 trades/month — a 3-trades/week high-payoff strategy is filtered
   out at `discovery.rs:1625` before anyone scores it.
4. Volume amplifies the cost-model error (§4): each modeled trade books
   understated costs, so the more trades, the larger the fictional edge.

Measured on the July-15 pool (22 269 quality-screen survivors): median
47 trades/month; the selected 34 sit at median 53/month with 470-3 328
trades each; the top of the pool by per-trade expectancy sits at
28/month — half the volume, twice the payoff (see §5).

## 4. The cost model vs the demo broker (IC Markets cTrader demo)

What the search charges per trade, every production lane
(`discovery_backtest_settings`, discovery.rs:1340-1370 — note the
`..BacktestSettings::default()` at 1368):

| cost | search charges | demo broker (data on this machine) | gap |
|---|---|---|---|
| spread | **flat 2.0 pips** every bar (`risk.backtest_spread_pips` 1.5 + `slippage_pips` 0.5, discovery.rs:523-525) | session-dependent; `SessionSpreadProfile` exists (eval.rs:94-111) but is **None in every production path** — only tests set it | overcharges London hours, undercharges rollover/news; strategy picks its own hours, the model can't see that |
| commission | **$7/lot flat** (`risk.commission_per_lot`, live config:121) | broker metadata: type 1, **$45/M USD** → $4.88/side at 1.084 ≈ **$9.76 round-turn** if per-side (symbol_metadata.rs:340-353; test at 1349 documents $4.95 at 1.10) | ~28 % undercharge if cTrader charges per side (verify against demo deal list) |
| swap | **always 0.0** — `EvaluationConfig` has no swap fields (strategy_gene.rs:747-776); `BacktestSettings::for_symbol` (eval.rs:421-451) is the only constructor that carries broker swap and has **zero production callers** | broker metadata EURUSD: **long −2.445 pips/night, short −0.105** (`data/symbol_metadata.json`) | every overnight hold is free in the search; kernel support exists end-to-end (eval.rs:624-629) and is dead |
| conversion fee | 0.0 (same wiring gap) | broker: 0.0 for EURUSD/USD-quote | none today, wrong for cross pairs |

The eval kernel itself charges correctly what it is given
(half-spread at entry + half-spread + commission at exit, eval.rs:839-896,
1088-1093; swap at exit, eval.rs:595-629). The inputs are what lie.

## 5. Measured: what a payoff-explicit objective would have selected

Artifacts: `EURUSD_M15.quality.json` (22 269 survivors, 4.19 GB, streamed) +
`EURUSD_M15.live_portfolio.json` (34 exported genes), run of 2026-07-14/15,
Risky mode. Payoff derived as `pf·(1−p)/p` (the artifact predates the
`payoff_ratio` field). Script: `scripts/measure_selection_objective.py`.

Proposed objective measured here: **E[R] × conf** where
`E[R] = wr·payoff − (1−wr)` (net expectancy per trade in R, costs already
inside the metrics) and `conf = min(1, √trades/10)`. No volume term.

| | quality pool (22 269) | ACTUALLY selected (34) | proposed top-34 |
|---|---|---|---|
| win rate (med) | 0.559 | 0.649 | 0.747 |
| payoff (med) | 1.081 | 1.313 | **1.993** |
| E[R]/trade (med) | +0.151 | +0.513 | **+1.187** |
| trades/month (med) | 47.2 | 52.8 | **28.3** |
| in-sample net (med) | 248 k | 450 k | 93 k |

- **Overlap between the actual selection and the proposed top-34: 0 / 34.**
  The two objectives disagree completely on the same pool.
- The proposed set takes HALF the trades at TWICE the payoff and ~5×
  the in-sample net expectancy per trade; it gives up 80 % of the
  in-sample compounded net — that net was priced with free swap and flat
  spread, i.e. it is exactly the number the cost model inflates.
- Best pool candidate by per-trade edge (`gene_115601_1148`: payoff 3.62,
  WR 0.653, 323 trades) was **not selected** by the current pipeline.
- Break-even WR: selected set 43.3 % (live measured 44.3 % — margin 1 pt);
  proposed set 33.4 % (margin 41 pts at the same in-sample WR).

The existing-but-off payoff gate, applied to the same pool (analyzer-side
metrics — the ones `TargetProfile.accepts` actually reads):

| `prop_search_min_payoff_ratio` | pool survivors | of the actual 34 |
|---|---|---|
| 1.2 | 2 193 (9.8 %) | 30 survive |
| 1.3 | 977 (4.4 %) | 19 survive |
| 1.5 | 291 (1.3 %) | 1 survives |
| 2.0 | 37 (0.2 %) | 0 survive |

## 6. Decisions for the operator (nothing landed silently)

**A. Turn on the payoff floor that already exists.** Set
`models.prop_search_min_payoff_ratio: 1.3` (knob: core config.rs:589,
gate: discovery.rs:1593). Consequence, measured: cuts the July-15 pool to
977 candidates and would have removed 15 of the 34 exported genes,
forcing their slots to higher-payoff candidates. Zero code change.

**B. Rank survivors by payoff-explicit net expectancy.** Replace the
tie-broken income ordering with `E[R]·conf` (or add it as the primary
sort key before quality_score at discovery.rs:4684). Consequence,
measured: a completely different portfolio (0/34 overlap), payoff
1.31→1.99, trades/month 53→28, break-even WR margin 1 pt→41 pt,
in-sample net −80 % (priced with the broken cost model).

**C. Cap the volume term in `ga_fitness_growth`.** Replace
`growth = g_trade × trades` (named.rs:252) with a saturating count, e.g.
`g_trade × n/(1+n/N_sat)` with `N_sat` set from the horizon cadence
(≈2 trades/day → N_sat ≈ 1 300 on this window), so per-trade edge decides
between candidates past the saturation point. Requires a scoring_version
bump to 6 and a fresh run to measure (the GA's population itself will
change, not just the ranking — this cannot be measured from existing
artifacts).

**D. Make the search charge what the demo broker charges.** Three wiring
fixes, no kernel changes: (1) add swap fields to `EvaluationConfig` and
carry them through `discovery_backtest_settings` (the values are already
in `data/symbol_metadata.json`); (2) build a `SessionSpreadProfile` from
broker/session data instead of leaving it `None`; (3) derive commission
from `commission_type`/`rate_decimal` (×2 if the demo deal list confirms
per-side) instead of the flat $7 override. Consequence: every in-sample
number shrinks toward the live truth; must re-run discovery to measure.
Verification anchor: the operator's measured costs = 152 % of gross.

**E. Keep the activity floors honest.** ≥1 trade/day + ≥15 trades/month
(live config:207-208) currently make low-frequency high-payoff shapes
unfindable regardless of objective. If A/B are taken, these floors work
against them; decide the minimum cadence the product actually needs.

Recommended order: **A + B first** (config + ranking, measurable on the
next run against this baseline), then **D** (honest costs), then **C**
(new fitness landscape, fresh search), with **E** decided alongside A.

---

*Method note: all numbers in §5 computed by
`scripts/measure_selection_objective.py` from the artifacts named above;
the script streams the 4 GB quality JSON line-wise and touches nothing.*
