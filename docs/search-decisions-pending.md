# Search — decisions pending the operator (2026-08-09)

The search-to-100 audit (19 agents, refuted, all findings reproduced against
master + the on-disk artifacts) reached one verdict that outranks every
performance question:

> **The search does not optimise for profit.** Risky fitness maximises
> `g_trade × trades` — per-trade log-growth times an *uncapped* trade count.
> Win rate is rewarded in every score; **payoff ratio is rewarded by no stage**;
> the one payoff-aware gate (`prop_search_min_payoff_ratio`) has been off since
> it was built. The cost model charges zero swap, a flat 2.0-pip spread, and
> ~72% of the broker's real commission. So the pipeline buys trade volume with
> fictional edge and systematically selects the shape with no decay margin.

Measured consequence on the real July pool: the exported genes sit at payoff
1.31 / win-rate 0.649 — breakeven 43.3% — against the live-measured reality of
win-rate 44.3%, payoff 0.803, PF 0.639, costs 152% of gross. One point of margin,
then losses. A payoff-explicit objective on the **same** candidate pool selects a
**0-of-34-overlap** portfolio at payoff 1.99 with **41 points** of margin.

**Until the objective is fixed, a faster or bigger search only finds the wrong
thing faster.** Do not spend card-hours scaling population before A+B land.

---

## What already landed (honesty fixes — no philosophy change)

These make the search *measure what it claims*; they do not change what
"good" means. All merged to master, verified:

- **Slice 2** — every selection stage now runs the gene's own stop regime (was:
  9 of 13 sites used fixed stops while the GA scored adaptive — 17.6× trade-count
  divergence). Selection changes as a consequence; measured, deliberate.
- **Slice 3** — GPU submissions sized from the card; search-more knob **OFF**
  (result-invariant until you turn it on).
- **Slice 4** — the objective audit (this analysis) + measurement script.
- **Slice 5** — the run profile captures every selection-changing knob, with a
  compile-error ratchet (falsifiability floor).
- Earlier: CPCV folds reach the card, the cubecl probe no longer starves
  prototype B of VRAM, kill zones resolve once, the promotion gate reads Settings.

## Pending your go-ahead (each changes what the system selects or does)

| # | Decision | Measured consequence | Cost |
|---|----------|----------------------|------|
| **A** | Set `models.prop_search_min_payoff_ratio = 1.3` (knob exists, always 0.0) | July pool 22,269 → 977 survivors; removes 15 of the 34 exported genes | config only |
| **B** | Rank survivors by payoff-explicit expectancy `E[R]·conf` instead of the volume-tie-broken income score | 0/34 overlap with today's portfolio; payoff 1.31→1.99; breakeven-WR margin 1pt→41pt; in-sample net −80% (that net was priced with *free* swap) | one ranking change |
| **C** | Saturating volume term in `ga_fitness_growth` (scoring v6) — stop rewarding raw trade count | changes the GA's own landscape; only measurable with a fresh search | code + a run |
| **D** | Charge the broker's real costs: swap (−2.445/−0.105 pips/night EURUSD), session spread profile, commission from metadata (×2 if per-side) | every in-sample number shrinks toward live truth (anchor: 152%-of-gross) | code + a run |
| **E** | The `≥1 trade/day` and `≥15 trades/month` activity floors make high-payoff low-cadence shapes unfindable — decide the minimum cadence the product needs | unblocks the shapes A+B would prefer | config |

**Minimum to stop selecting the wrong thing: A + B.** C/D/E deepen it.

### Also awaiting a call

- **Slice 6 (data-integrity gate)** — ready to merge; once it lands, `discover`
  will **refuse** D1/W1/MN1 ladders on the 5 symbols with corrupt 2014–2016 bars
  (AUDUSD/EURGBP/EURJPY/EURUSD/GBPUSD) instead of silently searching half a 2016.
  This is fail-loud correctness, but it changes what your next run does — hence
  flagged. The other 9 symbols are clean.
- **Re-import** — clears the corruption. Zero money, unattended
  `--bootstrap-data` per symbol, hours each, atomic in-place rewrite (never
  `rm -rf`). Side effect: clean f64 quotes, ~7× smaller files. Comparative runs
  (A-vs-B, seed) on identical bars are trustworthy **now**; absolute selection
  from the corrupt ladders is not until this runs.
- **Config unification** — the repo `config.yaml` (risky / pop 200 / 20000 gens
  / 1h) and the store `config.yaml` (prop_firm / pop 100 / 1000 gens / 24h)
  diverge today, so an app-launched run and a CLI run can search different
  things. Which file wins?

## The card plan once A+B are decided

The objective ON/OFF run is the one that finally measures whether a corrected
search selects strategies that survive their own holdout: same seed, same bars,
payoff-floor 1.3 + `E[R]·conf` ranking vs today's objective, on a clean symbol
(USDJPY/USDCHF now; EURUSD after re-import), comparing tail-holdout net/PF/DD of
the two portfolios (~8–24h for the pair).
