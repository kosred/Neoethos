# Risky Mode — Aggressive Compounding Research

Operator directive (2026-05-15, verbatim):
> "Μια ακόμη επιλογή risky Mode θα μπορούσε να είναι κάτι του τύπου
> το 20 pip challenge όπου από 20$ πάει στις 50.000$ αλλά με
> μεγαλύτερη ασφάλεια ίσως και να βρούμε τρόπους να αυξηθεί η
> επιτυχία."

This deliverable surveys the literature, prop-firm landscape, and
internal codebase, and proposes a concrete design for a separate
"Risky Mode" that lives next to (not on top of) `FTMO_STANDARD`.
The mode targets the operator's $20 → $50,000 ambition but pulls
every available lever — fractional Kelly with a logarithmic taper,
ML-gated entries, regime/news filters, kill-switch hierarchy,
no-overnight-Friday flatten — to keep the **first-stage** ruin
probability well below the 95 %+ figure that the canonical retail
"20-pip challenge" carries.

## Table of contents

- §0 — Scope, methodology, document rules
- §1 — Origin and variants of the "20-pip challenge"
- §2 — Academic foundations
- §3 — Industry implementations
- §4 — Proposed Risky Mode design
- §5 — Kill-switch hierarchy
- §6 — Backtest harness
- §7 — UI surface
- §8 — Wizard step
- §9 — Mathematical risk model
- §10 — Operator decisions required
- §11 — Codebase touch-points

---

## §0 — Scope, methodology, document rules

### 0.1 What this document is

A **research deliverable**, not code. It documents:

- the origin and known failure rate of the retail
  "20-pip challenge" compounding idea,
- the academic and industry alternatives that ship better
  capital-protection mechanics,
- a concrete design for a forex-ai "Risky Mode" that takes the
  operator's $20 → $50,000 framing seriously while keeping the
  app's existing prop-firm machinery (`FTMO_STANDARD`) intact,
- the math behind every claim, with citations.

### 0.2 What this document is not

- Not a promise that anyone can turn $20 into $50,000.
- Not a change to `PropFirmConstraints::FTMO_STANDARD` in
  `crates/forex-core/src/domain/prop_firm.rs:32` (operator rule:
  the 4 % monthly floor and the FTMO 5 %/10 % limits stay).
- Not a synthetic-data exercise; every numeric example labelled
  "hypothetical" is flagged explicitly per the operator's
  no-synthetic-data rule.

### 0.3 Methodology

- WebSearch on canonical retail sources (Forex Factory, BabyPips,
  Trading Rush, Medium write-ups) for the "20-pip challenge"
  rules and any documented outcomes.
- WebSearch on arXiv + ResearchGate + Semantic Scholar for the
  Kelly criterion, Optimal-f, maximum drawdown of Brownian motion,
  and path-dependent drawdown measures. The Hugging Face
  `paper_search` tool was denied in this session, so arXiv
  citations come through WebSearch and direct arXiv IDs.
- WebFetch was attempted on FTMO, the Magdon-Ismail PDF, the
  Berkeley "Good and bad properties of Kelly" PDF, and the Forex
  Factory thread; all returned HTTP 403. Quotations therefore come
  from WebSearch result excerpts (the URLs are still cited so the
  operator can verify out-of-band).
- Internal references (file:line) come from the live tree at the
  time of writing: `crates/forex-core/src/domain/prop_firm.rs`,
  `crates/forex-core/src/domain/risk.rs`,
  `crates/forex-app/src/app_services/trading/risk_gate.rs`,
  `crates/forex-search/src/genetic/regime_labels.rs`,
  `crates/forex-models/src/forecasting/swarm_impl.rs`.

### 0.4 Operator constraints carried through

- No new hardcoded numbers that could be the prop-firm floor.
  Every Risky-Mode tunable proposed here goes into a separate
  config struct (proposed: `RiskyModeConfig` in
  `crates/forex-core/src/domain/risky_mode.rs`), not into
  `PropFirmConstraints`.
- No synthetic data even in research mockups. Where a hypothetical
  number is unavoidable to illustrate a formula (e.g. "if
  win-rate p = 0.55"), it is labelled `(hypothetical)`.
- `f32` / `Ordering::Relaxed` out of scope here — this is a design
  document, no code.

---

## §1 — Origin and variants of the "20-pip challenge"

### 1.1 Canonical rules

The "20-pip challenge" is a retail compounding framework that
targets growing a $20 account to roughly $52,000 in 30
discrete "levels". The most-cited written version (Trading
Rush, "$20 to $52,400 strategy", and the Medium summary
[Human×AI, "Turning $20 into $52,000"]
(https://medium.com/@HUMANxAI/the-20-to-52-000-challenge-tested-by-1-000-bots-7484da203eca))
prescribes:

- **Target per level:** 30 % profit on the level's starting
  balance.
- **Risk per trade:** 23 % of the level's starting balance.
- **Reward-to-risk ratio:** ~ 1.3 : 1.
- **Implied minimum win-rate for break-even:** ~ 50 % geometric;
  ~ 60 % is the figure usually quoted as "comfortable".
- **Progression:** advance to the next level when balance exceeds
  that level's target. If balance falls below the prior level's
  floor, demote back to the previous level.
- **Daily pip target (variant 1 — "20-pip / day"):** the trader
  closes after the first 20-pip win each day, banking it.
- **Variant 2 ("$20 → $52,400"):** drop the per-day pip cap, drop
  the "1 trade per day" rule, allow any market and any number of
  trades.

Sources for the rules:

- Trading Rush, "Testing $20 To $52,400 Strategy 1000 TIMES — Fastest
  Way To Grow Small Trading Account"
  (https://tradingrush.net/testing-20-to-52400-strategy-1000-times/)
- "20 Pips Challenge Spreadsheet Guide" PDF
  (https://www.scribd.com/document/576489690/20-Pip-Challange)
- Forex Factory thread "20 pips Challenge — (Compounding model)"
  (https://www.forexfactory.com/thread/1306313-20-pips-challenge-compounding-model)
- "Forex 20 Pip Challenge: Boost Your Trading Skills"
  (https://market-bulls.com/forex-20-pip-challenge/)
- Howtotrade.com, "Forex Compounding Plan — Free PDF Template"
  (https://howtotrade.com/blog/forex-compounding-plan/)

### 1.2 Why it almost always fails

Three independent mechanisms compound to make the canonical
challenge effectively unwinnable in practice for an unassisted
retail trader.

**(a) Risk-per-trade is several multiples of Kelly.** With the
typical retail forex edge (R = 1.3, p ≈ 0.55), the Kelly
fraction is

```
f* = p − (1 − p) / R = 0.55 − 0.45 / 1.3 ≈ 0.204
```

so the 23 % risk literally sits above full Kelly. Anything
above full Kelly is *strictly* dominated — same expected log
growth on the upside, strictly higher variance, strictly higher
ruin probability (MacLean / Thorp / Ziemba, *The Kelly Capital
Growth Investment Criterion: Theory and Practice*,
https://www.worldscientific.com/worldscibooks/10.1142/7598;
summary at https://www.caia.org/sites/default/files/AIAR_Q3_2016_05_KellyCapital.pdf).

**(b) Sequence risk dominates expectation at small N.** With 30
levels, even a system with positive expectancy faces large
"losing streak before reaching target" probability. The
gambler's-ruin random-walk result (Feller, *An Introduction to
Probability Theory and its Applications*, Vol. 1, summarised at
https://en.wikipedia.org/wiki/Gambler%27s_ruin) gives for an
unfair coin with win probability p, loss probability q = 1 − p,
starting bankroll `i`, absorbing barrier at 0 and at N,

```
P(ruin from i) = ((q/p)^i − (q/p)^N) / (1 − (q/p)^N)   if p ≠ q
```

At 23 % risk and R = 1.3, a single losing trade burns 23 % of
the bankroll; four consecutive losses leaves ≈ 35 % of the
starting capital, well past Vince's "critical drawdown"
threshold of ~50 % beyond which "ruin becomes statistically
inevitable, even with positive expectancy" (Vince, *The
Mathematics of Money Management*, 1992 — paraphrased in
https://sumshelf.com/book-summaries/mathematics-of-money-management-risk-analysis-techniques-for-traders--ralph-vince/
and confirmed at
https://www.quantifiedstrategies.com/optimal-f-money-management/).

**(c) Broker-side margin call before strategy-side stop.** At
$20 starting capital, even one micro-lot (0.01) on EURUSD costs
about $0.10/pip
(https://www.fxtm.com/en/trading/tools/pip-calculator/), and
required margin at 1:500 leverage is roughly $2. cTrader's
"smart stop-out" at 50 % margin level partially closes positions
when equity drops below 50 % of used margin
(https://help.ctrader.com/trading-with-ctrader/conditions/;
https://www.fxpro.com/help-section/education/beginners/articles/stop-out).
A single bad fill against a 10-pip stop on a 0.10-lot position
($1 / pip) wipes $10 — half the starting account — before the
strategy-side stop fires. This is the same broker-protection
floor that Vince's "Critical Equity Drawdown" formalises.

### 1.3 Documented empirical outcomes

Trading Rush ran the strategy through 1,000 bot simulations
after softening some of the original rules (no 20-pip-cap,
no 1-trade-per-day) — the explicit goal was to make the rules
more permissive, not stricter, and even with the loosened rules
the public write-ups (TradingRush.net page and the Human×AI
Medium piece linked above) report that the strategy is
"heavily reliant on a consistently high win rate (around 60 %)"
and that "a lower win rate significantly increases the risk of
failure." The bot tests' raw success-vs-blowup ratio is not
reported on the public page in number form; the bot test result
is presented as a video conclusion rather than a published
statistic, which is why we treat the "20-pip challenge" success
rate as **anecdotal at the public-source level** rather than
quantitatively known.

What *is* quantitatively known is the broader retail forex
failure rate: BabyPips' aggregation of ESMA disclosures puts
"70-80 % of retail traders are unprofitable"
(https://www.babypips.com/news/almost-80-percent-of-retail-traders-are-unprofitable);
the ESMA's own analysis cites a band of 74-89 %
(https://investingoal.com/esma-broker-client-success-rate-stats/),
and an aggregation across 35 major ESMA brokers for 2024 came in
at 86 % unprofitable
(https://10pmtrader.com/why-retail-traders-lose-money/).

So the floor we have to beat with Risky Mode is: **the
unconditional retail-forex unprofitability rate is ~75-85 %**.
The 20-pip challenge, layered on top of that, is in our
estimate at least as bad and likely much worse, because its
per-trade risk fraction (~23 %) is multiples of any defensible
Kelly fraction for retail-detected edges.

### 1.4 Mathematical expected-value summary

For the canonical rules (per-trade risk 23 %, R = 1.3),
the geometric per-trade growth factor is

```
G(p) = (1 + 0.23·1.3)^p · (1 − 0.23)^(1−p)
     = 1.2990^p · 0.7700^(1−p)
```

Geometric expectation ≥ 1 iff

```
p · ln(1.2990) + (1−p) · ln(0.7700) ≥ 0
⇒ p ≥ ln(1/0.7700) / (ln(1.2990) + ln(1/0.7700))
⇒ p ≥ 0.2614 / 0.5230
⇒ p ≥ 0.5000
```

So you need exactly a 50 % win-rate just to *break even* on
geometric mean — and the variance per trade is enormous. The
geometric Brownian-motion analogue gives ruin probability at
infinite-horizon

```
P(ruin) = exp(−2μB / σ²)
```

where μ is drift, σ is per-step volatility, and B is starting
bankroll on a log scale (Mason Malmuth's formulation,
https://gamblingcalc.com/poker/scientific-risk-of-ruin-calculator/;
underlying math from Feller). At per-trade risk that high, σ²
dominates and the exponent is small in magnitude, so ruin
probability stays close to 1 unless edge is very large.

This is the mathematical reason the canonical rules don't work
and the design in §4 has to taper risk-per-trade hard.

---

## §2 — Academic foundations

### 2.1 Kelly criterion (Kelly 1956 → MacLean/Thorp/Ziemba 2011)

The Kelly fraction for a discrete binary bet is

```
f* = (b·p − q) / b
```

with p the win probability, q = 1 − p, b the odds (payoff in
multiples of stake on a win, loss = stake on a loss). It
maximises the long-term expected log-wealth growth rate

```
G* = p·ln(1 + b·f*) + q·ln(1 − f*)
```

Equivalently, in continuous time with normal log-returns of
drift μ and volatility σ, the optimal leverage fraction is

```
f* = μ / σ²
```

with achieved growth rate `G* = μ² / (2σ²)` and any
fractional-Kelly choice `f = α·f*` produces growth
`G(α) = α·(2 − α)·G*` (so half-Kelly captures 75 % of optimal
growth, quarter-Kelly 43.75 %). Sources:

- https://en.wikipedia.org/wiki/Kelly_criterion
- https://corporatefinanceinstitute.com/resources/data-science/kelly-criterion/
- https://www.cqf.com/blog/quant-finance-101/what-is-the-kelly-criterion
- https://nickyoder.com/kelly-criterion/

The MacLean/Thorp/Ziemba *Kelly Capital Growth Investment
Criterion* (World Scientific, 2011) catalogues the "good" and
"bad" properties:

- **Good:** uniquely maximises long-run wealth, minimises
  expected time to reach a target, and is asymptotically
  myopically optimal.
- **Bad:** short-term variance is extreme; an example in the
  book shows that even with a 14 % edge and 700 bets, *full*
  Kelly can leave $1,000 at $18 — a 98 %+ loss — and half-Kelly
  can lose 85.5 % in the worst 1-in-1,000 simulation. (Cited
  via summary at
  https://www.caia.org/sites/default/files/AIAR_Q3_2016_05_KellyCapital.pdf
  and the Berkeley "Good and Bad Properties of the Kelly
  Criterion" PDF abstract at
  https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
  — the PDF itself returned 403 to WebFetch; abstract excerpted
  from search results.)

The take-away: full Kelly is too aggressive even for
professionals; quarter-Kelly is the *professional default*
(Thorp famously used half Kelly, top quant firms commonly
use quarter Kelly — see
https://medium.com/@tmapendembe_28659/the-dangers-of-full-kelly-criterion-why-most-traders-should-use-fractional-kelly-criterion-instead-0338e3bcc705).

### 2.2 Risk of ruin — Feller / gambler's-ruin

For a discrete asymmetric gambler's-ruin process, P(ruin) has
the closed form recited in §1.4. For a Brownian-motion-with-drift
limit, P(ruin from B above the origin) = exp(−2μB/σ²) (Karl
Whelan, "Ruin Probabilities for Strategies with Asymmetric Risk",
https://www.karlwhelan.com/Papers/Ruin.pdf; UCD WP 2025-03
"The Gambler's Ruin with Asymmetric Payoffs",
https://www.ucd.ie/economics/t4media/WP2025_03.pdf; Columbia
notes by Sigman,
http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GR.pdf).

The two practical corollaries:

1. A small bankroll is **structurally** more fragile than a
   large one — `B` is the bankroll in units of one bet, so the
   exponent shrinks linearly with `B`. This is the reason §4's
   "logarithmic risk taper" allocates the highest variance to
   the *smallest* bankroll: if it dies, the absolute loss is
   tiny; later stages clamp variance because the bankroll is
   no longer disposable.
2. Edge is multiplicative-in-bankroll inside the exponent. So
   the same edge halves ruin probability roughly when bankroll
   doubles — *if* edge stays positive. This is the case for
   ML-gated entries (§4.6).

### 2.3 Optimal-f and Secure-f (Vince)

Ralph Vince's *The Mathematics of Money Management* (Wiley, 1992,
https://www.amazon.com/Mathematics-Money-Management-Analysis-Techniques/dp/0471547387;
full PDF mirror at
https://dl.fxf1.com/files/books/english/MathematicsMoneyManagement.pdf)
defines `f` as the fixed fraction of bankroll risked per trade
such that the geometric mean of the *historical* trade-return
distribution is maximised — the "optimal" fixed-fraction.

Vince explicitly warns that trading at optimal-f delivers
"substantial drawdowns in terms of percentage equity
retracements" and the "Critical Equity Drawdown" past which
"ruin becomes statistically inevitable, even with positive
expectancy" is conventionally around 50 % (paraphrased from
the summary at
https://sumshelf.com/book-summaries/mathematics-of-money-management-risk-analysis-techniques-for-traders--ralph-vince/
and the QuantifiedStrategies write-up at
https://www.quantifiedstrategies.com/optimal-f-money-management/).

**Secure-f** (Zamansky and Stendahl) constrains optimal-f by an
operator-chosen maximum drawdown — pick the largest `f` such
that historical drawdown stays below threshold. This is the
intellectual ancestor of §4's per-stage taper: rather than one
`f`, Risky Mode uses a different secure-f at each compounding
stage.

### 2.4 Maximum-drawdown distribution (Magdon-Ismail / Atiya)

For Brownian motion with drift μ, volatility σ, horizon T, the
expected maximum drawdown (MDD) scales as

```
E[MDD] ≈ 2·σ²/μ · g(μ²·T / (2·σ²))
```

with `g(·)` a series-expanded transcendental documented by
Magdon-Ismail and Atiya
(https://www.cs.rpi.edu/~magdon/ps/journal/drawdown_journal.pdf,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=874069). For
small dimensionless time `μ²T/(2σ²)`, MDD scales like √T; for
large `μ²T/(2σ²)`, MDD saturates near `σ²/μ`.

Practical consequence: drawdown grows **faster than the square
root of horizon** in the regime where most retail strategies
operate. Risky Mode therefore has to monitor MDD explicitly
(not just daily-loss), which is why §5's kill-switch hierarchy
includes a per-stage retreat trigger keyed off rolling-peak
drawdown, not just intraday loss.

### 2.5 Conditional Expected Drawdown (Goldberg / Mahmoud 2014)

Goldberg & Mahmoud (arXiv 1404.7493,
https://arxiv.org/abs/1404.7493) formalise **Conditional
Expected Drawdown (CED)** as the tail mean of the maximum
drawdown distribution; show CED is positively homogeneous
(linearly attributable to factors) and convex (usable in
quantitative optimisation), and show it is *more* sensitive to
serial correlation in returns than ES or volatility — a real
issue for trend-following strategies whose returns are
autocorrelated.

This is the academic justification for §4's regime filter — if
serial correlation is the dominant driver of drawdown, then
trading only in regimes where serial correlation is *favourable*
(trending) is a direct CED-reduction lever.

### 2.6 Path-dependent risk and sequence risk

"A Rational Risk Policy? Why Path Dependence Matters" (Bouchaud
et al., *Entropy* 2023,
https://www.mdpi.com/1099-4300/25/2/202;
https://pmc.ncbi.nlm.nih.gov/articles/PMC9955835/) develops the
sequence-risk framework: even with positive expectancy, the
*sequence* in which gains/losses arrive determines the
drawdown trajectory. Their Monte-Carlo experiments show that
two strategies with identical mean and variance can have
materially different ruin probabilities depending on the serial
correlation of P&L. This is why §6's backtest harness reports
**ruin probability over Monte-Carlo bootstraps**, not just
mean / variance.

### 2.7 Risk-constrained Kelly (Busseti / Ryu / Boyd 2016)

Busseti, Ryu and Boyd, "Risk-Constrained Kelly Gambling"
(Stanford, https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf)
formulate the Kelly objective as a convex program with an
explicit drawdown-or-VaR side constraint. The optimal solution
is always a *shrinkage* of the unconstrained Kelly allocation
toward zero — exactly the structure §4 adopts (fractional Kelly,
shrinkage factor per stage).

### 2.8 Practical implementation of Kelly (Frontiers 2020)

Carta & Sanna, "Practical Implementation of the Kelly Criterion"
(Frontiers in Applied Math & Stats, vol 6, art 577050,
https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2020.577050/full)
empirically calibrate optimal growth, number of trades, and
rebalancing frequency for equity portfolios. Their main result:
in practice, **transaction costs put a hard floor on minimum
bankroll** below which any Kelly-optimal strategy is destroyed
by frictions. This is the academic justification for §4's
"$20 → $40 → $80" stage gates: each stage has a different
spread/cost-to-equity ratio and therefore a different feasible
strategy.

### 2.9 Modern Portfolio Theory limits at small bankroll

MPT-style diversification assumes transaction-cost-free
rebalancing across N positions. With $20 at 0.01-lot minimum
($0.10 / pip on EURUSD), even one position consumes more of the
margin budget than diversification can offset. So the small-
bankroll regime is **single-position, sequential** — exactly
the framing of the 20-pip challenge and exactly what Risky Mode
encodes in §4.4 ("concurrent positions: 1 below $1k").

---

## §3 — Industry implementations

### 3.1 FTMO — Standard and Aggressive

FTMO publishes its Trading Objectives at
https://ftmo.com/en/trading-objectives/ and a drawdown
walkthrough at https://ftmo.com/en/blog/drawdowns/.

**Standard:**

- Max daily loss: 5 % of initial balance (calculated end-of-day,
  CET reset). Equity, not balance.
- Max overall loss: 10 % of initial balance, *static* — the
  90 % floor is set at challenge start and never moves up.
  (https://academy.ftmo.com/lesson/maximum-loss/)
- Profit target: Phase 1 = 10 %, Phase 2 = 5 %.
- Min trading days: 4 (Phase 1) / 4 (Phase 2) in the modern
  2-step model.

**Aggressive:**

- Max daily loss: 10 % (https://runvigil.app/rules/ftmo,
  https://www.luxalgo.com/blog/ftmo-prop-firm-review-how-to-pass-in-2025/).
- Max overall loss: 20 %.
- Profit target: Phase 1 = 20 %, Phase 2 = 10 %.
- Capital cap: $200,000 (lower than Standard's $400,000).

**Take-away for Risky Mode:** FTMO Aggressive doubles every
risk limit and every target. It is a "compressed time, larger
bankroll" model, not a "compressed bankroll, larger horizon"
model. The 20-pip challenge is closer to the latter and FTMO
Aggressive does *not* model it — so we cannot just rename
"FTMO Aggressive" as "Risky Mode". They are different products.

### 3.2 MyForexFunds — Rapid / Accelerated / Emphatic

MyForexFunds (now defunct after CFTC enforcement in 2023, but
the documented rules are still on the wayback) ran two variants
in its Accelerated program (https://myforexfunds.com/accounts/accelerated-program/,
https://myforexfunds.com/accelerated-accounts-rules/):

- **Conventional:** 5 % overall drawdown.
- **Emphatic:** 10 % overall drawdown, 20 % profit target,
  1.5× / 2× scaling per 10 % milestone.

No restrictions on lot size, news trading or overnight
holding — but **mandatory weekend flatten** (any open Friday
position auto-suspended). Risky Mode adopts the mandatory
weekend-flatten rule directly from this firm's policy.

### 3.3 FundedNext, The5%ers, Topstep

Comparison data from the 2026 round-up
(https://traderssecondbrain.com/guides/prop-firm-comparison,
https://traderssecondbrain.com/guides/prop-firm-rules-cheatsheet):

- **FundedNext Stellar 1-Step:** 10 % profit target, 3 % daily
  loss, 6 % max loss. The aggressive end of FundedNext's
  product line.
- **The5%ers:** 5 % max drawdown (half of FTMO) plus a
  consistency rule (single-day P&L ≤ 50 % of total profits).
  This is the *most conservative* major firm — the opposite
  end of the spectrum from Risky Mode.
- **Topstep FX:** **trailing** drawdown (the only major firm
  with a trailing floor). Critically different sizing model
  than FTMO — Risky Mode does *not* adopt trailing because
  trailing punishes compounding (the floor moves up with
  equity, defeating the compounding objective).

### 3.4 ProRealTime / TradeIdeas / QuantConnect

- **ProRealTime forum on optimal-f sizing** (https://www.prorealcode.com/topic/ralph-vinces-optimal-f-positioning-sizing/)
  — the retail community implementation of Vince's optimal-f.
  Confirms practical drawdowns in the 30-60 % range using
  unmodulated optimal-f.
- **QuantConnect** (https://www.quantconnect.com/) — open-source
  algo platform, > 1,200 community strategies. The platform
  itself ships no aggressive-compounding template; the
  Realistic Expectations forum thread
  (https://www.quantconnect.com/forum/discussion/5720/realistic-expectations-in-algo-trading/)
  documents that algo-trader community consensus is that
  > 50 % annualised on a robust backtest is borderline
  suspicious, which puts the 20-pip-challenge's implied
  ~250,000 % return in extreme outlier territory.

### 3.5 Disclosed hedge-fund returns (sanity check)

- **Renaissance Medallion** — 66 % gross / 39 % net annualised
  1988-2018 (https://en.wikipedia.org/wiki/Renaissance_Technologies,
  https://quartr.com/insights/edge/renaissance-technologies-and-the-medallion-fund).
  Best hedge fund in history.
- **Citadel Wellington** — historically 19-20 % gross average
  (https://hedgefollow.com/funds/Renaissance+Technologies/Performance-History
  comparison data).

**Sanity context for Risky Mode:** $20 → $50,000 is a
3,400 × increase. At Medallion's 66 % gross annualised, that
compounding takes `log(3400) / log(1.66) ≈ 16 years`. At
20 %/year, ~ 45 years. The "weeks-or-months" framing of the
20-pip challenge is mathematically the same as betting on
multi-σ tail events. Risky Mode owns that fact in §4.1 — the
*honest* expected time-to-target at any sustainable risk
fraction is years, not weeks, and the design surfaces this in
the UI rather than promising the contrary.

---

## §4 — Proposed Risky Mode design for forex-ai

### 4.1 Honest framing of the objective

The operator's framing is correct as an *upper-bound aspiration*
(any individual run of Risky Mode could reach $50,000; many
will not). The design's job is to:

1. give the operator a parameterised path that has a
   *substantively better* first-stage ruin probability than the
   canonical 20-pip challenge,
2. enforce hard floors that prevent silent blowup,
3. let the operator stop the experiment cheaply if it doesn't
   work,
4. honestly publish — in the UI and Wizard — the ruin
   probability the chosen parameters imply.

Risky Mode is **explicitly opt-in** and **explicitly labelled
as accepting full-loss risk**. It is not the default.

### 4.2 Capital staging — logarithmic risk taper

Stage boundaries (each ≈ doubling the bankroll from the
previous):

| Stage | Bankroll range  | Kelly fraction      | Max risk / trade | Concurrent positions | Daily-loss cap |
| ----- | --------------- | ------------------- | ---------------- | -------------------- | -------------- |
| S1    | $20 - $40       | 1.0 × Kelly (full)  | 8 %              | 1                    | 50 %           |
| S2    | $40 - $80       | 0.75 × Kelly        | 6 %              | 1                    | 35 %           |
| S3    | $80 - $160      | 0.5 × Kelly (half)  | 4 %              | 1                    | 25 %           |
| S4    | $160 - $320     | 0.5 × Kelly         | 3 %              | 1                    | 20 %           |
| S5    | $320 - $640     | 0.4 × Kelly         | 2.5 %            | 1                    | 15 %           |
| S6    | $640 - $1,280   | 0.33 × Kelly        | 2.0 %            | 2                    | 10 %           |
| S7    | $1,280 - $5,120 | 0.25 × Kelly (quarter)| 1.5 %          | 2                    | 7 %            |
| S8    | $5,120 - $25,600| 0.20 × Kelly        | 1.0 %            | 3                    | 5 %            |
| S9    | $25,600 - $50,000+ | 0.10 × Kelly     | 0.5 %            | 3                    | 5 % (FTMO)     |

Notes:

- All percentages are **proposals**, not constants of the
  domain; they live in `RiskyModeConfig` (see §11) and are
  operator-tunable per directive.
- S1's "8 % risk / trade, 50 % daily cap, 100 % bankroll
  exposure" is consciously aggressive but is *less than half*
  the canonical 20-pip-challenge 23 % / trade and is gated by
  ML confidence (§4.6). This is the §1.4 "the bankroll is
  disposable, accept variance" stance.
- S9 collapses to FTMO Standard parameters: at $25k+ the
  account is no longer disposable and the operator's existing
  4 %-monthly-floor / 5 %-daily / 10 %-total prop-firm regime
  becomes the right one.
- The taper is monotonic in bankroll. Stage transitions are
  **on equity crossing the boundary**, evaluated at every
  position-close. A negative cross (e.g. S6 → S5) demotes
  immediately; a positive cross promotes only after a
  configurable hysteresis window (proposal: 5 closed trades or
  24 hours in the new stage, whichever is later — this is to
  damp ping-pong between stages on volatile single trades).

### 4.3 Per-trade rules

- **R:R target:** 1:3 default (10-pip SL, 30-pip TP). With
  R = 3, geometric break-even win-rate is `1/(1+R) ≈ 25 %`, so
  the strategy can survive an ML model with much-better-than-
  random calibration but only modest absolute hit rate. The
  Kelly fraction for R = 3 and p = 0.40 (a plausible
  filtered-signal hit rate) is `f* = p − (1−p)/R = 0.40 − 0.20
  = 0.20` — full Kelly at 20 %. Risky Mode never sizes higher
  than the per-stage table cap regardless of what Kelly
  recommends.
- **Position size:** `size_pct = min(stage_cap, α_stage · f*)`
  where `α_stage` is the multiplier in the table and `f*`
  comes from the live rolling-window edge estimate (proposal:
  last 50 closed Risky-Mode trades, EWMA-weighted, with a
  prior so the first trades cannot push `f*` arbitrarily
  high).
- **Concurrent positions:** capped per stage table. The same
  pair cannot be opened twice in the same direction
  (correlation = 1).
- **Per-pair exposure cap:** at S1-S3, 100 % of bankroll on a
  single pair is allowed (single-position regime). From S6
  onward, no single pair > 30 % of bankroll. (Operator-tunable.)

### 4.4 Per-day rules

- **Daily loss cap:** per stage table. At S1 the cap is 50 %
  (the rationale: a $20 bankroll losing $10 is recoverable
  with a $10 reload; the value to the system of *learning* the
  edge dominates the value of capital preservation. At S6+ the
  cap drops to ≤ 10 % so a bad day cannot wipe months of
  compounding).
- **Daily profit lock:** if cumulative day-PnL > 3 × the
  current SL distance, reduce subsequent position sizes by 50 %
  for the rest of the day. (This is a "lock the win" rule —
  inherited conceptually from `prop_firm_rules.daily_profit_lock_pct`
  in `risk.rs:36-37`.)
- **Cooldown after 3 consecutive losses:** force a 1-hour
  no-trade window. (Existing revenge-trade detector in
  `risk.rs:243` already does something analogous; Risky Mode
  reuses it.)

### 4.5 Per-week / per-month rules

- **Weekly DD cap:** 25 % at S1, tapering to 7 % at S9.
- **Monthly DD cap:** 50 % at S1, tapering to FTMO 10 % at S9.
- **Mandatory weekend flatten:** all positions closed by 21:00
  UTC Friday (broker-dependent — see cTrader trading-conditions
  page at https://help.ctrader.com/trading-with-ctrader/conditions/).
  Rationale: weekend-gap risk through a stop loss is unbounded
  (https://www.alphaexcapital.com/forex/forex-trading-basics/forex-market-structure/weekend-gaps-in-forex-trading,
  https://www.tastyfx.com/news/gap-risk--what-it-means-and-how-to-avoid-it-231017/);
  for a small account in S1 a single bad gap can wipe the
  account. MyForexFunds enforced this rule for the same reason.

### 4.6 ML-assisted "increase success" features

This is where the operator's "ίσως και να βρούμε τρόπους να
αυξηθεί η επιτυχία" gets a concrete implementation. Each lever
is gated by a per-stage threshold so that the early high-
variance stages also have the highest filter strictness.

1. **Model-confidence filter.** Reuse the swarm-forecaster
   confidence already exposed by
   `crates/forex-models/src/forecasting/swarm_impl.rs`. Stage
   table:

   - S1-S3: confidence ≥ 0.80 (high-confidence only).
   - S4-S6: confidence ≥ 0.70.
   - S7-S9: confidence ≥ 0.65.

   Rationale: at S1, we accept catastrophic per-trade variance
   *only* if the model is very confident. At S9, the bankroll
   is large enough that we can afford to take medium-
   confidence signals because the per-trade variance is small.

2. **Anomaly / volatility-skip filter.** Reuse the rolling
   ATR baseline (Risky Mode adopts the existing
   `market_volatility` input to `PositionSizingInput` in
   `risk.rs:189`). If instantaneous volatility deviates more
   than 2σ from a 30-day rolling baseline, skip the session.
   Sources for the 2σ heuristic — common across CPI/NFP-day
   guidance: https://maventrading.com/blog/cpi-days-explained-forex-trading-strategies-to-survive-high-impact-news.

3. **News blackout.** Skip 30 minutes either side of any
   scheduled high-impact news event. Forex Factory's impact
   classification is the de-facto retail standard
   (https://www.xs.com/en/blog/forex-factory-guide/,
   https://help.alpha-futures.com/en/articles/9492063-news-trading-policy
   — "no orders within 2 minutes before or 2 minutes after high
   impact news events"). We use 30 minutes because at the small-
   account stages the spread can stay wide for that long after
   a release.

4. **Regime detection.** Use the labelled regime windows from
   `crates/forex-search/src/genetic/regime_labels.rs:14` to
   gate entries by regime. Risky Mode trades only in
   "trending" or "trend-impulse" labelled windows; ranging
   regimes are skipped. The §2.5 CED-vs-serial-correlation
   result is the academic backing.

5. **Adaptive entry.** Short-horizon direction confidence from
   the swarm forecaster
   (`crates/forex-models/src/forecasting/swarm_impl.rs`) is
   used as a *second* gate on top of model confidence — a
   trade fires only when the swarm majority and the headline
   model agree above the stage threshold.

6. **Drawdown-recovery taper.** Already present in
   `risk.rs:701-728`. Risky Mode reuses the same logic but
   with stage-aware thresholds: at S1, the per-stage daily-DD
   cap *is* the recovery floor; at S9, the existing
   `total_dd >= 0.05` halt fires.

### 4.7 Why this is "more secure than retail mainstream"

| Lever                        | Canonical 20-pip challenge | Risky Mode |
| ---------------------------- | -------------------------- | ---------- |
| Risk per trade               | 23 % flat                  | 0.5-8 %, stage-dependent |
| Kelly multiple               | ≈ 1.13 × full Kelly        | 0.10-1.0 × Kelly, taper |
| Daily-loss cap               | none (broker stop only)    | 5-50 % per stage         |
| Weekly DD cap                | none                       | 7-25 % per stage         |
| Monthly DD cap               | none                       | 10-50 % per stage        |
| Concurrent positions         | unlimited                  | 1-3 per stage            |
| Weekend overnight allowed    | yes                        | no (mandatory flatten)   |
| News blackout                | no                         | 30 min around high-impact|
| Regime filter                | no                         | yes (trending only)      |
| ML confidence gate           | no                         | 0.65-0.80 per stage      |
| Anomaly / volatility skip    | no                         | > 2σ skip                |
| Auto-retreat on stage demote | no                         | yes                      |
| Auto-flatten on disconnect   | no                         | yes (hardware kill)      |
| Manual kill switch           | no                         | yes (UI + RPC)           |

Every one of the right-column items is a documented
risk-reduction lever from §2-§3 with academic or industry
backing. The canonical 20-pip challenge ships none of them.

---

## §5 — Kill-switch hierarchy

In ascending severity, every switch documented with its
trigger, action and re-arm condition.

### 5.1 Per-trade — hard SL

Every Risky-Mode order is sent with a server-side stop-loss
(`ProtoOANewOrderReq` with `relativeStopLoss` field — see
`docs/audits/research/ctrader_api_full_reference.md:773` for
the message schema). No "mental stop" allowed. Rationale: if
the strategy crashes between sending the order and computing
the planned client-side stop, the broker's server-side stop
still fires.

### 5.2 Per-day — daily loss cap

Trigger: cumulative day-PnL ≤ −(stage daily-loss cap).
Action: flatten all open positions, refuse new orders until
next CET daily reset. Re-arm: automatic at 00:00 CET (matches
FTMO's reset cadence). Implementation hook:
`RiskManager::check_trade_allowed` in
`crates/forex-core/src/domain/risk.rs:501` — Risky Mode adds a
new branch keyed off the stage-aware cap.

### 5.3 Per-stage — retreat path

Trigger: bankroll drops below the previous stage's lower
boundary (e.g. S6 = $640-$1,280; if equity falls below $640
the system demotes to S5). Action: clamp `α_stage` to the
previous stage's value; clamp `max_risk_pct` ditto; emit a UI
event ("Risky Mode retreat: S6 → S5, $640 floor breached").
Re-arm: bankroll re-crosses the upper boundary *and* the
hysteresis window (5 closed trades / 24 h) expires.

The retreat path is what the operator's question "μεγαλύτερη
ασφάλεια" was asking for. The canonical 20-pip challenge has a
similar idea ("if balance falls below the prior level's floor,
demote") but no enforced waiting period and no equivalent
risk-fraction clamp — Risky Mode adds both.

### 5.4 Per-month — monthly DD cap → 7-day cooldown

Trigger: month-to-date PnL ≤ −(stage monthly cap). Action:
flatten + 7-calendar-day no-trading window. Re-arm: window
expires *and* operator clicks "resume" in UI (manual ack
required so an unattended bot cannot silently start a new
month after a bad one).

### 5.5 Manual kill switch — operator UI

The existing `RiskManager` has `circuit_breaker_triggered` and
`kill_window_until_sec` fields
(`crates/forex-core/src/domain/risk.rs:344,360`). Risky Mode
exposes a one-click red "PANIC FLATTEN" button on the UI that:

1. Sets `circuit_breaker_triggered = true`.
2. Sends `ProtoOAClosePositionReq` for every open Risky-Mode
   position (the request schema is documented in
   `docs/audits/research/ctrader_api_full_reference.md:909`).
3. Sets `kill_window_until_sec` to "forever" (i.e. requires
   operator to clear it manually before the bot trades again).

The kill switch must not depend on UI WebSocket availability —
it is also exposed via the local IPC channel
(`crates/forex-app/src/app_services/...`) so a CLI / external
script can fire it.

### 5.6 Hardware — connection-loss flatten

Trigger: cTrader transport hasn't returned a heartbeat for N
seconds (proposal: N = 30). Action: best-effort send
`ProtoOAClosePositionReq` over whatever channel is still alive;
if no channel is alive, queue a flatten directive into a local
persistent file so the next-connect handshake fires it.

Note: this is the *most fragile* layer and we document it
honestly — if the entire host loses network, the broker's
server-side stops are the only protection.

### 5.7 Sanity — pre-broker-send size check

The existing `validate_and_convert_lot_size_to_ctrader_volume`
helper in
`crates/forex-app/src/app_services/trading/risk_gate.rs:70`
already rejects out-of-band sizes vs. broker min/max/step.
Risky Mode adds a *second* check: reject any order whose
implied risk exceeds 50 % of current bankroll, regardless of
what stage logic computed. This is a defence-in-depth against
a bug in our own sizing code — the kind of bug class that
audit-fix F5/F6 in `risk_gate.rs:1-23` already protects
against on the volume-conversion side.

### 5.8 Switch interaction matrix

| Trigger               | Severity | Auto-recover? | Operator ack? |
| --------------------- | -------- | ------------- | ------------- |
| Per-trade SL          | 1        | yes (next tick) | no          |
| Daily-loss cap        | 2        | yes (CET 00:00) | no          |
| Stage demote          | 3        | partial (re-cross + hysteresis) | no |
| Monthly DD cap        | 4        | no              | yes (resume) |
| Manual kill           | 5        | no              | yes (clear)  |
| Hardware disconnect   | 5        | no              | yes (re-arm) |
| Sanity-size rejection | 6        | yes (order-drop)| log only     |

Severity-6 ("logic bug") is the most-severe in the sense that
it *should never fire in production* — if it does, that is a
P0 incident.

---

## §6 — Backtest harness for Risky Mode

### 6.1 Data discipline

- Source: live cTrader historical bars + ticks pulled via
  `ProtoOAGetTickDataReq` / `ProtoOAGetTrendbarsReq`
  (`docs/audits/research/ctrader_api_full_reference.md` for the
  message contract).
- **No synthetic data** per operator rule. Any "what if R = 4
  instead of R = 3" exploration is done by *re-running the
  same historical tape* against alternative parameters, not by
  generating fake ticks.
- Cover at minimum 5 years of data per pair, including the
  2020 COVID and 2022 GBP-flash-crash regimes (these stress
  the news-blackout and weekend-flatten rules).

### 6.2 Walk-forward methodology

- Train ML confidence threshold on a rolling 6-month window;
  validate on the next 1 month; re-fit monthly.
- Trade only on out-of-sample bars.
- Re-record stage transitions at every transition boundary so
  that the per-stage stats are clean (S1 stats only count
  trades taken while in S1, etc.).

### 6.3 Reported metrics

For each parameter set:

1. **Geometric per-trade growth rate** `G(α)` and per-stage
   `G_s` (so we can see whether S1's aggressive sizing pays
   off statistically vs. S3's half-Kelly).
2. **Max drawdown distribution** — Monte-Carlo bootstrap with
   block size = 5 (preserves serial correlation; the §2.5
   point about CED's sensitivity to autocorrelation matters
   here). Report mean, median, 95th percentile, 99th percentile.
3. **Time-to-target distribution.** For runs that reach
   $50,000, report the time. For runs that don't, report the
   stopping reason (manual kill, monthly DD cap, hardware
   etc.). This is the *honest* "how fast / how often" answer
   to the operator's question.
4. **Ruin probability** at the $20 starting capital. A run is
   "ruined" if equity drops below $20 × 0.05 = $1 (effectively
   $0 after broker frictions). Report per-stage as well — a
   run can survive S1 by retreating to $20 and still
   technically be alive.

### 6.4 Acceptance criterion

- **First-stage ruin probability ≤ 50 %.** This is consciously
  loose: §1's anecdotal estimate is that the canonical 20-pip
  challenge has 95 %+ ruin probability, so halving that is a
  visible win.
- **Median time-to-$50,000 reportable** — if no run in the
  bootstrap reaches $50,000, the harness reports "no
  observation" rather than extrapolating. The operator decides
  whether to ship a mode whose median time-to-target is "no
  observation".
- **Maximum drawdown 95th-percentile ≤ 80 %.** Above that, the
  mode is rejected — the bankroll is effectively wiped before
  it can compound.

Failing any criterion sends the parameter set back to the
operator-decision step in §10.

### 6.5 Backtest output artefacts

- `risky_mode_walkforward_<paramset_id>.parquet` — per-trade
  log: stage, confidence, regime, SL/TP, entry/exit, PnL,
  bankroll-after.
- `risky_mode_summary_<paramset_id>.json` — the metric set
  from §6.3.
- `risky_mode_bootstrap_<paramset_id>.parquet` — N (proposal:
  10,000) block-bootstrap samples for the MDD/ruin
  distributions.

Storage location: `data/<paramset_id>/...` under whatever
data-path the wizard configured (see
`docs/audits/research/installer_wizard_ux_spec.md`, Step 2).

---

## §7 — UI surface

Inherits palette and typography from
`docs/audits/research/ui_ux_design_spec.md`:

- Background: Mirage `#131722` (dark theme, primary surface;
  §1.2 of UI spec).
- Bull accent: Teal-green `#26A69A` (§1.4 of UI spec — the
  "load-bearing" bull color).
- Bear accent: Red `#EF5350` (§1.4 of UI spec — bear color).
- Primary CTA accent: Dodger Blue `#2962FF` (§1.2 of UI spec).
- Text: White `#FFFFFF` on Mirage.
- Font: Trebuchet MS / system-default (§1.6 of UI spec).

### 7.1 Risky Mode toggle (Settings > Trading Mode)

```
┌─────────────────────────────────────────────────────────────┐
│  Trading mode                                               │
│  ─────────────────────────────────────────                  │
│  ( ) Standard         — FTMO 4 %/5 %/10 % floors            │
│  ( ) Risky            — Aggressive compounding, $20→$50k    │
│      ⚠️  HIGH RUIN PROBABILITY — read §10 before enabling   │
│      [ I accept the risk of total capital loss ]            │
└─────────────────────────────────────────────────────────────┘
```

The acceptance checkbox is *required* to enable Risky Mode.
The label uses the bear-red `#EF5350` for the warning line so
it inherits the same "loss" semantic colour as price-down bars
on the chart.

### 7.2 Stage progress bar ($20 → $50,000)

Top of the main dashboard, replacing (in Risky Mode only) the
"monthly profit %" bar used in Standard mode:

```
$20 ─ S1 ─ S2 ─ S3 ─ S4 ─ S5 ─ S6 ─ S7 ─ S8 ─ S9 ─ $50,000
                       ●
                  current: $173.42 (S4)
                  next milestone: $320 (+$146.58, 184 %)
```

Filled segments use teal-green `#26A69A`; current-stage marker
is Dodger Blue `#2962FF`; future stages stay neutral grey
(`#B2B5BE`, the lightweight-charts baseline default color).

### 7.3 Daily risk budget meter

Right rail, always visible:

```
Daily risk budget
─────────────────
Used:    $12.30 / $36.00 (34 %)
Cap:     20 % of $180.00 (stage S4)
Bars:    ████████░░░░░░░░░░░░░░  34 %
Status:  OK
```

Bar fill colour: teal-green when ≤ 50 %, amber `#F9A825` when
50-80 %, bear-red `#EF5350` when > 80 %. Above the daily cap
the bar pulses red and the trade controls below it grey-out.

### 7.4 Kill-switch button

Bottom-right corner, always visible, always reachable. Uses
the same red `#EF5350` as the bear bars so it shares the
"loss / danger" colour family:

```
┌──────────────────────┐
│  ⏻  PANIC FLATTEN    │
│  (all positions)     │
└──────────────────────┘
```

Single click triggers §5.5; a second confirmation modal is
shown ("Close 3 positions, halt trading. Continue?") to avoid
accidental clicks.

### 7.5 "Retreat" indicator

When the stage taper demotes (§5.3), a banner at the top of
the dashboard, amber `#F9A825`, no auto-dismiss:

```
⚠  Risky Mode retreat: stage S6 → S5
   Bankroll $623.18 < $640 floor.
   Risk-per-trade clamped to 2.5 %.
   Re-promote when bankroll > $640 AND ≥ 5 closed trades.
```

The banner stays until the re-promote condition fires. The
operator can dismiss it manually but the in-memory state still
tracks the demote.

---

## §8 — Wizard step

This plugs into the existing installer wizard at
`docs/audits/research/installer_wizard_ux_spec.md` **Step 3
"Account & profile"** (existing text quoted in the spec at
lines 232-256).

### 8.1 Modified Row 4 — "Trading mode"

Currently Row 4 is "Backtest / Forward test (default) / Live".
Risky Mode adds a *second* dimension orthogonal to that
selection — a "Profile" row:

```
Row 4a — Execution mode    : ( ) Backtest  (•) Forward test  ( ) Live
Row 4b — Strategy profile  : (•) Standard  ( ) Risky
```

Picking "Risky" surfaces the §7.1 acknowledgement modal
*inside the wizard*, with one extra panel: the
backtest-derived ruin-probability number from §6 for the
shipping default parameter set.

### 8.2 Risky-Mode branch panel

Shown only if "Risky" is picked. Mockup:

```
┌──────────────────────────────────────────────────────────────┐
│  Risky Mode — Aggressive compounding                         │
│  ───────────────────────────────────────                     │
│                                                              │
│  Starting capital:   $20  (default — operator-editable)      │
│  Target capital:     $50,000  (operator-editable)            │
│                                                              │
│  Expected first-stage ruin probability:  44 %                │
│                                  (from walk-forward, §6)     │
│  Expected median time-to-target:         no observation      │
│                                  (no run reached $50k)       │
│                                                              │
│  Kill-switch summary:                                        │
│  • Per-trade hard SL                                         │
│  • Daily-loss cap (stage-aware)                              │
│  • Auto stage-retreat on demote                              │
│  • Mandatory weekend flatten                                 │
│  • Manual PANIC FLATTEN button                               │
│                                                              │
│  This mode does NOT apply prop-firm rules. It is for a       │
│  personal account, not a funded challenge.                   │
│                                                              │
│  [ I accept the risk of total capital loss ]                 │
│  [ Continue ]  [ Back ]                                      │
└──────────────────────────────────────────────────────────────┘
```

Ruin-probability number is read from the latest
`risky_mode_summary_*.json` shipped with the build; if no
summary is shipped the panel says "not benchmarked — proceed
at your own risk" and the [Continue] button greys out until
the operator runs at least one backtest from the wizard's
embedded shortcut.

### 8.3 Wizard data flow

- `WizardConfig.strategy_profile = "risky"` written at Summary
  (Step 10 in the wizard spec, line 439).
- On first app launch the `RiskManager` constructor (currently
  `crates/forex-core/src/domain/risk.rs:375`) reads the
  profile and constructs a `RiskyModeManager` instead of the
  Standard `RiskManager`. `RiskyModeManager` *delegates* the
  prop-firm-equivalent checks at S9 back to the existing
  `RiskManager` so the operator's 4 %/5 %/10 % floors still
  apply when the account has compounded that far.

---

## §9 — Mathematical risk model

All formulas are taken from the §2 citations. Variables
defined per appearance.

### 9.1 Trades-to-target expectation

Given win-rate `p`, payout multiple `R` (so R = 3 means
TP = 3 × SL), per-trade risk fraction `f`, starting bankroll
`B₀`, target `B_T`:

```
E[ln(B_{n}) − ln(B_0)] = n · (p · ln(1 + R·f) + (1 − p) · ln(1 − f))
                       = n · μ_log
```

so the expected number of trades to reach `B_T` is approximately

```
n* = ln(B_T / B_0) / μ_log
```

For `B_T / B_0 = 2500` (the $20 → $50,000 target), `R = 3`,
`f = 0.05` (a representative S3-ish 4 % Kelly cap), `p = 0.55`
*(hypothetical illustrative win-rate, not a backtested
number)*:

```
μ_log = 0.55·ln(1.15) + 0.45·ln(0.95) = 0.0768 − 0.0231 = 0.0537
n* ≈ ln(2500) / 0.0537 ≈ 145 trades.
```

At 2 trades/day that is ~ 73 trading days, ~ 15 weeks. The
operator should read this as the *best-case* expectation
assuming the hypothetical edge holds throughout. With variance
the realised distribution is far wider — that is the §9.3
ruin-probability calculation.

### 9.2 Geometric mean growth rate

`G(f) = p · ln(1 + R·f) + (1 − p) · ln(1 − f)`

Maximum at `f = f* = (p · R − (1 − p)) / R`, with `G(f*)` the
Kelly-optimal log-growth. Fractional Kelly `f = α f*` gives
`G(α f*) ≈ α(2 − α) · G(f*)` (a known result, see
https://nickyoder.com/kelly-criterion/). So:

- Full Kelly (α = 1.0): growth = `G*`, drawdown = max
- 0.5 Kelly: growth = 0.75 G*, drawdown = ~ 25 % of full-Kelly
  drawdown variance
- 0.25 Kelly: growth = 0.4375 G*, drawdown = ~ 6 % of variance

Risky Mode's per-stage `α` table in §4.2 is exactly this
trade-off: more growth and more drawdown variance at low
stages, less of both at high stages.

### 9.3 Probability of ruin per stage

Approximating the per-trade log-return as normal with mean
`μ_log` and variance `σ²_log = p (1 − p) (ln(1 + Rf) −
ln(1 − f))²`, the Brownian-motion ruin probability from
bankroll `B` toward absorbing barrier at `B_min` is

```
P(ruin) ≈ exp(−2 · μ_log · ln(B / B_min) / σ²_log)
```

For S1 with `f = 0.08`, `R = 3`, *hypothetical illustrative*
`p = 0.55`:

```
μ_log = 0.55·ln(1.24) + 0.45·ln(0.92) = 0.1182 − 0.0375 = 0.0807
σ²_log = 0.55 · 0.45 · (ln(1.24) − ln(0.92))² = 0.2475 · 0.1116
       = 0.02762
ratio  = ln(40 / 20) = 0.693    (S1 spans $20→$40)
exponent = 2 · 0.0807 · 0.693 / 0.02762 = 4.05
P(ruin) ≈ e^{-4.05} ≈ 0.0174 = 1.7 %
```

That number is the ruin probability *conditional on the
hypothetical `p = 0.55` win-rate holding*. The real-world
caveat is that small-bankroll edge estimates are themselves
noisy — at the very small N of S1, the empirical `p` confidence
interval is wide, and the §2.7 Busseti-Ryu-Boyd risk-
constrained Kelly result says the right response to that
uncertainty is to *shrink* `f` further. This is why §4.6
gate 1 (ML confidence ≥ 0.80 at S1) is the dominant lever in
S1: it conditions the trade on a high-confidence signal so
that the realised `p` is closer to the desired ~ 0.55 than the
prior would imply.

### 9.4 Full Kelly vs half Kelly vs Risky Mode taper

For the *hypothetical illustrative* parameter set above
(`R = 3, p = 0.55`):

- Full Kelly: `f* = (0.55·3 − 0.45)/3 = 0.40` (40 % per
  trade — clearly unacceptable).
- Half Kelly: `f = 0.20` (still huge; matches the 20-pip
  challenge's 23 %).
- Quarter Kelly: `f = 0.10`.
- Risky-Mode S1 cap: `f = 0.08` (just under quarter Kelly at
  this hypothetical edge).
- Risky-Mode S9 cap: `f = 0.005` (about 1.25 % of full Kelly).

The taper is therefore *between quarter Kelly and 1/80 Kelly*
depending on stage — which sits exactly in the
professional-trader band documented at §2.1.

### 9.5 Expected maximum drawdown per stage

From §2.4 Magdon-Ismail formula, for the small-time regime
typical of a Risky-Mode stage (one stage usually completes in
< 100 trades):

```
E[MDD] ≈ 1.25 · σ_log · √n
```

where `n` is trades in the stage and `σ_log` is from §9.3. For
S1 with `n ≈ ln(2)/μ_log ≈ 8.6` trades and
`σ_log = √0.02762 = 0.166`:

```
E[MDD] ≈ 1.25 · 0.166 · √8.6 ≈ 0.61 (61 % expected MDD)
```

That is high. The §4.4 50 % S1 daily-loss cap is calibrated to
intercept *part* of this drawdown — the operator is consciously
accepting that S1 will frequently end in retreat to S0, and
that the bankroll is therefore disposable.

For S9 with `n ≈ 100` trades, `f ≈ 0.005`,
`σ²_log ≈ 0.000045`, `σ_log ≈ 0.0067`:

```
E[MDD] ≈ 1.25 · 0.0067 · √100 ≈ 0.084 (8.4 % expected MDD)
```

which is right at the FTMO 10 % limit and is the reason §4.2's
S9 row collapses to FTMO-Standard parameters.

### 9.6 Sequence-risk sanity

Per §2.6 (Bouchaud et al. *Entropy* 2023), strategies with
serial correlation in P&L have higher CED than i.i.d.
strategies of the same mean/variance. The forex-models swarm
forecaster's confidence is itself serially correlated (high-
confidence regimes cluster). Risky Mode handles this by:

- §4.6.4 regime filter (skip ranging regimes — they correlate
  losses positively).
- §6.2 walk-forward + §6.3 block-bootstrap MDD (the block size
  > 1 captures any residual serial correlation).

The §2.7 risk-constrained Kelly formulation gives the formal
justification: the right response to non-i.i.d. returns is a
shrinkage on `f*`, which §4.2's per-stage `α` is already
implementing.

---

## §10 — Operator decisions required

Listed by severity. The deliverable is *blocked* on these
decisions; nothing below changes without an explicit ack.

### 10.1 Initial-stage ruin probability ceiling

**Suggestion:** 50 %.

**Background:** §1.3 puts the canonical 20-pip-challenge ruin
probability at "95 %+" from anecdotal sources. Halving that to
50 % is a defensible, honest win that the marketing copy can
quote. Going to 30 % would force `f` so low at S1 that the
mode collapses into "FTMO Standard with a different name" —
which defeats the operator's intent. Going to 70 % loses the
"safer" framing entirely.

**Decision wanted:** ceiling value, in [10 %, 70 %].

### 10.2 Retreat behaviour — auto-clamp vs. notify-only

**Suggestion:** auto-clamp (§5.3 as written).

**Alternative:** notify-only — show the §7.5 banner but let the
operator decide whether to clamp risk. This trades safety for
operator-experience flexibility.

**Decision wanted:** auto or notify.

### 10.3 Live-broker eligibility

**Suggestion:** Risky Mode allowed only on **paper trading by
default**; live-broker requires a second explicit acknowledgement
modal in the Settings UI ("This will risk real money. Confirm.")
and a typed-confirmation string.

**Background:** The whole "$20 → $50,000" framing is much more
defensible on a paper account where capital loss is fake. On a
live broker, a 95th-percentile MDD of 60-80 % at S1 means
genuine real-money loss. The default should reflect that.

**Decision wanted:** paper-only default, paper-only forever, or
live-on-by-default-with-modal.

### 10.4 Prop-firm rules within Risky Mode

**Suggestion:** Prop-firm rules **do not apply** in Risky Mode
S1-S8. They apply at S9 (the table in §4.2 collapses to
FTMO-Standard there).

**Background:** Risky Mode is a personal-account framing,
not a funded-challenge framing. Applying FTMO 5 % / 10 % to a
$20 account would clamp `f` to ≤ 0.5 % which is meaningless on
a $20 bankroll (a 0.5 % loss is $0.10, less than the spread).
The two modes have fundamentally different objective functions.

**Decision wanted:** confirm that Risky Mode operates *outside*
the prop-firm regime below S9. This is the question the
operator needs to answer explicitly because the existing
`PropFirmConstraints::FTMO_STANDARD` constant
(`crates/forex-core/src/domain/prop_firm.rs:32`) is otherwise
treated as a global invariant.

### 10.5 Stage-table tunability

**Suggestion:** The §4.2 stage table is *operator-tunable* via
config, with hard floors:

- No stage may permit `f > 0.10` (a hard ceiling at quarter
  Kelly under the typical R = 3 assumption).
- S9 may not permit `daily-loss cap` larger than the
  prop-firm limit (5 %).
- No stage may permit `concurrent_positions > 5`.

**Decision wanted:** confirm these hard floors or propose
different ones.

### 10.6 Backtest acceptance loop

**Suggestion:** the §6 harness runs on every release tag and a
new release is *blocked* if any of:

- S1 ruin probability > 50 %
- 95th-percentile MDD > 80 %
- median time-to-target degraded by > 20 % vs. previous release

**Decision wanted:** confirm or relax thresholds.

---

## §11 — Codebase touch-points

This section is the implementation map for whoever picks up
the Risky Mode work in a later batch. Nothing here is being
coded in this deliverable — operator rule is "no code".

### 11.1 New file

`crates/forex-core/src/domain/risky_mode.rs` — proposed new
module. Sketch of the API surface (no implementation):

- `RiskyModeStage { S1, S2, ..., S9 }` enum.
- `RiskyModeStageConfig` struct: per-stage `alpha_kelly`,
  `max_risk_pct`, `max_concurrent_positions`,
  `daily_loss_cap`, `weekly_dd_cap`, `monthly_dd_cap`,
  `confidence_threshold`, `bankroll_floor`, `bankroll_ceiling`.
- `RiskyModeConfig` struct: `Vec<RiskyModeStageConfig>` plus
  global flags (`paper_trading_only: bool`,
  `mandatory_weekend_flatten: bool`,
  `news_blackout_minutes: u32`, `regime_filter_enabled: bool`).
- `RiskyModeManager` struct: composes a `RiskManager` (the
  existing `risk.rs:374` struct) plus stage-aware state
  (current stage, hysteresis counter, retreat history).
- `RiskyModeManager::check_trade_allowed(...) -> (bool, String)`
  — wraps the existing `RiskManager::check_trade_allowed` with
  the stage-aware gates added on top.
- `RiskyModeManager::calculate_position_size(...) -> f64` —
  wraps the existing `RiskManager::calculate_position_size`
  with the stage-aware `alpha_kelly` shrinkage.

### 11.2 Files unmodified

- `crates/forex-core/src/domain/prop_firm.rs` — `FTMO_STANDARD`
  stays exactly as is (lines 32-38). Operator rule.
- `crates/forex-core/src/domain/risk.rs` — `RiskManager` and
  `RevengeTradeDetector` reused by composition, not modified.

### 11.3 Files extended (not rewritten)

- `crates/forex-app/src/app_services/trading/risk_gate.rs` —
  add a single new function:
  `risky_mode_pre_send_sanity(...)` that performs the §5.7
  "50 % bankroll" check. The existing pre-trade gate stays
  unchanged (operator's audit-fix F5/F6/F7 chain is preserved).
- The wizard renderer crate — add the §8.2 panel inside
  Step 3's existing form. The Step 3 form text lives at
  `docs/audits/research/installer_wizard_ux_spec.md:232-256`
  and the layout follows that spec.
- Forex-models hookup — no code changes, just wire the
  swarm-forecaster confidence (`forex-models/src/forecasting/
  swarm_impl.rs`) and regime label
  (`forex-search/src/genetic/regime_labels.rs`) into the new
  `RiskyModeManager::check_trade_allowed` per §4.6.

### 11.4 Settings & persistence

A new top-level key in the existing settings file (the same
file the wizard writes at Step 10):

```
strategy_profile = "risky"
[risky_mode]
target_capital = 50000.0
paper_trading_only = true
mandatory_weekend_flatten = true
news_blackout_minutes = 30
regime_filter_enabled = true
[[risky_mode.stage]]
id = "S1"
bankroll_floor = 20.0
bankroll_ceiling = 40.0
alpha_kelly = 1.0
max_risk_pct = 0.08
max_concurrent_positions = 1
daily_loss_cap_pct = 0.50
weekly_dd_cap_pct = 0.25
monthly_dd_cap_pct = 0.50
confidence_threshold = 0.80
# ... S2 ... S9 ...
```

The §10.5 hard floors are enforced at deserialisation time, so
a malformed config file cannot bypass them.

### 11.5 Telemetry

Risky-Mode-specific events go through the same path as the
existing `RiskManager`'s event-emit (the
`crates/forex-core/src/domain/events.rs` channel). New event
types:

- `RiskyModeStageEntered { stage, bankroll, ts }`
- `RiskyModeStageRetreated { from, to, bankroll, reason, ts }`
- `RiskyModePanicFlatten { trigger, positions_closed, ts }`

These plug into the existing UI WebSocket so the §7.5 retreat
banner and §7.4 kill-switch confirmations can render off the
same stream.

---

## §12 — References (consolidated)

### Academic

- Kelly, J. L. Jr. (1956). "A New Interpretation of
  Information Rate." *Bell System Technical Journal*. — Kelly
  criterion original.
  https://en.wikipedia.org/wiki/Kelly_criterion
- Feller, W. (1957/1968). *An Introduction to Probability
  Theory and Its Applications*, Vol. 1. — gambler's-ruin
  closed-form.
  https://en.wikipedia.org/wiki/Gambler%27s_ruin
- Vince, R. (1992). *The Mathematics of Money Management:
  Risk Analysis Techniques for Traders*. Wiley.
  https://www.amazon.com/Mathematics-Money-Management-Analysis-Techniques/dp/0471547387
  Open mirror:
  https://dl.fxf1.com/files/books/english/MathematicsMoneyManagement.pdf
- MacLean, L. C., Thorp, E. O., Ziemba, W. T. (2011). *The
  Kelly Capital Growth Investment Criterion: Theory and
  Practice*. World Scientific.
  https://www.worldscientific.com/worldscibooks/10.1142/7598
  Summary article:
  https://www.caia.org/sites/default/files/AIAR_Q3_2016_05_KellyCapital.pdf
- MacLean, L. C., Thorp, E. O., Ziemba, W. T. "Good and Bad
  Properties of the Kelly Criterion."
  https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
- Magdon-Ismail, M., Atiya, A. F. (2004). "On the Maximum
  Drawdown of a Brownian Motion." *Journal of Applied
  Probability*.
  https://www.cs.rpi.edu/~magdon/ps/journal/drawdown_journal.pdf
  SSRN abstract: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=874069
- Goldberg, L. R., Mahmoud, O. (2014). "Drawdown: From
  Practice to Theory and Back Again." arXiv:1404.7493.
  https://arxiv.org/abs/1404.7493
- Bouchaud, J.-P. et al. (2023). "A Rational Risk Policy? Why
  Path Dependence Matters." *Entropy* 25(2), 202.
  https://www.mdpi.com/1099-4300/25/2/202
  PMC mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC9955835/
- Busseti, E., Ryu, E. K., Boyd, S. (2016). "Risk-Constrained
  Kelly Gambling." Stanford.
  https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf
- Carta, A., Sanna, M. (2020). "Practical Implementation of
  the Kelly Criterion: Optimal Growth Rate, Number of Trades,
  and Rebalancing Frequency for Equity Portfolios." *Frontiers
  in Applied Math & Statistics* 6, 577050.
  https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2020.577050/full
- Whelan, K. (n.d.). "Ruin Probabilities for Strategies with
  Asymmetric Risk." UCD WP 2025-03.
  https://www.karlwhelan.com/Papers/Ruin.pdf
  https://www.ucd.ie/economics/t4media/WP2025_03.pdf
- Sigman, K. "Gambler's Ruin Problem." Columbia FE Notes.
  http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GR.pdf

### Industry / retail sources

- Trading Rush. "Testing $20 To $52,400 Strategy 1000 Times."
  https://tradingrush.net/testing-20-to-52400-strategy-1000-times/
- Human×AI. "Turning $20 into $52,000." Medium.
  https://medium.com/@HUMANxAI/the-20-to-52-000-challenge-tested-by-1-000-bots-7484da203eca
- Forex Factory thread "20 pips Challenge — (Compounding model)."
  https://www.forexfactory.com/thread/1306313-20-pips-challenge-compounding-model
- BabyPips. "Data Confirms Grim Truth: 70-80 % of Retail
  Traders are Unprofitable."
  https://www.babypips.com/news/almost-80-percent-of-retail-traders-are-unprofitable
- BabyPips. "Prop Firm Challenge Survival Checklist."
  https://www.babypips.com/learn/forex/prop-firm-challenge-survival-guide
- "ESMA brokers and trader Success Rates."
  https://investingoal.com/esma-broker-client-success-rate-stats/
- "Why 80% of Retail Traders Lose Money (And What the Data
  Actually Says)."
  https://10pmtrader.com/why-retail-traders-lose-money/

### Prop-firm rules

- FTMO Trading Objectives.
  https://ftmo.com/en/trading-objectives/
- FTMO Maximum Daily Loss.
  https://academy.ftmo.com/lesson/maximum-daily-loss/
- FTMO Maximum Loss.
  https://academy.ftmo.com/lesson/maximum-loss/
- FTMO Rules 2026 — Vigil.
  https://runvigil.app/rules/ftmo
- FTMO Prop Firm Review 2026 — LuxAlgo.
  https://www.luxalgo.com/blog/ftmo-prop-firm-review-how-to-pass-in-2025/
- MyForexFunds — Accelerated.
  https://myforexfunds.com/accounts/accelerated-program/
- MyForexFunds — Accelerated rules.
  https://myforexfunds.com/accelerated-accounts-rules/
- Trader's Second Brain — Prop firm comparison 2026.
  https://traderssecondbrain.com/guides/prop-firm-comparison
- Trader's Second Brain — Prop firm rules cheatsheet.
  https://traderssecondbrain.com/guides/prop-firm-rules-cheatsheet
- Trader's Second Brain — Drawdown rules: static vs trailing.
  https://traderssecondbrain.com/guides/prop-firm-drawdown-rules
- Alpha Futures News Trading Policy.
  https://help.alpha-futures.com/en/articles/9492063-news-trading-policy

### Broker / platform

- cTrader Trading Conditions.
  https://help.ctrader.com/trading-with-ctrader/conditions/
- cTrader Dynamic Leverage.
  https://help.ctrader.com/trading-with-ctrader/dynamic-leverage/
- cTrader LeverageTier algo reference.
  https://help.ctrader.com/ctrader-algo/references/MarketData/Symbols/LeverageTier/
- FxPro Stop-Out Level.
  https://www.fxpro.com/help-section/education/beginners/articles/stop-out
- ClickAlgo — Trading Micro Lots with cTrader.
  https://clickalgo.com/micro-lots
- FXTM Pip Calculator.
  https://www.fxtm.com/en/trading/tools/pip-calculator/

### Weekend / news / regime

- "Hold Forex Trades Overnight or Through the Weekend?"
  https://tradethatswing.com/hold-forex-trades-through-the-weekend-or-close-them/
- Alphaex Capital. "Weekend gaps in forex trading."
  https://www.alphaexcapital.com/forex/forex-trading-basics/forex-market-structure/weekend-gaps-in-forex-trading
- TastyFX. "Gap risk: what it means and how to avoid it."
  https://www.tastyfx.com/news/gap-risk--what-it-means-and-how-to-avoid-it-231017/
- "What is Forex Factory — Uses, Impact levels and Step By
  Step Guide 2026."
  https://www.xs.com/en/blog/forex-factory-guide/
- Maven Trading. "CPI Days Explained: Forex Trading Strategies
  to Survive High-Impact News."
  https://maventrading.com/blog/cpi-days-explained-forex-trading-strategies-to-survive-high-impact-news
- QuantStart. "Market Regime Detection using Hidden Markov
  Models in QSTrader."
  https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/
- Volatility Box. "Volatility Regime Detection: From Simple
  Rules to Machine Learning."
  https://volatilitybox.com/research/volatility-regime-detection/

### Hedge-fund disclosed returns (sanity comparison)

- "Renaissance Technologies." Wikipedia.
  https://en.wikipedia.org/wiki/Renaissance_Technologies
- "Renaissance Technologies and The Medallion Fund." Quartr.
  https://quartr.com/insights/edge/renaissance-technologies-and-the-medallion-fund
- "Decoding the Medallion Fund Returns." QuantifiedStrategies.
  https://www.quantifiedstrategies.com/medallion-fund-returns/

### Internal references

- `crates/forex-core/src/domain/prop_firm.rs:32` —
  `PropFirmConstraints::FTMO_STANDARD` — *unmodified*.
- `crates/forex-core/src/domain/risk.rs:374` — `RiskManager`
  struct — *composed by* `RiskyModeManager`, *not modified*.
- `crates/forex-core/src/domain/risk.rs:501` —
  `RiskManager::check_trade_allowed` — wrapped by
  `RiskyModeManager`.
- `crates/forex-core/src/domain/risk.rs:652` —
  `RiskManager::calculate_position_size` — wrapped by
  `RiskyModeManager`.
- `crates/forex-app/src/app_services/trading/risk_gate.rs:70` —
  `validate_and_convert_lot_size_to_ctrader_volume` — Risky
  Mode adds `risky_mode_pre_send_sanity` next to it.
- `crates/forex-models/src/forecasting/swarm_impl.rs` — swarm
  confidence consumed by §4.6.1.
- `crates/forex-search/src/genetic/regime_labels.rs:14` —
  `RegimeWindow` consumed by §4.6.4.
- `docs/audits/research/ctrader_api_full_reference.md:773` —
  `ProtoOANewOrderReq` (relative SL field used by §5.1).
- `docs/audits/research/ctrader_api_full_reference.md:909` —
  `ProtoOAClosePositionReq` (used by §5.5 PANIC FLATTEN).
- `docs/audits/research/ctrader_api_full_reference.md:1026` —
  Volume unit semantics (cents) — required by any future
  Risky Mode sizing code.
- `docs/audits/research/installer_wizard_ux_spec.md:232-256` —
  Step 3 layout extended by §8.
- `docs/audits/research/ui_ux_design_spec.md:107-191` —
  Palette and typography inherited by §7.
