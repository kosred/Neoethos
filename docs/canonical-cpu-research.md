# Run receipt-bound CPU research

`canonical-research` connects the existing exact acquisition matrix, broker
symbol snapshot, D1 screening-cost builder, streaming discovery and chronological
holdout evaluation in one invocation. It uses the explicitly supplied settings
file for startup, CPU admission, runtime overrides and discovery.

Candidate selection and sizing calibration use screening cost assumptions. The
separate final holdout remains reserved until replay is requested. An explicitly
supplied saved quote-replay bundle connects the existing quote and
execution-economics consumer after the portfolio is locked.
A negative screening run is not a verdict on the broker-cost pipeline.
An empty result names the last observed population that fell to zero, rather
than the stage with the largest raw rejection count. Missing terminal counters
leave the cause unknown; the diagnostic points to recorded evidence without
recommending weaker acceptance criteria.

New CPU trade logs retain `logged_trades[].trades[].exit_reason`: `stop`,
`target`, `trailing_stop`, `max_hold`, `data_gap` or `session_close`. The existing
device-outcome adapter also preserves reported exit codes; its generic `stop`
does not distinguish an original stop from a trailing stop. These are simulated
exit diagnostics, not broker fills. Legacy logs omit this field and their exit
cause must not be inferred from duration or PnL.

Use [the account evidence entrypoint](broker-account-evidence.md) to inspect the
actual account currency, broker PnL and reported commission/swap components.

```powershell
cargo build --locked -p neoethos-cli
& ./target/debug/neoethos-cli.exe canonical-research `
  --authority-root C:/path/to/canonical-trendbar-authority-v1 `
  --data-root C:/path/to/data `
  --plan-sha256 <acquisition-plan-sha256> `
  --matrix-sha256 <acquisition-matrix-sha256> `
  --broker-symbol-contract C:/path/to/bsc1-snapshot.json `
  --broker-account-snapshot C:/path/to/account/account-snapshot.json `
  --settings-source C:/path/to/config.yaml `
  --symbol EURUSD --base-timeframe H4 --higher D1 `
  --out-dir C:/path/to/new-research-run
```

The matrix selects the exact broker scope and base identity. An unrelated
dataset with the same display symbol cannot win an ambiguous inventory lookup.
Every requested higher timeframe must exist directly in that same matrix.
Settings, broker facts and cost assumptions are checked before feature work.
An existing output directory is refused so previous runs remain recoverable.

For a chronological 70% research split, add `--training-percent 70` and set
`models.discovery_runtime.funnel_stage1_pct` and `prefilter_insample_frac`
to `1.0`. Clear the global and timeframe-specific row caps. The feature
receipt binds the training prefix for both normalization and the GA. With a
non-default split the full prefix is searched, including in Risky mode. The
rest is divided between selection calibration and the final holdout. Supply
the same training percentage when reopening such a run.

`--screening-round-trip-pips 0.4` uses an operator model of 0.4 pip per
completed round trip for spread, slippage and commission together. The
command retains the broker cost envelope and binds the operator assumption
into the research contract. Signed overnight carry and applicable PnL
conversion still use the broker inputs. Use the same flag when reopening
the run. Broker quote replay requires the broker-derived execution model.

For large searches, `--population` times `--generations` defines the available
GA evaluation slots. The result's `whole_run_distinct_genomes` counts distinct
evaluated genomes. The optional `models.search_runtime.archive_mode: active`
also retains finite trading candidates with nonpositive training returns for
later validation. The retained archive and `--candidates` each have their own
cap. The candidate census records actual admission and every subsequent gate.

The default exploratory budget is 64 candidates per generation, 16 generations,
five indicators per gene, 16 finalist candidates and at most two streaming
batches. Override these explicitly with `--population`, `--generations`,
`--max-indicators`, `--candidates` and `--max-batches`; each accepts only a
positive integer. Zero does not mean unlimited work here. `--cpu-threads` is the
existing process CPU cap. Omitted `--higher` uses the supplied settings' ladder.

For a saved reviewed replay, supply all three flags:
`--quote-replay-bundle <json> --quote-replay-sha256 <sha256> --quote-replay-store <dir>`
and `--max-batches 1`. The store is the existing broker-truth acquisition store.
The bundle is `StoredFinalistQuoteReplayV1` schema 1: the exact acquisition outcome,
review metadata, and per-position entry financial inputs plus closed execution
economics. Positions are keyed by strategy lane and canonical decision row, with
an exact quote-ledger hash check; identical quotes in different lanes retain
their separate sizing and cash. The input hash and
acquisition link are checked before feature work. Replay reopens the reviewed
Vortex evidence and validates the locked portfolio, selection scope, sizing,
causal conversion and signed cash components. Missing, duplicate, unused or
inconsistent position inputs are refused. Deserialized metadata grants no
authority; the existing validator must independently reopen the immutable evidence.

To continue an existing selection, add
`--resume-research <research.json> --resume-research-sha256 <file-sha256>` and
`--max-batches 1` to the same canonical command, with a new output directory.
Use the original settings, broker snapshots, plan and matrix. The result's file
hash, internal evidence identity, complete feature receipt and rebuilt cost
contract must agree. Empty selections cannot be resumed. Search-budget overrides
(`--population`, `--generations`, `--max-indicators`, `--candidates`) are refused
because this mode does not run Search or repeat sizing calibration.

Resume reopens the exact recorded feature working set, including a later batch,
instead of selecting columns from today's free RAM. The existing funnel field
`live_trading_policy_v1` now persists the archived signal/exit configuration;
previous writers omitted it from JSON. A legacy result without that policy or
the recorded recipe/working set is refused. The existing signal producer reconstructs
the full SMC signals and final plan from canonical data and the archived policy.
Changed source values, feature math, final windows or policy fail the binding;
the command never silently reruns selection to repair a mismatch.

Without a replay bundle, this mode records acquisition coordinates in the existing
`research.json.streaming.json`: `selection_mode: FrozenResearchResumed`, the saved
file hash, original `source_batch_cursor` and `locked_finalists` containing the
signal-plan/portfolio/scope identities and exact evaluation window. These
coordinates are preparation data, not quote evidence. Supply the same saved
research plus all three quote flags when the reviewed bundle is available.
Research inputs are bounded to 512 MiB (the complete candidate/evidence ledger
can exceed 150 MiB); replay bundles retain their 64 MiB decoder limit. Both must
be regular files.

The bundle must match this frozen portfolio and final SMC policy. Omitting the
resume flags retains the original new-search behavior, which can select a
different portfolio and correctly refuse an older bundle. This interface does
not automatically acquire broker evidence. Current account/deal snapshots alone are not sufficient to produce
this bundle or establish historical fee policy. The local integration regression
uses synthetic observations in real Vortex storage; it does not establish that a
broker-backed finalist bundle has been acquired or that a strategy is profitable.

The streaming index records `quote_replay_status`, the input SHA256 and the
evidence path. `ReplayedResearchOnly` saves the existing V3 evidence to
`research.json.quote-replay.json`. `NotExecutedNoSelectedPortfolio` means the
quote consumer did not run; `Failed` preserves the batch failure. Neither is a
successful quote validation. With no bundle, the status is `NotRequested`.

Each run saves the exact `settings.yaml` and `broker-symbol-contract.json`
snapshots, `run-arguments.json`, `screening-costs.json`, `discovery-arguments.json`, the exact
research envelopes and `research.json.streaming.json`. The streaming index
records all successful and failed batches, whether evidence is complete for the
attempted batches, the candidate-selection outcome and whether the streaming
space was exhausted. A completed experiment with no candidate portfolio exits
successfully. Missing or failed batch evidence still returns an error after
preserving available evidence.

Inspect `discovery_result.funnel_profile.stage1_evaluation_window` to see the
actual row range on which the GA evolved genes. The default fast screen scores
the earliest 25% of the selection window. To evolve against all selection rows,
set `models.discovery_runtime.funnel_stage1_pct: 1.0` in the explicit research
settings file. This changes the search objective's history, not the reserved
calibration or final holdout. Record every experiment; choosing a profile from
its results is further selection, not independent validation.

The GA anneals its SMC threshold while evolving. Before finalist ranking, every
archived candidate and the complete last population are measured again at the
actual final threshold. Replay uses the same exact evaluator in waves no larger
than the admitted population; cancellation or a refused/incomplete evaluation
fails the handoff. Old profits measured at earlier thresholds cannot select the
validation cohort. This does not relax quality or walk-forward requirements.

Professional template admission now visits each resolvable family before its
next parameter variant. The initial budget remains `min(population / 10, 50)`;
with all five families available, population 200 admits four seeds per family.
The same owner supplies diversity-rescue seeds. Its structured
`neoethos_search::seed_templates` event records requested/emitted counts and
each family's count **before** population filtering; these counts do not attest
that seeds survived deduplication, SMC evaluation or finalist selection.
Missing families are skipped and the existing random fill retains population
size. A fixed RNG seed reproduces complete genomes, but changing traversal
changes random SMC rotations and gene order even when all 50 templates fit.
Compare the exact source, costs, RNG seed and compute budget on development
windows; an unchanged family census is not an unchanged GA population or an
OOS improvement. The synthetic `portfolio_selection_probe.py` enumerates role
signatures, not market backtests.

The existing `PreparedNettedCanonicalBarResearchV1::from_training_scope`
accepts explicit `InSample` and `SelectionValidation` scopes and refuses
`Holdout`; `from_input` retains the inverse final-scope restriction. Both use
the same cost-bound bar lifecycle and preserve the exact scope identity.
This adapter is used by the opt-in specialist research path below, not by the
desktop's normal portfolio selector. Passing the scope/accounting regression
does not establish a portfolio-aware production selector or quote validation.

For a new PropFirm CPU experiment, opt in through the supplied settings file:
`models.discovery_runtime.weekly_cadence: true`. This adds the versioned rolling
weekly objective; omitted/false retains legacy scoring. Every five consecutive
UTC weekdays in the complete evaluation interval must contain a trade entry.
Inactive prefixes, gaps and tails count. Multiple entries on one day cannot
cover another day. Observed Sunday FX entries cover the following Monday
when it lies inside the interval; Sunday and Monday cover the same day.
Saturday entries and a terminal Sunday without an evaluated Monday cover no day.
All fills retain their costs and trade counts. This sealed research calendar
is not broker-session or holiday evidence. A window with fewer than five weekdays is refused.

The existing CPU position lifecycle records completed trades' entry timestamps
alongside its unchanged eleven financial metrics, without retaining a full
population trade ledger. Feasible candidates precede infeasible candidates in
parent/survivor selection, bounded archive retention and finalist admission.
When all candidates are infeasible, normalized inactivity, net-loss and risk
deficits still distinguish them. Exact deficit ties use the original objective.
The full selection-window quality replay independently enforces cadence and
records `rolling_five_weekday_cadence` rejections. None of this is OOS evidence.

The weekly minimum is additional to the explicit run's existing acceptance
floors. A profile still configured for 20 trades/month continues to require
20; enabling this objective does not silently turn it into a weekly-only
profile. Record any operator-authorized activity-profile change in a new
settings snapshot, preserving the frozen old experiment and financial/risk
floors. Weekly limits participate in the search identity; the exact interval
and month convention are retained in `live_trading_policy_v1`'s sealed evaluator.
GA handoff metadata retains aligned cadence observations and checkpoints retain
them when present. Native routes refuse this objective before population
execution because they do not produce these observations. Novelty blending is
also refused rather than allowing novelty to override financial feasibility.

`funnel_profile.walkforward_selection_cohort` retains all completed internal WF
trials, including rejected candidates. Each links to the exact genome and its
index in `discovery_result.candidates`, selection scope and search config hash.
The compact fold records contain actual executed trade counts, PnL and risk
breaches; return arrays are released with each validation wave. Rejection reasons
distinguish nonpositive average PnL, fewer than 60% positive folds in Risky mode,
prop-firm rule failures and invalid/incomplete fold measurements. A zero-trade
fold remains part of the existing positive-fold denominator. Multiple reasons
can apply to one candidate. Older profiles omit these diagnostics rather than
inventing measurements. These selection diagnostics are not deployment artifacts.

Internal WF/CPCV evaluate the already-normalized selection frame. The global
normalizer is fitted on the first 80% selection prefix, which includes their
internal test rows; there is no per-fold normalization refit. The outer
calibration and final holdout rows are excluded from that fit. Internal WF/CPCV
results and candidate-selection export therefore do not establish untouched
chronological OOS evidence or promotion authority; promotion requires the
separate quote-validated outer holdout.

### Risky portfolio investigation, 2026-11-07

Source baseline: `0e45bfd7` on `codex/frozen-research-continuation`. These
findings were checked against current callers and archived run artifacts. No
new financial search, broker execution or GPU run was performed. The frozen
campaign remains stopped. New source/report changes are not part of its old
executable. Local source is six commits ahead of the published research branch;
the separate `research-correctness` worktree is an older, divergent checkpoint.

The recent runs below are under
`%LOCALAPPDATA%/neoethos/experiments/codex-takeover-20261026/runs/`.
Read `research.json` under `discovery_result.funnel_profile`, and the matching
`-oos/frozen-oos.json`. The OOS files contain diagnostic candidates; both actual
selected portfolios are empty.

| Risky run suffix | GA returned | Admitted to validation | Internal WF passed | Calibration passed | Selected portfolio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2-100-confidence-s20261021` | 2740 | 512 | 27 | 0 | 0 |
| `v2-100-confidence-s20261029` | 2883 | 512 | 28 | 0 | 0 |

The first calibration rejected 21 nonfinite metric summaries and six
unprofitable summaries. The second rejected 25 nonfinite summaries and three
capital-exhaustion paths. It is not established that relaxing a target or
diversity gate would rescue any of these candidates. The eight diagnostic OOS
candidates all lost money over the same 395.5-day tail. Net PnL ranged from
`-24.7944` to `-136.9846` on initial capital `100`. None reached `50000` in any
of its 12 overlapping 180-day goal windows. These windows are not independent
trials. Negative balances and drawdown above 100% describe this older numerical
policy, not executable broker economics. Returns after a nonpositive starting
balance are undefined; a median of only the remaining defined months must not
be presented as a typical calendar-month return.

The selection receipt covers about 8.5 years, but these two Risky survival-policy
runs actually evolved and replayed the last 12317 selection bars, about 178.26
days. `trim_to_row_budget` deliberately takes the last goal-length episode.
Their effective alphabet was 323 columns after prefiltering 3411. Therefore
these runs do not demonstrate survival across the full eight-year history.
Shortening the episode also does not supply exposure to multiple regimes.

The current pipeline has specific obstacles to a specialist portfolio:

* `risky_goal_calibration_rejection` requires each surviving candidate to reach
  the full account goal in at least one calibration episode and survive every
  episode. This happens before portfolio construction. A useful component
  cannot qualify by contributing only part of a portfolio's account goal.
* Risky internal WF requires positive average PnL and at least 60% positive
  folds, with inactive folds in the denominator. This tests individual broad
  coverage; it is not conditional validation of a specialist's declared state.
* Ordinary GA archive retention ranks by net profit. Its handoff is globally
  capped before validation, and Jaccard novelty compares indicator sets, not
  synchronized returns, entry overlap, session or causal market state. Weekly
  runs have a separate bounded vector archive; neither establishes that
  complementary trading behaviors are retained by the ordinary Risky search.
* Final diversity uses signal correlation. `DiversityKey` is telemetry and
  `StrategyRegimeProfile` describes retrospective calendar performance; neither
  is the connected causal specialist router required for live decisions.
* The existing trader combines ready gene votes into one net direction and
  averages confidence and brackets on the winning side. It does not execute
  each selected gene as an independently allocated strategy. Adding genomes
  can cancel signals or change executed stops. Per-gene OOS summaries cannot
  be added to obtain a shared-account portfolio result.
* The legacy `monthly_upside_v1` score awards five points at a 50% win rate
  and rewards the best month relative to the median. It does not directly
  optimize probability of reaching the configured account goal by its deadline.
  This is an incentive mismatch in runs using that table, not a demonstrated
  explanation for the newer `ga_fitness` runs' losses.

New survival-policy goal-window reports add `goal_progress_v1`: the peak
realized balance before the existing stop, and whether/when it first crossed
80% of the configured target balance. For `50000`, the diagnostic threshold
is `40000`; it is not 80% success probability or 80% of the gain. The threshold
does not bank capital, stop the simulation or satisfy the full-target gate.
A threshold crossing followed by ruin remains a ruined path. Legacy policy
windows retain their existing shape. New OOS documents explicitly declare
`evaluation_scope: individual_strategies` and
`combined_portfolio_evaluated: false`.

The required repair sequence is independent mode profiles (see
[configuration ownership](config-single-source-of-truth.md#mode-isolation)),
chronological training and selection with preprocessing fitted inside each
training fold, retention of candidates by measurable trading behavior, then
portfolio construction and replay under one shared capital/exposure budget.
Keep component eligibility separate from account-level goal acceptance; move
the account target only when the assembled portfolio has an actual connected
evaluator. Any specialist activation rule must use information available at the
decision time and be selected exclusively on training/calibration data. Assess
the frozen system on a later outer period. The already-inspected tail is
diagnostic and cannot become a fresh confirmation set through relabelling.

Report account-level net monthly returns, positive-month fraction, worst month,
losing streaks, drawdown, capital survival, target/80%-threshold attainment and
time to each threshold. Record every configuration/seed tried. More candidates,
more generations and more archived genomes alone do not establish an edge.
The statistical rationale is selection bias under multiple trials
([Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf));
retaining multiple behavior niches is supported as a search method by
[MAP-Elites](https://arxiv.org/abs/1504.04909). Neither paper establishes that
these NeoEthos strategies or the proposed portfolio are profitable.

### Matched GA selection comparison, 2026-11-07

The follow-up explicitly permits bounded experiments and excludes large
timeframes. The new independent campaign is
`%LOCALAPPDATA%/neoethos/experiments/ga-diversity-20261107/`; its frozen
versioned plans, executable/DLL hashes, source hashes, nine settings files and
run-specific comparison scripts identify the experiment. It reuses the guarded
`scripts/research_campaign.py` runner. The old campaign's STOP, plan and
executable are preserved. CPU execution is explicit; no GPU or live execution
is inferred.

The three arms are rank-weighted parents with rank-weighted survivors,
rank-weighted parents with generational replacement, and uniform parents with
generational replacement. Each has three predeclared seeds, population 128,
64 generations, at most five terms per gene and 512 post-GA validation
candidates. Thus each has 8192 generation-evaluation slots; final-policy replay
and distinct-genome counts are reported separately. The first run records the
input recipe. Fresh isolated searches use the same pinned feature budget/cache
and must match its consumed input receipt, ordered feature alphabet and all
consumed economic values/non-settings source components before entering the
comparison. Each complete settings/cost-provenance hash is retained separately.
Prior search memory is not loaded. The experiment was initially capped at
45 minutes; measured repeated input verification led to a prospectively declared
60-minute total cap before any matrix OOS outcome, with the same nine profiles.
The 10-minute job limit, 18 GiB admission
headroom and existing memory guards bound the work.

All arms use M15 alone, with multi-resolution disabled and no higher-timeframe
features. The existing prospective intraday reference profile supplies fixed-pip
generation, a 16-bar holding limit, trailing disabled, fixed per-trade risk
3/13 and the v3 account-observation/target-banking policy. Goals remain
100 to 50000 in 180 days. Its EUR monetary inputs remain reference screening
assumptions. The captured D1 price is used solely for the unchanged pip-value
and commission conversion reference, not as an input signal or confirmation.
These common settings differ from the older long-holding runs;
only comparisons within this frozen matrix isolate selector policy.

The IS-only diagnostic shortlist requires at least 20 trades, positive net PnL
and no total closed-balance loss. It selects two goal-pace and two monthly
consistency candidates, deduplicated to at most four. Every cohort is frozen
before any diagnostic tail replay. These candidates do not bypass portfolio
or promotion gates. Report exact entry/exit tape counts, a bounded sample of
entry-time overlap and calendar-aligned monthly-return correlations, alongside
individual OOS monthly PnL, drawdown, trade/month losing streaks, capital
survival and full/80%-target observations. None of these diversity statistics
by itself establishes independent strategies or a shared-account portfolio.

The selector probability-mass test verifies linear ranking's inverse CDF;
the best of N has probability 2/(N+1), rather than deterministic parenthood.
A separate reproduced bug affected novelty-enabled stagnation: per-generation
normalization could leave the top blended selection score unchanged despite
improving financial fitness. Ordinary novelty-enabled progress now tracks the
best finite original financial fitness. The novelty-disabled and weekly
progress routes retain their existing semantics. This matrix keeps novelty
disabled, as required by the Risky survival policy, so it tests survivor/parent
selection separately from that repair. Unit checks are implementation evidence;
the exposed historical tail remains diagnostic, not independent confirmation.

The initial strict recorded-input attempt stopped the second run before GA.
The complete assumption-source identity embeds the whole settings file SHA:
changing only survivor policy/fraction changed that identity. The two cost
documents differed solely in the settings source-component hash; all resolved
monetary values and other source components were equal. The strict production
guard was retained. Revision 2 uses fresh searches, verifies that configuration
differences are restricted to declared selector/seed fields, and checks the
complete consumed economic contracts independently of whole-settings provenance.
The original plan, failed attempt, logs and successful baseline are preserved;
the unchanged baseline is reused. Revision 3 freezes all nine IS cohorts before
starting replay under the final total deadline. These mechanical/timing changes
do not alter configurations, selected candidates or any promotion criterion.

Before any matrix OOS result, the IS evidence exposed one internal-WF passer
that survived all 12 calibration episodes but failed solely because it did not
reach 50000 alone. A separate auxiliary diagnostic rule therefore covers all
such WF passers across every arm. Its IDs are frozen with the complete cohort
before replay and its results remain separate from the original four-candidate
shortlists. This tests the account-goal/component mismatch without changing a
primary selection, goal, cost, strategy or acceptance gate.

The total deadline stopped revision 3 at 21:22:45 UTC. All nine IS searches
completed, but only seven of nine OOS cohort replays completed; the eighth
owned process was terminated by the deadline guard and the ninth was not
started. This is a partial matrix, not nine completed forward tests. The
paired policy comparison below uses only seeds 2026110701 and 2026110702,
eight frozen primary candidates per arm. The third uniform-parent cohort is
reported separately. All artifacts, including the interrupted logs and
`failure-v3.json`, are preserved.

| Policy | Mean distinct genomes per IS run, three seeds | Positive primary OOS candidates, two matched seeds | Median individual OOS net return |
| --- | ---: | ---: | ---: |
| Rank parents, 10% ranked survivors | 7525 | 0/8 | -99.8831% |
| Rank parents, generational replacement | 8192 | 1/8 | -97.1765% |
| Uniform parents, generational replacement | 8192 | 0/8 | -99.7118% |

This small comparison supplies no statistically established best selector.
Increased genome counts did not produce a usable portfolio. Every one of the
nine searches selected zero portfolios. Across all seven completed replays,
27 primary candidates plus one auxiliary candidate produced two positive
individual totals. Neither reached 50000 or the 40000 diagnostic threshold
in any of the 12 overlapping goal windows. The two positive primary totals
were +11.0392% with 80.5184% drawdown and +3.8142% with 68.7562% drawdown over
395.5 days. Their median calendar-month returns were -1.1088% and -7.3485%.
Each had seven positive calendar months out of fourteen; these totals do not
demonstrate a dependable monthly income. The auxiliary WF passer lost
99.9999999952%, with one positive month out of fourteen. Thus merely removing
its individual full-goal rejection would have admitted an observed failure.
Positive mathematical balances in the survival counter do not establish broker
tradability: minimum-lot and margin/stop-out constraints are unbound here, and
fractional sizing can leave an economically exhausted account above zero.

The deeper diagnosis is a mismatch between the search's statistical support
and what a high backtest score is being asked to establish:

* `trim_recent_history` evolves every Risky survival run on the last
  deadline-length selection episode: 12317 M15 rows, 178.2604 calendar days
  here. The 8.5-year selection receipt is not the GA's actual evaluation span.
  A 180-day account objective does not require learning exclusively from one
  180-day historical example.
* The nine searches spent 73728 generation-evaluation slots. Among their
  744 positive-net IS candidates, the median was five trades and 615 had fewer
  than twenty. These counts include related/repeated hypotheses and must not
  be treated as independent trials. Risky `score_risky_ga_fitness_goal_v6`
  rewards realized log-growth pace. After the zero-trade/invalid guards it
  does not distinguish sample sizes at equal net return and span. Risky
  survival setup also clears the single-candidate expectancy-significance
  check. More successful optimization of this objective can select sampling
  luck more efficiently; a larger archive cannot validate the resulting edge.
* Exact cached validity masks expose another source of limited support.
  The median positive candidate had all its terms valid on only 0.6901% of
  the episode's rows. 440/744 had less than 1% eligibility; 29/36 frozen
  primary/auxiliary shortlist members had less than 5%. Frequently selected
  confirmed-divergence outputs are finite only on event bars. The signal
  builder requires every term to be valid, so a sparse term also restricts
  the whole gene's possible decision times. This is a measured constraint,
  not proof that sparse event strategies are inherently bad or the sole cause
  of failure. Do not fill all invalid cells with zero: warmup, missing input
  and a confirmed absence of an event require different semantics.
* Internal WF is applied after the genomes have been optimized on the same
  selection frame; it does not rerun the complete search on each earlier
  fold's training prefix. The frame's normalizer is also fitted on the wider
  selection interval. Passing that diagnostic is therefore not independent
  evidence that the discovery procedure generalizes to later data.

`failure-mechanism.csv` separates loss of entry edge from excessive sizing
using each frozen trade's cost-charged R multiple (net PnL / initial stop
risk), rather than cash PnL dominated by changing account size. All 27 primary
replayed candidates had positive mean R in IS. In diagnostic OOS, 23 had
nonpositive mean R; the cross-candidate median fell from +0.2092R to -0.1811R.
Two more retained positive mean R but had negative mean log-balance growth
at the fixed 3/13 risk fraction. Thus most failures already occur in the
underlying trade payoffs, while two separately demonstrate sizing/volatility
drag. These are arithmetic diagnoses of recorded trades, not a new account
replay or selection rule. The ledger's 1e-9 risk-denominator floor is excluded
from per-risk calculations; 881 late OOS trades across the primary/auxiliary
set were excluded, and no usable OOS trade had a risk-cap/sizing mismatch.
For complete usable paths, log-growth accounting reconciles with the recorded
balance. All raw trade and monthly PnL sums also reconcile with reported net.

The corresponding repair must change the learning/evaluation experiment:
use several chronological training episodes with the same 180-day account
goal, fit preprocessing and feature selection only inside each permitted
training prefix, and assess the complete frozen discovery/selection procedure
on later outer periods. Test an uncertainty-aware objective and explicit
confirmed-event semantics separately, against the preserved baseline. Keep
all tuning within training/selection periods and retain the count of attempts;
the already-inspected tail cannot certify the repair. Do not require every
component to meet the full account target or every regime before a shared
portfolio evaluator exists. The current experiment has not implemented or
financially validated those repairs. The novelty-stagnation source fix remains
separate, because novelty was disabled throughout this matrix.

### Frozen-population training selection repair experiment

The next authorized experiment uses the existing saved populations from the
three `rank_survivors` seeds above. `canonical-research` now accepts the
research-only `--diagnose-training-episodes-out <new-json>` continuation. It
requires the exact saved research/hash and original settings, refuses other
diagnostic or trading modes, and preserves the existing source/financial
verification. It is Risky survival-v3 only; ordinary search and PropFirm
configuration are unchanged.

The experiment addresses post-GA candidate selection first, without claiming
to retrain genetic reproduction on additional episodes. Four non-overlapping
180-calendar-day episodes span the selection interval: first, latest and two
evenly spaced between. Every episode restarts the same account, uses the
saved genes and sealed exits/costs/risk, and precedes calibration/outer OOS.
Its last episode must reproduce each saved gene's trade count and net PnL.
All are training/selection episodes, with frozen selection-fitted normalization;
they are not independent walk-forward tests. Inactive episodes receive neutral
utility zero. Valid measured losses/goal pace receive clipped v6 utility in
[-1,1], preventing a single jackpot from dominating several ruined episodes.
Arithmetic faults fail closed rather than becoming a loss observation.

`%LOCALAPPDATA%/neoethos/experiments/risky-training-episodes-20261107/plan.json`
freezes both arms, binary/source/input identities and the comparison before new
results. Both arms use the same candidate pool with at least 20 latest-episode
trades, positive latest net and drawdown below total loss. Select four by latest
utility versus four by mean training-episode utility, retaining deterministic
identity ties. Positive aggregate eligibility is counted separately: diagnostic
top-four replay does not authorize a negative-utility candidate. All three
cohorts are frozen before any new outer replay. The reused guarded runner caps
the whole cycle at 30 minutes and each child at eight minutes, using four CPU
threads and the existing memory/disk guards. M15 alone, original per-trade risk,
costs, exits and 100-to-50000/180-day goal remain fixed.

Compare paired seed medians of OOS net R per trade, individual net/monthly
returns, positive-month rate, worst month, drawdown, losing streaks and full/80%
goal attainment. The predeclared directional check requires better median R
in at least two of three seeds and the pooled candidate median. This is a
small screening comparison, not a significance test. Shared candidates across
arms, related genomes, repeated future periods and overlapping goal windows
are not independent trials. No per-gene profits are summed into a portfolio.
The already-inspected tail remains diagnostic and cannot certify a repair.

CPU research uses the existing normalization projection to exclude columns with
no valid cell in the in-sample normalization interval. Both base and aligned
higher-timeframe columns use that same interval. Holdout values cannot rescue a
column. The retained schema, projection option and fitted state are bound into
each batch's content receipt. Raw indicator values and strict GPU recipes retain
their existing semantics. A discovery failure also preserves its full
error chain in `failure.json` beside the original run inputs.

The account snapshot and all five broker responses are hash-checked, checked
against the exact acquisition account, and frozen beside the run. Configured
account currency must match the observed deposit currency; mismatches are
refused before creating outputs or building features. This check proves the
observed account currency, not historical commission or swap policy.

All outputs remain `ResearchOnly` / `NotPromotionEligible`. Candidate selection
and completing an exploratory budget do not demonstrate net profitability or
exhaust the strategy space. D1-derived conversion and commission plus configured
spread/slippage are screening assumptions. Raw Bid/Ask archives need independent
verification and the existing reviewed quote-replay authority before they can
support financial validation or promotion. Chronological holdout evaluation
within this run does not establish that the data was never examined previously.

Use a CPU build for this command. CUDA builds refuse it; their device execution
continues through the existing sealed `native-research` route. No broker request,
order, model installation or promotion is performed by this command.

Method references: [cTrader historical data](https://help.ctrader.com/open-api/symbol-data/)
and [Statistical Overfitting and Backtest Performance](https://www.davidhbailey.com/dhbpapers/overfitting.pdf).

## Conditional specialist research on one account

The opt-in `canonical-research --specialist-portfolio true` continuation accepts
an exact saved research/SHA pair, including a zero-portfolio candidate archive.
It requires one batch and a new output directory and cannot be combined with
other diagnostics or quote replay. The additional output is
`research.json.specialists.json`; normal portfolio acceptance is unchanged.

`neoethos-trader::specialist_portfolio` fits a discrete routing table on the
recorded `InSample` scope. Four fixed contexts use completed UTC hours only:
up, down, range and high volatility. A missing base candle immediately clears
context. A route selects one expert before opposing votes can cancel, retaining
that expert's confidence and brackets. This is not a fitted news model or a
replacement for the existing ML ensemble. The experiment uses M15 inputs.

The bounded training pool samples up to sixteen archived candidates. One
fixed-seed crossover generation expands it; training semantics version 2 keeps
the original parent pool available when fewer than two routes were admitted.
Choosing cash must not silently disable genetic exploration. Every proposed
assignment records its training score, realized balance, trade count and entry
counts by context. Assigning an expert still requires twenty training trades
in that context and improvement over the current account score. Cash remains
available. This does not establish statistically significant edge.

The primary experiment caps the archived risk at 1%, explicitly recording both
the original and evaluated bands. The same selected genes are compared with
ordinary voting at identical costs and risk. A separate archived-risk diagnostic
replays the already frozen policy without refitting. Both routes use
`PreparedNettedCanonicalBarResearchV1`, one position and one account budget;
per-gene PnL is never added into a fictitious portfolio. The report preserves
closed-trade UTC cash months, terminal open-position marks and quantity-grid
status. A null grid means fractional research lots, not broker executability.

The later `SelectionValidation` and `Holdout` scopes never fit routes or genes.
They keep the recorded normalization, effective feature order and sealed exit
policy. Previously exposed historical tails remain diagnostic; this flag grants
no promotion authority. The November 11 bounded comparison, source snapshots,
frozen plans and all failed attempts are under
`%LOCALAPPDATA%/neoethos/experiments/specialist-shared-account-20261111/`.

The completed November 11 diagnostic compared three fixed archived cohorts.
The first source version admitted no experts and generated no children. After
the training-only bootstrap repair was declared, all three cohorts tested sixteen
parents and thirty crossover children. No score, context, cost or risk rule was
tuned from the subsequent account outcomes. `plan-v3.json` binds executable
`39ce469b54a1c0e774e6c1f7f3ffa00af52d9a0e3aacc1f2378b9b815b9b8ff0`;
`summary-v3.json` retains all twenty-four account comparisons. The original
30-minute deadline was retained; both comparisons finished in 1600.85 seconds
from the original start. Prior cache hashes were unchanged.

| Frozen seed | Admitted experts | Calibration, 1% cap | Final holdout, 1% cap |
| --- | ---: | --- | --- |
| 2026110701 | 0 | 0 trades, 0% | 0 trades, 0% |
| 2026110702 | 1 | 2 trades, +1.4695% | 0 trades, 0% |
| 2026110703 | 0 | 0 trades, 0% | 0 trades, 0% |

Training was January 2016 through July 2024, calibration July 2024 through
August 2025 and holdout August 2025 through September 2026. Each later interval
touches fourteen UTC calendar months. The only admitted specialist combines
`smc_fib_2000` and `neighboring_trailing_stop_bullish_band` and routes the high
volatility context. It traded on two of 283 calibration weekdays and none of
284 holdout weekdays. Its same-gene unconditioned voting baseline made three
calibration trades (+3.1468%) and one holdout trade (-1.1054%). At the archived
23.0769% risk the same baseline's holdout loss was -25.5093%; the specialist
still made no holdout entries. Neither risk interpretation hit the account goal.
The specialist's zero-profit cash months are not profitable months or a
financial validation pass.

The source now exposes training trial scores, realized balances and context
support. Positive in-sample PnL is not enough to infer transferable edge: for
example, the first cohort's best supported proposal ended at EUR111.25 from
EUR100 with only 41 trades across the full training interval and a negative
regularized score. The second admitted proposal had 159 training trades but
only two calibration trades and no holdout entries. Thus this experiment does
not establish monthly income, daily cadence, independent OOS profitability or
an accepted live portfolio. The next training change belongs in the existing
genetic/search owners: evaluate support and shared-account behavior across
chronological training episodes, rather than hoping a small post-search router
can recover a reliable income stream from sparse archived genes. Preserve the
already exposed future periods as diagnostic evidence.

Operator correction, November 12: ordinary EURUSD targets must be at most
20 **executed** pips. Targets of 25-30 pips are exceptional-day cases requiring
an explicitly defined and separately validated exception; the experimental
`hourly_volatility` label alone does not establish that permission. Targets of
70-80 pips are not an operator requirement. The bound concerns the final entry
bracket after any adaptive calculation, not just the displayed gene parameter.
Previous V1 artifacts retain their sealed limits. The joint V2 path described
below applies this cap to the actual entry bracket.

The completed specialist comparison was checked against its pinned sources and
archive hashes: all sixteen parents in each cohort had `stop_vol_mult=0`, all
archives had fixed targets no greater than 25 pips, and crossover inherited the
whole SL/TP pair from a parent. Thus its thirty children per cohort could not
inflate targets to 70-80 pips. The only selected child retained SL14.2075203113 /
TP24.9823302558 from its parent. These frozen results retain their original
25-pip search bound; they are not retroactively relabeled as a 20-pip experiment.

### Joint specialist books with normal targets at most 20 pips

`canonical-research --specialist-portfolio joint-v2` writes a separate
`research.json.joint-specialists.json`. The old `true` option keeps its V1
routing semantics and artifact suffix. Joint V2 can activate several experts in
one hourly state. Each expert retains its own position, entry quantity and
SL/TP while sharing one realized balance and account-equity sizing denominator.
Opposite sleeves remain separate modeled positions; this is not evidence that a
particular broker account permits hedging.

The existing `PreparedNettedCanonicalBarResearchV1` owns the simultaneous replay
through `evaluate_joint`. It reuses the canonical confidence sizer, protective
barrier ordering, closed-trade recording, carry and conversion-fee settlement.
All same-bar closes settle before the next entry batch. Entry requests share the
remaining aggregate stop-price-risk and gross-notional budgets proportionally;
an explicit lot grid rounds down. Stable expert IDs resolve position-cap ties.
New V2 research uses at most four open sleeves, 1% aggregate initial stop-price
risk at admission and a 30x reference gross-notional limit. Costs and gaps are
additional to stop-price risk. The ratios may rise on existing positions after
an equity loss; they are admission limits, not continuous broker-margin proof.

Training admits only parents whose stored target is at most 20 pips and
crossover preserves a parent's complete bracket. The account also rejects an
actual entry target above 20 after adaptive resolution. It does not clamp sealed
genes or silently allow 25-30 pips in the high-volatility state. This cap belongs
to the new joint path; it does not rewrite previous artifacts or change all live
application modes.

The bounded training protocol samples sixteen eligible archived parents, breeds
training children, and genetically selects the whole book's state activation and
risk shares (six generations of sixteen books). Four disjoint 180-day InSample
episodes score the combined, cost-charged account with cash as a candidate. Genes
need not individually make money in every state or fold. The full package is
frozen before opening calibration and holdout. These episodes are training
stability checks: archived parents were already discovered on the selection
interval, so this is not independent nested walk-forward validation or a full
replacement of upstream genetic discovery. News routing and the supervised ML
ensemble are not added by this discrete state-allocation model.

Reports retain attributed trades, every calendar month including cash months,
active entry days, losing streaks, sampled-close equity drawdown, pending-cost
open-position marks and a separate conservative adverse-bar stress bound.
Simultaneous adverse sleeve extrema are a bound rather than an observed OHLC
path. A nonpositive capital bound stops further replay without inventing a
liquidation or crediting later recovery. Realized cash and open marks stay
separate. The command remains `research_only=true`, `promotion_eligible=false`:
it does not run all challenge/promotion gates, place broker orders, attest real
GPU execution, or make a historical tail fresh again.

The new local experiment owns its pinned executable, source snapshot, fixed
protocol, logs and results under
`%LOCALAPPDATA%/neoethos/experiments/specialist-joint-20pip-20261112/`. It compares
all three predeclared cohorts with cash and a first-four equal-risk diagnostic;
that diagnostic is not selected or promoted using the later-window results.

Local execution on 2026-11-12 compiled the Windows CPU CLI and completed two of
the three planned cohorts. The related execution suite passed 94/94, the trader
suite 74/74, the CLI authority checks 15/15, and the final account suite 18/18
(including nine simultaneous-account oracles). The duplicate check against HEAD
reported zero new clones. This establishes compiled CPU research behavior, not
GPU, supervised-ensemble, desktop live-feed or broker execution integration.

| Archived cohort | Learned book | Equal-four calibration | Equal-four holdout | Holdout trades | Profitable holdout months | Peak holdout positions |
| --- | --- | --- | --- | --- | --- | --- |
| 2026110701 | Cash, no entries | -0.3782% | -1.5664% | 142 | 6/14 | 3 |
| 2026110702 | Cash, no entries | -6.0333% | -5.8448% | 102 | 4/14 | 1 |
| 2026110703 | Not executed | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable |

Each completed cohort evaluated seventy distinct training books. None strictly
beat cash under the declared regularized fitness; this does not assert that
every book's unpenalized net PnL was negative. Cash has zero profitable months.
The equal-four arms are predeclared functional diagnostics, not approved live
strategies. All reported money uses the sealed reference cost assumptions and
fractional research lots (`lot_grid=null`), with 1% as the maximum aggregate
admission risk, not the archived 23.0769% per-trade risk. The largest holdout
concurrency of three positions is actual shared-account replay behavior.

The third cohort never started. Available memory fell as the desktop application's
resident memory grew. The controller preserved the original 20 GiB process /
18 GiB admission envelope, tried separate smaller 10/14 and 8/12 GiB envelopes
only for the unlaunched cohort, and stopped waiting when available RAM reached
9.1 GiB. Model code, genes, seed, financial parameters and the original absolute
deadline were unchanged; no completed financial run was repeated. All three
controller attempts and STOP markers remain preserved. No worker was left
running and no profitability or three-cohort completion is claimed.

The canonical local result is `summary.json` in that experiment directory.
`verification.json` records the 2/3 completion, all resource attempts and an
independent reconstruction of all eight account traces from entry/exit events.
Every trace's position count, realized balance, monthly PnL, terminal pending-cost
mark and entry count reconciled. The executable, current pinned source and prior
cache hashes remained unchanged. These outcomes do not satisfy the profitable
monthly-income objective; a prospective frozen policy and quote/account evidence
are still required before any promotion claim.

### Continual specialist research V3

`canonical-research --specialist-portfolio continual-v3` writes
`research.json.continual-specialists.json`. It uses the existing specialist/joint
owner and shared CPU account, with a signature-distinct bank of normal-target
archived experts instead of the sixteen-parent sample. The bank can grow to 256;
at most four account positions remain open. This is a bounded research option.

Each monthly decision fits a combined book on older 180-day history and compares
cash, the incumbent and a challenger on a separate recent 60-day interval. A
`max_hold_bars + 1` embargo separates fit from validation and validation from the
decision bar. Fit subwindows come from the existing rolling-regime window owner.
The book's experts need not individually profit in every subwindow. Allocation
remains conditional on the four causally completed-hour market states.

The older fit can create up to two signature-distinct crossover descendants per
update. Parents come from the older fitted book, with archive exploration if
that book is cash. Recent validation cannot generate descendants or choose their
parents. Both the incumbent and the earlier fit champion seed the next search;
renewal cannot discard them merely by rerandomizing its initial population.
Observed entry directions and brackets identify exact behavioral twins within
each state; a book containing twins is rejected. Revision 2 represents a state
with no observed entries as `None`; two inactive specialists never count as
twins. Inactivity outside a specialist's condition cannot veto the whole book.
This is not complete correlation
or semantic deduplication of the entire archive.

New experts append to stable archive indices. Earlier schedules allocate zero
to experts that did not yet exist. The selected entry book stays frozen for the
next monthly interval. One replay applies all those books to a continuous shared
account within each forward interval; updates retain already-open quantities and
SL/TP rather than resetting cash or liquidating positions. A matched control
replays the first monthly book unchanged over the same interval. Calibration and
the later forward interval are separately initialized account experiments.

Training telemetry now retains each episode's realized money, pending-cost mark
and drawdown alongside regularized fitness. Realized PnL and conservative marks
known at the past cutoff are used; positive unclosed marks cannot count as
income. A nonpositive fitness does not imply every episode's raw PnL was negative.
Cash continues to mean zero income and zero profitable months.

This is an explicitly adaptive/prequential historical diagnostic: older parts
of a forward interval may become rolling training or recent-validation history
only after their decision-time cutoff. Those scopes are reclassified relative
to that decision and retain their exact input ranges. This is not a static,
untouched holdout or independent nested validation: the original archive and
historical tail have already been inspected. The complete renewal procedure and
its search budget, not just each final genome, require prospective validation.

The architecture draws on the idea of adding, retaining and reweighting experts
under changing conditions, rather than assuming one permanent learner; see
[Kolter and Maloof's Dynamic Weighted Majority](https://www.cs.cmu.edu/~zkolter/pubs/kolter-jmlr07.pdf).
Increasing the bank also increases selection opportunities; see
[Bailey et al. on backtest overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
Neither publication establishes an edge for this implementation or EURUSD.

Normal actual-entry targets remain at most 20 pips. Reference costs, fractional
research lots unless a checked grid is sealed, and `promotion_eligible=false`
remain explicit. This increment is archive-seeded crossover and joint-book
learning; upstream discovery gates, the supervised ML ensemble, live quote
execution and broker margin/stopout integration are outside this increment.

The current source passed 19 account tests, 84 trader tests and 15 CLI authority
tests and compiled as a Windows CPU executable. The fixed two-cohort experiment
owns its plan, executable, snapshots, logs and results under
`%LOCALAPPDATA%/neoethos/experiments/continual-specialists-20261116/`. Its primary
comparison is adaptive monthly renewal versus the same first monthly book held
fixed, with cash as a reference. It does not resume the stopped November 12
controllers or change their financial results.

The first V3 run exposed a new-filter regression: empty observed-state signatures
were treated as clones. The regression test was reproduced failing, then passed
with the `None` correction. The affected cohort 701 output and interrupted 702
remain preserved. A separately pinned `correctness-r1` verification rechecks 701
only with behavior revision 2; lookbacks, seeds, search budget, costs and risk
are unchanged and the original absolute campaign deadline is retained. Results
from the earlier filter must not be treated as current validation evidence.

The corrected revision-2 cohort completed before the original deadline. Its
bank grew from 80 to 136 signature-distinct candidates through 56 new offspring,
with 28 monthly decisions, 878 evaluated training books and two book changes.
Eight training books had positive regularized fitness. These are candidate and
search counts, not counts of proven profitable strategies.

| Revision-2 account | Calibration return | Later forward return | Calibration trades | Profitable calendar months |
| --- | --- | --- | --- | --- |
| Adaptive | -3.0539% | 0.0000% | 8 | 0/14 in each interval |
| Frozen first book | 0.0000% | 0.0000% | 0 | 0/14 in each interval |

The adaptive book activated on 2024-11-01 after **only two** recent-validation
trades produced +0.2911% reference-account PnL. The subsequent eight trades lost
3.0539%; sampled-close equity drawdown reached 3.4952%. It had five active entry
days in calibration and none in the later interval. This does not establish
profitable adaptation or near-daily income. The second corrected cohort was not
rerun within the original budget.

`correctness-r1/verification.json` independently reconciles all four account
traces, positions, cash and monthly PnL, and verifies that every entry belonged
to the book available at its signal-time decision. Executed source snapshots,
the pinned executable and prior cache hashes were verified. The old affected
run remains separate from these corrected results.

Following that diagnostic, current **protocol revision 3** adds the existing
`RegimeLabelPolicy` evidence floor (currently eight closed recent-validation
trades) before a challenger or incumbent can authorize new entries. The bank
retains inactive experts. This minimum is a basic sample screen, not statistical
significance, multiple-testing control or a profitability guarantee. The new
gate passed its regression test and the full 84-test trader suite and compiled;
it was **not financially rerun on the already-seen tail**. The measured table
above belongs to revision 2. Prospective data and evaluation of the complete
adaptive procedure are still required.

### Risky goal fidelity (2026-11-16)

The operator confirmed that the primary objective is **EUR100 -> EUR50,000 in
180 days**, using the existing ten-year history, with ordinary targets <=20
pips. The V1/V2/V3 specialist experiments above used an explicit 1% research
overlay and monthly fitness. They are diagnostics, not faithful evaluations of
that Risky objective. The subsequent `continual-r3-verification-20261116` job was
stopped with its own STOP before completion when this mismatch was identified;
its partial logs are preserved and are not financial results.

Use the separately versioned `--specialist-portfolio risky-v4` for that goal.
It requires a sealed Risky growth recipe, a valid configured goal/deadline and
actual simulation capital equal to the goal's starting capital. The original
sealed risk band is retained. Normalized specialist allocations share one
aggregate stop-risk budget equal to that band's ceiling, rather than each
expert receiving another copy. The 30x gross-notional constraint matches the
existing live sizing constant; this does not assume 400x or broker margin rules.
Every report/log declares mode, currency, goal, deadline and both risk bands.

V4 reuses the existing canonical `ga_fitness_goal` pace formula on the lesser of
realized balance and pending-cost net equity, with cash at zero progress. Its
training aggregation remains mean minus 0.25 episode-score standard deviation.
There is no monthly-profit or arbitrary drawdown penalty in this Risky objective;
capital exhaustion is still refused. Short fit/validation intervals provide a
pace **proxy**, not evidence of reaching the goal. The eight-trade recent
validation evidence floor, causal scheduling, embargo and clone rejection remain.

Each predeclared calibration/forward interval is limited to its first 180 days
and starts one independent EUR100 account. Monthly updates occur inside that
account without resetting its cash. The shared ledger stops processing prices
after the calendar deadline. A coverage-only timestamp after the deadline may
prove history coverage; its prices never settle or open a trade. Target success
requires a flat account with realized cash >=50,000 before or at the deadline;
a crossing while another sleeve remains open is recorded separately and cannot
count as banked success. Existing sleeves keep protection, pending positions are
reported without fabricated liquidation, and insufficient horizon coverage has
its own status. The complete procedure still uses an already-inspected tail and
reference OHLC costs; these are chronological diagnostics, not independent OOS,
broker execution, GPU evidence or promotion authorization.
