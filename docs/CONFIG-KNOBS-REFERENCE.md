# NeoEthos Configuration Knobs — Operator Reference

**Audience:** the operator (you), and the Flutter Settings screen that surfaces these knobs at runtime.

**Status:** authoritative as of 2026-05-25. Updated alongside any new
`*RuntimeOverrides` struct or env-var registration.

This document catalogs EVERY runtime knob the bot honours, grouped by
subsystem. Each entry lists:

- **Name** (legacy env-var name and/or the typed-overrides field path)
- **Type + range** (parser shape and what values are accepted)
- **Default** (what the bot does when the operator doesn't override)
- **Effect** (what changes in observable bot behaviour)
- **Risk profile** (Conservative / Balanced / Aggressive recommendation)

The Flutter "Advanced Settings" screen reads this catalog (via
`GET /settings/knob-catalog`) and renders each knob with the
descriptions below as tooltips. Presets (Conservative / Balanced /
Aggressive) ship as one-click switches — see the **Presets** section
at the bottom.

> **Doctrine note (2026-05-25):** env vars are read ONCE at startup
> into typed `*RuntimeOverrides` structs (see `runtime_overrides.rs`
> in each crate, and `neoethos_core::env_overrides`). Hot-path code
> reads the typed value via `current_*_runtime_overrides()`. The
> Flutter UI writes to `config.yaml` + tells the operator to restart
> for the new value to take effect.

---

## Section 1 — Broker connectivity (cTrader)

### `FOREX_BOT_CTRADER_READ_TIMEOUT_SECS`
- **Type:** `u64`, `[0, 3600]` (0 disables timeout)
- **Default:** `30`
- **Effect:** caps the underlying TCP read inside `execute_via_session`.
  Without this cap, a broker stall could wedge the trading loop
  indefinitely. With it, the I/O error bubbles up, the session is
  dropped, and the next attempt re-authenticates.
- **Conservative:** `30` (default — safe in unstable networks)
- **Balanced:** `30`
- **Aggressive (low-latency colo):** `15` (faster failover)
- **When to lower:** if you're on a fast VPS near the cTrader DC and
  you'd rather fail fast.
- **When to raise:** if you see intermittent timeouts on a slow
  consumer connection.

### `FOREX_BOT_CTRADER_MAX_ATTEMPTS`
- **Type:** `u32`, clamped `[1, 5]`
- **Default:** `3`
- **Effect:** maximum attempts (initial + retries) for a single
  `execute_via_session` call. Retry safety relies on the broker
  deduping by `clientOrderId`.
- **Conservative:** `2` (fewer retries → safer for the prop-firm gate)
- **Balanced:** `3`
- **Aggressive:** `5` (more retries; only if you trust the broker's dedup)

### `FOREX_BOT_CTRADER_BACKOFF_BASE_MS`
- **Type:** `u64`, clamped `[10, 2000]`
- **Default:** `200`
- **Effect:** base backoff in ms for cTrader retries. Actual delay
  doubles per attempt with 0-99ms jitter, capped at 5s total.
- **Conservative:** `500` (slower retries; gentler on the broker)
- **Balanced:** `200`
- **Aggressive:** `100`

### `FOREX_BOT_CTRADER_ALLOW_PARTIAL_FILL`
- **Type:** `bool` (`1` / `true` / `yes` → on; anything else → off)
- **Default:** `false`
- **Effect:** when `false`, partial fills are rejected and bubble up
  as an error. When `true`, they're accepted as final.
- **Conservative:** `false` (consistent risk-per-trade math)
- **Balanced:** `false`
- **Aggressive:** `true` (accept what you can get on illiquid pairs)

### `FOREX_BOT_CTRADER_STREAM_MAX_ATTEMPTS`
- **Type:** `u32`, clamped `[1, 5]`
- **Default:** `3`
- **Effect:** maximum attempts for `load_live_chart_update`
  (streaming chart updates). Each call is a stateless poll.
- **Recommendation:** keep at default unless you see explicit
  streaming retry errors in the log.

### `FOREX_BOT_CTRADER_STREAM_BACKOFF_BASE_MS`
- **Type:** `u64`, clamped `[10, 2000]`
- **Default:** `200`
- **Effect:** same as `FOREX_BOT_CTRADER_BACKOFF_BASE_MS` but for the
  streaming layer.

### `FOREX_BOT_CHART_MERGE_SIDE`
- **Type:** enum `mid` | `bid` | `ask` (case-insensitive)
- **Default:** `mid`
- **Effect:** chooses which side of the spread the chart-merge layer
  uses when a single price is needed (e.g. for the latest-close
  display). `mid` is the broker-standard convention.
- **Conservative/Balanced/Aggressive:** all `mid` — only override if
  you're modelling worst-case slippage explicitly.

### `CTRADER_TRANSPORT` (test/dev only)
- **Type:** `wss` (Open API JSON over WebSocket Secure)
- **Default:** `wss`
- **Effect:** transport selection. Leave at default — only the
  WebSocket transport is exercised in production. Useful for
  testing harness only.

---

## Section 2 — Risk & PnL safety

### `FOREX_BOT_PROP_ACCOUNT_CURRENCY`
- **Type:** ISO-4217 currency code (`USD`, `EUR`, `GBP`, `JPY`, etc.)
- **Default:** **unset → hard-fail at the risk gate.** No synthetic
  default per real-data directive (2026-05-24).
- **Effect:** account currency for the risk-per-trade math. The
  pip-value computation in `risk_gate` requires this to convert from
  symbol-quote pip value to account currency. With a misconfigured
  account currency, the gate would size positions incorrectly (e.g.
  treating GBP account as USD overstates risk by ~20%).
- **Conservative/Balanced/Aggressive:** match your broker account.
  Operator-supplied via Settings → Broker Setup once at account
  configuration time; never auto-defaulted.

### `FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE`
- **Type:** `f64`, finite + `> 0.0`
- **Default:** `None` (cross-pair pip-value math fails without it)
- **Effect:** live quote→account FX rate override for cross pairs.
  Used when the broker hasn't shipped a real rate yet (initial spin-up
  of a fresh session). Once the broker is feeding live spots, this
  override is irrelevant.
- **Recommendation:** set only for offline backtests / paper trading
  where no live broker is connected.

### `FOREX_BOT_PNL_AUDIT_DRIFT_FRACTION`
- **Type:** `f64`, clamped `[1e-5, 0.05]`
- **Default:** `0.001` (0.1 %)
- **Effect:** drift threshold for the PnL audit log. When broker-side
  unrealized PnL diverges from local mark-to-market by more than this
  fraction of notional, a warning is logged. Helps detect bad
  pip-value mappings before they cost money.
- **Conservative:** `0.0005` (5bp — alert on small drift)
- **Balanced:** `0.001` (10bp — default)
- **Aggressive:** `0.005` (50bp — only loud drift)

### `FOREX_BOT_PNL_CIRCUIT_BREAKER_FRACTION`
- **Type:** `f64`, clamped `[1e-4, 0.20]`
- **Default:** `0.01` (1 %)
- **Effect:** circuit-breaker threshold. When drift exceeds this
  fraction of notional, the auto-trader halts pending operator review.
  Upper bound is 20% so the breaker cannot be silenced by a typo.
- **Conservative:** `0.005` (50bp — halt on small drift)
- **Balanced:** `0.01`
- **Aggressive:** `0.05` (5%) — only halt on major drift; not
  recommended for prop-firm runs.

### `NEOETHOS_PROP_FIRM_PRESET`
- **Type:** enum `ftmo` | `myforexfunds` | `fundednext` | `the5ers` | `none`
- **Default:** `ftmo`
- **Effect:** which prop-firm preset seeds `RiskConfig::default()`.
  The preset sets daily-loss / max-drawdown / profit-target /
  min-trading-days defaults. Operator can still override individual
  fields in `config.yaml`.
- **All profiles:** pick the preset that matches your funded account.

### Risk-per-trade fraction (from prop-firm preset or `RiskConfig`)
- **Type:** `f64`, `[0.0, 0.05]` (0–5 %)
- **Default:** preset-derived (FTMO: 0.01 = 1 %)
- **Effect:** maximum loss per single trade, as a fraction of equity.
  The risk gate rejects orders that would risk more than this.
- **Conservative:** `0.005` (50bp/trade)
- **Balanced:** `0.01` (100bp/trade — default)
- **Aggressive:** `0.02` (200bp/trade — risky)

### Risky Mode `risk_per_trade_fraction` (the compounding mode)
- **Type:** `f64`, recommended Kelly-aligned ≤ `0.30` (30 %)
- **Default:** `0.30` (lowered from `0.40` 2026-05-25 per Kelly half-criterion)
- **Effect:** in Risky Mode (€100 → kα-thousands), the larger
  per-trade risk that drives compounding. Comes paired with an
  operator-signed §6.4 acknowledgement that the ruin probability can
  reach ~99% (§7.1).
- **Conservative:** STAY IN PROP-FIRM MODE; do not enable Risky.
- **Balanced (if you've signed the §6.4 ack):** `0.20`
- **Aggressive (you understand the math):** `0.30`–`0.50`

---

## Section 3 — Discovery / GA search

### `FOREX_BOT_SEARCH_SEED`
- **Type:** `u64` (any non-negative integer)
- **Default:** `None` → non-deterministic (OS RNG)
- **Effect:** RNG seed for the genetic search. Setting any value
  makes the run deterministic (same population evolution every time).
- **Recommendation:** set during validation runs to compare changes.
  Leave unset for production search.

### `FOREX_BOT_NOVELTY_WEIGHT`
- **Type:** `f64`, finite
- **Default:** `0.0` (novelty disabled)
- **Effect:** weight applied to the novelty bonus during candidate
  ranking. `0.0` keeps ranking purely on fitness. `> 0.0` favours
  diverse genes even at slightly lower fitness — useful when the GA
  gets stuck in a local optimum.
- **Conservative:** `0.0`
- **Balanced:** `0.05`
- **Aggressive:** `0.15`

### `FOREX_BOT_PROP_STAGNATION_GENS`
- **Type:** `usize`, `>= 1`
- **Default:** `2`
- **Effect:** number of stagnant generations the search tolerates
  before triggering early-stop or gate-relaxation.
- **Conservative:** `1` (stop early; save compute)
- **Balanced:** `2`
- **Aggressive:** `5` (let the search push through stagnation)

### `FOREX_BOT_PROP_TOURNAMENT_SIZE`
- **Type:** `usize`, `>= 2`
- **Default:** `max(population / 12, 3)`
- **Effect:** tournament size for selection. Larger tournaments →
  stronger selection pressure → faster convergence but less diversity.
- **Recommendation:** leave at default unless you know your search
  is converging too slowly (raise) or losing diversity (lower).

### `FOREX_BOT_PROP_ARCHIVE_CAP`
- **Type:** `usize`, clamped `[population, 200_000]`
- **Default:** derived as `min(population × generations, 50_000)`
- **Effect:** maximum genes stored in the archive. Larger archive →
  more candidates for the final picker but more RAM.
- **Recommendation:** default unless RAM-constrained or running a
  very-long deep search.

### `FOREX_BOT_PROP_SMC_GATE_START` / `_END` / `_CURVE` / `_STAGNATION_STEP`
- **Type:** `f32`, finite
- **Defaults:** start `0.75`, end `0.35`, curve `1.0`, stagnation_step `0.03`
- **Effect:** the SMC-gate threshold curve. Starts at `start`, eases
  to `end` along a power curve of exponent `curve`, and relaxes by
  `stagnation_step` per stagnant generation once patience is exceeded.
  Higher thresholds → only strong SMC confluence passes; lower
  thresholds → more signals pass but with weaker confluence.
- **Conservative:** `start: 0.85, end: 0.45, curve: 1.5` (very strict)
- **Balanced:** defaults
- **Aggressive:** `start: 0.65, end: 0.25, curve: 0.7` (permissive)

### `FOREX_BOT_DISABLE_SMC_GATE`
- **Type:** `bool` (`1` / `true` / `TRUE`)
- **Default:** `false`
- **Effect:** hard-bypass for the SMC gate. When `true`, the gate
  collapses (active SMC sum forced to 0) so raw signals pass through.
  Useful for isolating "SMC indicators don't trigger on this symbol"
  from "signal generation is broken".
- **Conservative/Balanced/Aggressive:** all `false` for production.
  Diagnostic-only toggle.

### `FOREX_BOT_PROP_ARCHIVE_MODE` / `_MIN_NET` / `_MIN_PF` / `_MIN_SHARPE`
- **Type:** mode = `net` | `pf` | `sharpe`; floors = `f64`
- **Defaults:** mode `net`, min_net `0.0`, min_pf `1.0`, min_sharpe `0.0`
- **Effect:** archive admission criteria. Strategies must clear the
  selected mode's floor to be archived for the final pick.
- **Conservative:** `mode: net, min_net: 500.0, min_pf: 1.5, min_sharpe: 0.5`
- **Balanced:** defaults
- **Aggressive:** `mode: pf, min_pf: 1.1` (more candidates archived)

### `FOREX_BOT_PROP_PARENT_SELECTION` / `_SURVIVOR_SELECTION`
- **Type:** enum `rank_weighted` | `tournament` | `truncation`
- **Defaults:** both `rank_weighted`
- **Effect:** how the GA picks parents (for crossover) and survivors
  (for the next generation). `rank_weighted` is the default-stable
  choice; `tournament` is faster on large populations; `truncation`
  is most aggressive (deterministic top-K).
- **Recommendation:** keep at default unless you know what you're doing.

### `FOREX_BOT_PROP_RANDOM_IMMIGRANTS` / `_SURVIVOR_FRACTION` / `_SELECTION_TEMPERATURE`
- **Defaults:** immigrants `0.25`, survivors `0.10`, temperature `0.75`
- **Effect:** GA population-management knobs.
  - Immigrants: fraction of each generation replaced with random new genes (diversity injection)
  - Survivor fraction: elite carry-over to next generation
  - Selection temperature: softer (`< 0.5`) = more random; sharper (`> 1.0`) = more deterministic
- **Conservative:** `immigrants: 0.10, survivors: 0.20, temperature: 0.5`
- **Balanced:** defaults
- **Aggressive:** `immigrants: 0.40, survivors: 0.05, temperature: 1.5`

### `FOREX_BOT_PREFILTER_TOP_K` / `_INSAMPLE`
- **Type:** `usize`
- **Effect:** stage-1 pre-filter — only the top-K candidates by
  in-sample score advance to stage-2.
- **Recommendation:** keep at default unless deep-tuning.

### `FOREX_BOT_FUNNEL_STAGE1_PCT` / `_WINDOW`
- **Type:** `f64` / `usize`
- **Effect:** funnel scoring — what fraction of the in-sample window
  is used for stage-1, and the window size in bars.

### `FOREX_BOT_MIN_HISTORY_YEARS`
- **Type:** `usize` (years)
- **Default:** depends on `DiscoveryRuntimeOverrides`
- **Effect:** minimum historical-data requirement before a symbol is
  eligible for discovery. Prevents the GA from optimizing against
  too-short windows.
- **Conservative:** `5`
- **Balanced:** `3`
- **Aggressive:** `2`

### `FOREX_BOT_DISCOVERY_MODE`
- **Type:** enum (operator's discovery-mode set)
- **Effect:** which discovery-mode preset governs filtering.
  Permissive vs. strict vs. operator-tuned.

---

## Section 4 — Cost model / pip-value

### `FOREX_BOT_PROP_PIP_VALUE`
- **Type:** `f64`, positive + finite
- **Default:** `None` → use the broker-supplied metadata
- **Effect:** override the per-symbol pip value (account currency
  per standard lot per pip). Only used when the operator explicitly
  pins it. Otherwise pip value comes from `symbol_metadata.json`
  which is sourced from cTrader `ProtoOASymbol` records.
- **Recommendation:** leave unset; let the broker metadata drive.

### `FOREX_BOT_PROP_PIP_VALUE_PER_LOT`
- Same as above but expressed per-lot rather than per-pip-per-lot.

### `FOREX_BOT_PROP_SPREAD_PIPS`
- **Type:** `f64`, non-negative + finite
- **Default:** broker-quoted
- **Effect:** override the spread (in pips) used during backtest
  cost calculation. Useful for stress-testing strategies against
  wider spreads than the broker currently shows.
- **Conservative:** `+ 0.5` over broker-quoted (stress-test buffer)
- **Balanced:** broker-quoted (None)
- **Aggressive:** `0.0` (zero-friction; for theoretical comparisons)

### `FOREX_BOT_PROP_COMMISSION`
- **Type:** `f64`, non-negative + finite
- **Default:** broker-quoted (typically `$3-7 / round-trip / standard lot`)
- **Effect:** override commission per trade.
- **Recommendation:** stress-test with the worst-case commission your
  broker quotes for your account class.

### `FOREX_BOT_REJECT_PIP_FALLBACK`
- **Type:** `bool`
- **Default:** `false`
- **Effect:** when `true`, the cross-pair pip-value fallback
  `bail!()`s instead of silently returning the quote-currency pip
  value. Recommended on prop-firm runs so you fail loudly if a cross
  pair lacks an FX rate.
- **Conservative:** `true` (fail loudly)
- **Balanced:** `false` (default — tolerant)
- **Aggressive:** `false`

---

## Section 5 — Quality / acceptance filtering

### `FOREX_BOT_PROP_MIN_TRADES_PER_MONTH`
- **Type:** `usize`
- **Default:** `4`
- **Effect:** strategies with fewer trades per month than this are
  rejected as low-frequency / undersampled.
- **Conservative:** `8` (only well-sampled strategies pass)
- **Balanced:** `4`
- **Aggressive:** `2`

### `FOREX_BOT_TRADING_DAYS_PER_MONTH`
- **Type:** `f64`, `>= 1.0`
- **Default:** `21.0`
- **Effect:** trading days per month used in the normalization math.
  Forex trades ~22 days/month; the default is conservatively rounded.
- **Recommendation:** leave at default; matters only for cross-strategy
  comparison normalization.

---

## Section 6 — Backtest runtime

### `FOREX_BOT_BACKTEST_INITIAL_EQUITY`
- **Type:** `f64`, positive + finite
- **Default:** `100_000.0`
- **Effect:** starting equity for the backtest simulation.
  Independent of any live account.

### `FOREX_BOT_BACKTEST_MAX_MONTH_BUCKETS`
- **Type:** `usize`
- **Default:** `240` (20 years)
- **Effect:** maximum month buckets the backtest will track for
  by-month statistics. Caps RAM on very-long history runs.

### `FOREX_BOT_RUST_THREADS` / `RAYON_NUM_THREADS`
- **Type:** `usize`
- **Default:** num CPU cores
- **Effect:** Rayon worker thread count. Lower if you want to keep
  CPU available for other tasks; raise (up to num CPUs) for fastest
  search.
- **Conservative:** `cpu_cores - 2` (leave 2 cores free)
- **Balanced:** all cores
- **Aggressive:** all cores + lower OS process priority

### `FOREX_BOT_SEARCH_EVAL_PRECISION` / `FOREX_BOT_TRAIN_PRECISION`
- **Type:** enum `fp32` | `bf16` | `fp16` | `bf4` | `fp8`
- **Default:** `fp32`
- **Effect:** numeric precision for evaluation. Lower precision is
  faster on GPUs but introduces rounding noise. `fp32` is the
  safe default.
- **Recommendation:** keep at `fp32` unless you have a GPU with
  significant bf16/fp16 acceleration AND you've validated that
  rounding noise doesn't break your discovery results.

### CUDA-specific knobs (when `--features gpu`)
- `FOREX_BOT_SEARCH_EVAL_CUDA_KERNEL` — `0/1` to disable/enable GPU eval kernel
- `FOREX_BOT_SEARCH_BACKTEST_CUDA_KERNEL` — same for backtest kernel
- `FOREX_BOT_SEARCH_EVAL_KERNEL_UNITS` — units-per-cube override
- `FOREX_BOT_SEARCH_EVAL_CUDA_DEVICE` — device index (default 0)

---

## Section 7 — Logging / persistence

### `RUST_LOG`
- **Type:** tracing-subscriber filter (e.g. `info,sqlx=warn`)
- **Default:** production filter from `Settings`
- **Effect:** controls log verbosity. Lower to `error,neoethos=info`
  for quieter logs; raise to `debug` for diagnostics.

### `NEOETHOS_USER_DATA_DIR`
- **Type:** absolute path
- **Default:** `dirs::data_local_dir()` (Windows: `%LOCALAPPDATA%`)
- **Effect:** override the user-data root (logs + state).

### `FOREX_BOT_SYMBOL_METADATA`
- **Type:** absolute file path
- **Default:** `data/symbol_metadata.json` (auto-populated from cTrader)
- **Effect:** override the symbol-metadata source file. Useful for
  offline backtests with a frozen symbol set.

### `FOREX_BOT_LIVE_JOURNAL_PATH`
- **Type:** absolute path
- **Default:** under user-data-dir
- **Effect:** override the live trading journal location.

### `NEOETHOS_PENDING_ACTIONS_PATH` / `NEOETHOS_RISKY_MODE_STATE_PATH`
- **Type:** absolute paths
- **Default:** under user-data-dir
- **Effect:** test/CI overrides for the persistence files.

---

## Section 8 — Server / network

### CLI `--config <path>` flag (preferred) / hard fallback `./config.yaml`
- **Effect:** which `config.yaml` the bot loads at startup. Threaded
  through the `AppApiState::config_path()` typed boundary so every
  route handler honours it.

### `NEOETHOS_SERVER_BIND`
- **Type:** `host:port` (e.g. `127.0.0.1:7423`)
- **Default:** `127.0.0.1:7423`
- **Effect:** override the HTTP server bind address. Useful when
  running the backend on a different machine than the Flutter
  front-end.

---

## Presets

The Flutter Settings screen offers three one-click presets. Each
preset writes the corresponding values to `config.yaml`; the operator
can fine-tune individual fields afterwards.

### Conservative (default for new users)
- Risk-per-trade: `0.5 %` (`risk_per_trade: 0.005`)
- Prop-firm preset: FTMO
- Discovery: `min_history_years: 5`, archive_min_net: `500`
- SMC gate: strict (`start: 0.85, end: 0.45`)
- PnL circuit breaker: `0.5 %` drift
- cTrader retries: `2` (fewer retries)
- Risky Mode: **disabled**
- Suitable for: prop-firm passing, capital preservation, beginners.

### Balanced (production recommended)
- Risk-per-trade: `1 %`
- Prop-firm preset: matches your account
- Discovery: defaults
- SMC gate: defaults (`0.75 → 0.35`)
- PnL circuit breaker: `1 %` drift
- cTrader retries: `3`
- Risky Mode: **disabled**
- Suitable for: funded accounts, multi-month campaigns.

### Aggressive (advanced users only)
- Risk-per-trade: `2 %`
- Prop-firm preset: matches (or "none" if Risky Mode)
- Discovery: `min_history_years: 2`, permissive archive criteria
- SMC gate: relaxed (`0.65 → 0.25`)
- PnL circuit breaker: `5 %` drift
- cTrader retries: `5`
- Risky Mode: **available** (requires signed §6.4 acknowledgement)
- Suitable for: operators who understand Kelly mathematics, accept
  the 99 % ruin probability ceiling, and have a separate
  prop-firm-passing account running the Conservative preset.

---

## Reading this doc from the UI

The Flutter Settings screen renders this catalog by:

1. Calling `GET /settings/knob-catalog` to fetch the JSON
   representation (field name, type, range, default, current value,
   help-text excerpt).
2. Rendering each knob in a collapsible card with the help text as
   a tooltip / expandable info box.
3. Offering preset switches at the top (Conservative / Balanced /
   Aggressive / Custom).
4. Writing changes to `config.yaml` via `POST /settings/knobs` and
   prompting the operator to restart for the new value to take
   effect.

**Backward compatibility:** every knob continues to be settable via
its legacy `FOREX_BOT_*` env var. The Settings screen and env vars
both write into the same typed `*RuntimeOverrides` struct, so they
are interchangeable at the read site.
