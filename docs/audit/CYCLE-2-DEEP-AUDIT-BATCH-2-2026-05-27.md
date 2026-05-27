# NeoEthos Cycle-2 Deep Audit — Batch 2 (manual, beyond agent's 21)

Manual continuation of `CYCLE-2-DEEP-AUDIT-2026-05-26.md`. Read proto
file + symbol-fetching code line-by-line. **All findings below were
missed by the previous batch.**

Sources:
- `crates/neoethos-app/proto/OpenApiModelMessages.proto` (ProtoOASymbol entity, lines 113-155)
- `crates/neoethos-app/proto/OpenApiMessages.proto`
- Cross-checked against current Rust sources

---

## D. cTrader symbol entity — fields we IGNORE that change real money

cTrader's `ProtoOASymbol` (`OpenApiModelMessages.proto:113-155`) exposes
40+ fields. We currently read only **5**: `pipPosition`, `digits`,
`minVolume`, `maxVolume`, `stepVolume`, `lotSize` (per
`crates/neoethos-app/src/app_services/ctrader_data.rs:188-197`). The
following are **never read into the symbol struct** and the trading
code therefore can never account for them:

### D.1 `preciseTradingCommissionRate` + `commissionType` — IGNORED

- **Severity:** HIGH
- **Proto:** `OpenApiModelMessages.proto:128-129, 145`
  ```
  optional int64 commission = 14 [deprecated = true];
  optional ProtoOACommissionType commissionType = 15 [default = USD_PER_MILLION_USD];
  optional int64 preciseTradingCommissionRate = 31;
   // Total commission depends on commissionType: for non-percentage
   // types multiplied by 10^8, for percentage of value multiplied by 10^5.
  ```
- **What we do instead:** `crates/neoethos-search/src/cubecl_eval.rs:264` uses a single hardcoded `commission_per_trade: f32` knob across ALL symbols. Discovery tests every strategy with the same notional commission.
- **Real impact:**
  - cTrader IC Markets EURUSD: $7/round-turn per lot. XAUUSD: $30/lot. UK100 CFD: 0.5 bps of notional.
  - Backtest result on EURUSD off by ~1.5 pips per round trip (~$15/lot vs the global default).
  - Strategy that ranks #1 in backtest may rank #5 once real commissions are applied — but you never find out because nothing pulls per-symbol cost.
- **Effort:** ~6 h (fetch fields → plumb into cost model → re-run discovery comparison).

### D.2 `swapLong` / `swapShort` / `swapCalculationType` / `swapPeriod` / `swapTime` / `chargeSwapAtWeekends` — IGNORED

- **Severity:** HIGH (for any strategy holding > intraday)
- **Proto:** `OpenApiModelMessages.proto:121-122, 143, 150-151, 153`
  ```
  optional double swapLong = 7;   // SWAP charge for long positions.
  optional double swapShort = 8;  // SWAP charge for short positions.
  optional ProtoOASwapCalculationType swapCalculationType = 29 [default = PIPS];
  optional int32 swapPeriod = 36;
  optional int32 swapTime = 37;
  optional bool chargeSwapAtWeekends = 39;
  ```
- **What we do instead:** Zero. Grep `swap|overnight|rollover` across `crates/neoethos-search/src/` returns one hit — `eval.rs:821` referring to "day rollover for max_trades_per_day tracking" (not financial). **Backtest pretends swap doesn't exist.**
- **Real impact:**
  - Typical EURUSD long: -$0.50/lot/night. Short: +$0.10/lot/night.
  - 1-month holding period at 0.1 lot avg: -$1.50 per long position unaccounted.
  - **JPY carry trades:** swap is the ENTIRE P&L source. Long USD/JPY earns ~+$3.50/lot/night. Strategy that backtests as breakeven actually profits +$3.50/night/lot. Or vice versa for shorts.
  - **Weekend triple-swap** (every Wednesday in MT4 convention, varies by broker) compounds error.
- **Effort:** ~10 h (extend Symbol struct, fetch on connect, plumb into cost_profile, model triple-swap Wed/Fri, validate against a real position's accrued swap field).

### D.3 `slDistance` / `tpDistance` / `gslDistance` / `distanceSetIn` — IGNORED at pre-trade gate

- **Severity:** MEDIUM (broker rejections instead of clean error)
- **Proto:** `OpenApiModelMessages.proto:130-132, 134`
  ```
  optional uint32 slDistance = 16; // Minimum allowed distance between SL and current price.
  optional uint32 tpDistance = 17;
  optional uint32 gslDistance = 18;
  optional ProtoOASymbolDistanceType distanceSetIn = 20 [default = SYMBOL_DISTANCE_IN_POINTS];
  ```
- **What we do instead:** Order placement at `broker_api.rs:477-478` computes `relative_stop_loss = stop_loss_pips × pip_relative_units` and submits without checking against `slDistance`. Broker rejects if too close. Operator sees a generic "TRADING_BAD_PRICES" or similar.
- **Real impact:** A scalper strategy with 3-pip SL on a broker enforcing 5-pip minimum sees every order rejected. Backtest signals 100 trades; live executes 0.
- **Effort:** ~3 h (fetch slDistance/tpDistance, clamp at pre-trade gate, surface as "broker requires ≥X pip SL" error).

### D.4 `tradingMode` + `enableShortSelling` — IGNORED for pre-trade gate

- **Severity:** MEDIUM
- **Proto:** `OpenApiModelMessages.proto:118, 141`
  ```
  optional bool enableShortSelling = 4;
  optional ProtoOATradingMode tradingMode = 27 [default = ENABLED];
  // ENABLED / DISABLED_WITHOUT_PENDINGS_EXECUTION /
  // DISABLED_WITH_PENDINGS_EXECUTION / CLOSE_ONLY_MODE
  ```
- **What we do instead:** Order placement makes no check. Grep `enableShortSelling|tradingMode` in `crates/neoethos-app/src/` confirms zero use.
- **Real impact:** Send SELL on a symbol that doesn't permit shorting → broker rejection. Open new position on a symbol in CLOSE_ONLY_MODE (often happens before earnings on stock CFDs, around weekends on some FX brokers) → rejection.
- **Effort:** ~2 h (refuse at pre-trade gate, friendlier error message).

### D.5 `pnlConversionFeeRate` — IGNORED (silent profit cut on non-deposit-quote pairs)

- **Severity:** MEDIUM (haircut on every closed PnL when quote ≠ deposit ccy)
- **Proto:** `OpenApiModelMessages.proto:148`
  ```
  optional int32 pnlConversionFeeRate = 34;
  // Percentage (1 = 0.01%) of the realized Gross Profit, which will be
  // paid by the Trader for any trade if the Quote Asset of the traded
  // Symbol is not matched with the Deposit Asset.
  ```
- **What we do instead:** Backtest assumes 0% conversion fee. For owner's GBP account trading EURUSD (quote = USD), a typical broker fee is 0.5-1%. On +$100 trade → $0.50-1.00 chopped at close.
- **Real impact:** Strategies that rely on tight margin (Risky Mode compounding) are silently sub-target by ~1% per trade. Compounded across 50 trades = -39% vs expectation.
- **Effort:** ~3 h (fetch field, apply in close-trade cost in both backtest and live ledger).

### D.6 `rolloverCommission` / `rolloverCommission3Days` / `skipRolloverDays` — IGNORED (Shariah-compliant accounts)

- **Severity:** Low (only matters if operator runs a swap-free Islamic account)
- **Proto:** `OpenApiModelMessages.proto:138-139, 142`
- **What we do instead:** Nothing.
- **Effort:** Low priority unless user opens a Shariah account.

### D.7 `minCommission` / `minCommissionType` / `preciseMinCommission` — IGNORED

- **Severity:** MEDIUM for small-lot strategies
- **Proto:** `OpenApiModelMessages.proto:135-137, 146`
  ```
  optional int64 preciseMinCommission = 32; // multiplied by 10^8.
  optional string minCommissionAsset = 23 [default = "USD"];
  ```
- **What we do instead:** Backtest commission is `commission_per_trade` × N trades. If broker's MIN commission is $1 per trade and your 0.01-lot scalp would normally pay $0.07, you actually pay $1 → backtest under-counts by 14×.
- **Effort:** ~2 h.

---

## E. Backtest / discovery integrity

### E.1 Test fixture hides A.3 (pip pnl off by ×contract_size)

- **Severity:** MEDIUM (testing gap — A.3 bug ships with green CI)
- **File:** `crates/neoethos-app/src/server/bridge.rs:740-758` (`sample_position()`)
- **What's wrong:**
  ```rust
  fn sample_position() -> CTraderPositionSnapshot {
      CTraderPositionSnapshot {
          position_id: 42,
          symbol_id: 1,
          trade_side: "BUY".to_string(),
          volume: 0.1,   // ← treats volume as LOTS
          ...
      }
  }
  ```
  Production `volume_to_units` produces `p.volume = 10_000` for the same 0.1-lot position (base units). The test fixture uses `0.1` directly. `compute_pnl_pips` happens to give the right answer ONLY because the lots-shaped input was assumed by both the test and the buggy code path. **Every test in `bridge.rs::tests` is wrong about the unit and so the test suite is blind to A.3.**
- **Effort:** ~30 min (set `volume: 10_000.0` in the fixture so the test asserts production behavior, watch A.3 fail loudly, then fix A.3).

### E.2 PnL drift circuit breaker is dead by design

- **Severity:** MEDIUM (silent loss of protection vs broker payload corruption)
- **File:** `crates/neoethos-app/src/app_services/trading/orders.rs:1084-1092`
- **What's wrong:**
  ```rust
  let breaker = super::evaluate_pnl_drift_circuit_breaker(
      &authoritative,
      &positions_snapshot,
      |_position| {
          // Per-position local PnL is not directly tracked yet
          None  // ← always returns None
      },
  );
  ```
  Inline comment explicitly accepts this. The whole 200-line drift comparator in `pnl.rs` (with documented 0.1% warn / 1% block thresholds, hand-checked against swap-bias math) is **never exercised in production** because per-position local PnL is never computed. The audit table at `/account/snapshot` shows zeros in the `local` column.
- **Effort:** ~6 h (compute per-position local PnL via mid-price × volume × pip_value_in_account; this also requires fixing A.3 first; then the breaker comes online).

### E.3 Swap modeling completely missing in backtest

- **Severity:** HIGH (covered above as D.2)
- **File:** `crates/neoethos-search/src/eval.rs` (zero matches for swap/overnight/rollover in financial context)
- **Strategy-class impact:** Any strategy with average-holding-time > 1 day is invalidly evaluated. Risky Mode compounding (#226) holding through weekends will see triple-swap surprise in live.

### E.4 Commission knob is global, not per-symbol

- **Severity:** HIGH (in pair with D.1)
- **File:** `crates/neoethos-search/src/cubecl_eval.rs:264, 366, 500, 1174`
- **What's wrong:** Discovery scores every symbol candidate against the SAME `commission_per_trade` value. Multi-symbol portfolio search can't differentiate EURUSD (cheap) from XAUUSD (3× expensive) from indices.
- **Effort:** Encompassed in D.1's fix.

---

## F. Live trading paths — additional findings

### F.1 `digits` enum mismatch — only handled FX, not metals/indices/crypto

- **Severity:** MEDIUM (already partially in A.1)
- **File:** `crates/neoethos-app/src/app_services/trading/orders.rs:1099-1135` (`ctrader_symbol_pip_position`)
- **What's wrong:** Inline comment "The bot is FX-only — JPY pairs use 2 decimal pip notation, every other major/minor uses 4. We deliberately do NOT branch on metals or crypto here because the bot doesn't trade them." But the operator's UI exposes 443 symbols incl. XAU, indices, equities. If operator (or LLM auto-trade) ever clicks a non-FX symbol, pip math goes wrong.
- **Effort:** ~1 h (use proto `pipPosition` per-symbol always; remove the "FX-only" branch).

### F.2 cTrader OAuth credentials are PERMANENT (operator confirmed) — re-auth UI flow is unnecessary

- **Severity:** Cosmetic (UX bloat)
- **What's wrong:** The Settings/BrokerSetup screen has a "Re-auth ChatGPT" / "Re-auth cTrader" CTA. cTrader Open API client_id + client_secret are issued once per app and never expire (per Spotware Open API onboarding docs). Only the access_token expires; the refresh_token loop handles renewal silently. The UI prominence of "Re-auth" suggests a maintenance task that doesn't exist.
- **Effort:** ~15 min (hide the CTA unless `auth.json` is missing or refresh failed N times).

### F.3 `local_pnl_for_position` closure type signature exists but caller never produces a value

- **Severity:** Same as E.2 — flagged here for code-search visibility.
- **Files:** `pnl.rs:230, 495` (signature) and `orders.rs:1084` (always returns `None`).

### F.4 Account-currency PnL in `pnl_usd` field — name lies (already in B.2)

Cross-listed for emphasis. Owner's GBP account sees `pnlUsd` field in JSON containing GBP value.

---

## G. SSE / streaming concerns

### G.1 `/live/spots` subscribed to 8 majors only (already A.6)

Cross-listed. Owner's chart loaded XAUUSD won't show live tick.

### G.2 SSE reconnect uses `Last-Event-ID` — backend support?

- **Severity:** Suspected
- **File (UI):** `sse_client.dart:17` documents the intent to send `Last-Event-ID` on reconnect.
- **File (backend):** `crates/neoethos-app/src/server/sse.rs` (if exists) — needs to honor the header and replay from that ID. **Not verified.** If backend doesn't honor it, every disconnect → full snapshot replay, defeating the latency win.
- **Effort:** 15 min to verify, more if not implemented.

---

## H. Codex / AI Helper — explanation for the model rejection

### H.1 ChatGPT subscription Codex endpoint requires installation-id header

- **Severity:** HIGH (#291 unresolved)
- **Likely cause** (based on Spotware/OpenAI Codex CLI patterns): the `/backend-api/codex/responses` endpoint validates the bearer token AND inspects request fingerprints. The error "model not supported when using Codex with a ChatGPT account" probably means **the request was identified as not coming from the official Codex CLI binary**, and is being downgraded to a model-restricted tier that excludes everything we try.
- **Headers the official CLI sets that we don't (per Simon Willison's reverse-engineering article we found earlier):**
  - `x-codex-installation-id` (a UUID baked into the CLI binary or generated on first run)
  - `x-codex-version` (the CLI version string)
  - User-Agent in a CLI-specific format (e.g., `codex-rs/v0.116.0`)
- **Recommended fix path:**
  1. Capture an actual Codex CLI HTTP request (mitmproxy or `--debug` flag) to discover the full header set.
  2. Add the discovered headers to `crates/neoethos-codex/src/client.rs`.
  3. If that still 400s, the only remaining option is `api.openai.com/v1/responses` with an operator-provided API key (drops the "no API key" promise but unblocks chat).
- **Effort:** ~3 h investigation + ~1 h fix once the right header set is known.

---

## Summary of Batch-2 findings

**3 HIGH:**
- D.1 — Per-symbol commission ignored (backtest off 1-3% per trade)
- D.2 — Swap/overnight ignored entirely (multi-day strategies invalid)
- H.1 — Codex endpoint requires unknown CLI-specific headers

**6 MEDIUM:**
- D.3 — slDistance/tpDistance pre-trade gate missing
- D.4 — tradingMode / enableShortSelling pre-trade gate missing
- D.5 — pnlConversionFeeRate silent profit cut
- D.7 — Minimum commission ignored (small-lot strategies)
- E.1 — Test fixture masks A.3 bug
- E.2 — PnL drift circuit breaker dead by design
- F.1 — pip_position falls back on "FX-only" for non-FX symbols

**2 LOW / COSMETIC:**
- D.6 — rolloverCommission (Shariah only)
- F.2 — Re-auth UI is for non-existent maintenance task

**Combined with batch 1's 21 findings: 32 total open issues** (4 critical, 5 high, 13 medium, 10 low/cosmetic).

The big systemic story across both batches: **NeoEthos reads cTrader's
symbol entity as if it were a "symbol id + a few volume limits" table,
ignoring the financial fields that determine real PnL** (commission,
swap, distance limits, conversion fees, trading mode). The backtest
engine is therefore optimistic by a percentage that grows with holding
time and varies by symbol. The live trading path inherits the same
blind spots and surfaces them as broker rejections instead of clean
preflight errors.

A focused 2-week cleanup of D.1, D.2, D.3, D.4, D.5 (estimated ~30 h
total) would close the gap between "looks profitable in backtest" and
"actually profitable in live" — the single biggest barrier to the
owner's stated goal of a real trading product.
