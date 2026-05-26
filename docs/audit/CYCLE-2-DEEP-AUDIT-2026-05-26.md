# NeoEthos Cycle-2 Deep Audit — 2026-05-26

Evidence-based systemic audit across cTrader API correctness, backend/UI
schema alignment, math/race conditions. Triggered by operator feedback
that previous fix cycles were treating symptoms, not root causes.

**Sources consulted:**
- `crates/neoethos-app/proto/OpenApiMessages.proto`
- `crates/neoethos-app/proto/OpenApiModelMessages.proto`
- Spotware Open API docs: https://help.ctrader.com/open-api/messages/
- Proto field comments inline (more authoritative than the docs site)

---

## A. cTrader Open API correctness

### A.1 Relative SL/TP magnitude — 10× too wide on 5-digit FX, 1000× too narrow on JPY & gold

- **Severity:** CRITICAL (real-money risk in either direction)
- **File:** `crates/neoethos-app/src/app_services/broker_api.rs:479-496`
- **Current code:**
  ```rust
  let pip_relative_units: f64 = if digits >= 4 {
      10f64.powi((digits - 4) as i32 + 1)
  } else { 1.0 };
  let relative_stop_loss = stop_loss_pips.map(|p| (p * pip_relative_units).round() as i64);
  ```
- **What the docs say:** `OpenApiMessages.proto:102` — `relativeStopLoss = Specified in 1/100000 of unit of a price (e.g. 123000 in protocol means 1.23)`.
- **What's wrong:**
  - For 5-digit FX (`digits=5`), correct multiplier is `pip_size × 100_000 = 0.0001 × 100_000 = 10`. The code computes `10^(5-4+1) = 100` → **SL 10× wider** than user typed. A 20-pip SL on EURUSD ships `relativeStopLoss=2000` which the broker interprets as 200 pips (0.0200).
  - For 3-digit JPY (`digits=3`) and 2-digit XAU (`digits=2`), the `else` branch returns `1.0`. Correct multipliers: 1000 (JPY: 0.01 × 100_000) and 1000 (XAU: 0.01 × 100_000) — **1000× too narrow**. A 20-pip XAU SL ships `20` which broker reads as 0.00020 price distance ≈ zero, SL fires immediately.
  - The inline comment block at lines 484-487 actually describes the *correct* formula. **The code disagrees with its own comment.**
- **Right formula:** `pip_relative_units = pip_size × 100_000`, sourced from `neoethos_core::symbol_metadata::resolve(symbol).pip_size`. Equivalently `10^(5 - pipPosition)` where `pipPosition` is the proto field.
- **Effort:** ~30 min (replace 4-line block + add tests for EURUSD/USDJPY/XAUUSD/EURJPY/GBPUSD).

### A.2 Risk-gate `estimated_loss` overestimated by `contract_size` (≈100,000× on FX)

- **Severity:** CRITICAL (gate blocks every order with SL, or — if env-tuned to compensate — passes oversized orders)
- **File:** `crates/neoethos-app/src/app_services/trading/risk_gate.rs:293-294`
- **Current code:**
  ```rust
  // cTrader volume is in cents of a standard lot, so divide.
  let estimated_loss = pip_distance * (order.volume as f64 / 100.0) * pip_value_per_lot;
  ```
- **What the docs say:** `OpenApiMessages.proto:89` — `volume = represented in 0.01 of a unit (e.g. 1000 in protocol means 10.00 units)`. So `order.volume / 100 = base-currency units`, **not lots**. To get lots: `order.volume / lot_size_in_cents` where `lot_size_in_cents` is `ProtoOASymbol.lotSize` (`OpenApiModelMessages.proto:144`, "Lot size of the Symbol (in cents)") = 10_000_000 for EURUSD.
- **What's wrong:** For 1 lot EURUSD, `order.volume = 10_000_000`. Code computes `10_000_000 / 100 = 100_000` and multiplies by `pip_value_per_lot ≈ 10 USD/lot` → `estimated_loss = pip_distance × 1_000_000`. A 20-pip SL gives $20,000,000 of "estimated loss" on a 1-lot trade whose real risk is $200. The test at `trading_tests.rs:1317-1318` even encodes the confusion ("volume = 100000 represents 1000 standard lots") — actually 100,000 wire = **0.01 lot**, not 1000.
- **Right formula:** `estimated_loss = pip_distance × (order.volume as f64 / lot_size_in_cents) × pip_value_per_lot`. Equivalently `pip_distance × lots × pip_value_per_lot`.
- **Effort:** ~1 h (plumb `lot_size` into the gate, fix the conversion, update 4-5 tests).

### A.3 `compute_pnl_pips` passes base-currency units where the function wants lots

- **Severity:** HIGH (PnL pip count off by `contract_size` ≈ 100,000× whenever live-tick override is unavailable — weekends, exotic symbols, anything not in `DEFAULT_STREAMED_SYMBOLS`)
- **File:** `crates/neoethos-app/src/server/bridge.rs:627` calls `compute_pnl_pips` at `bridge.rs:710`
- **Current code:**
  ```rust
  let mut pnl_pips = compute_pnl_pips(resolved_name.as_deref(), pnl_usd, p.volume);
  // ...
  fn compute_pnl_pips(resolved_name: Option<&str>, pnl_account_ccy: f64, volume_lots: f64) -> f64 {
      let denom = meta.pip_value_quote * volume_lots;
      pnl_account_ccy / denom
  }
  ```
- **What's wrong:** `p.volume` is set at `ctrader_account.rs:450,469` via `volume_to_units(position.trade_data.volume) = wire / 100.0` → **base-currency units** (1000.0 for 0.01-lot EURUSD), not lots (0.01). Param is named `volume_lots` and `meta.pip_value_quote = 10` USD/lot, so denominator is `10 × 1000 = 10_000` instead of `10 × 0.01 = 0.1`. PnL pips off by factor `contract_size`.
- **Effort:** ~10 min (divide by `meta.contract_size` at call site, or pass `volume_lots = p.volume / meta.contract_size`).

### A.4 Risk-gate `pip_distance` ignores live `pipPosition` from broker

- **Severity:** Medium
- **File:** `risk_gate.rs:243` — `let pip_multiplier = 10.0_f64.powi(pip_position);`
- **What's wrong:** Caller passes `pip_position` as hardcoded literal (`4` or `2` in tests). The broker exposes the canonical value via `ProtoOASymbol.pipPosition` (`OpenApiModelMessages.proto:117`). When broker re-tunes a symbol, literal goes stale silently.
- **Effort:** ~45 min.

### A.5 Close-position race: snapshot `volume` stale vs server

- **Severity:** Medium
- **File:** `bridge.rs:603` → `orders.rs:104-107`
- **What's wrong:** UI sends `volume = position.volumeUnits` captured up to ~5 s ago. cTrader's `ProtoOAClosePositionReq.volume` is `required` (no "0 = full close" sentinel). Vulnerable to broker-side volume changes between snapshot and request.
- **Mitigation:** Re-read via `ProtoOAReconcileReq` immediately before close, or accept partial close failures and retry with broker's error-payload volume.
- **Effort:** ~2 h.

### A.6 Live spot streamer subscribed to 8-symbol whitelist only

- **Severity:** Medium (only majors get the < 2 s pip refresh; gold, indices, exotics show broken pip count from A.3)
- **File:** `crates/neoethos-app/src/app_services/live_spots_streamer.rs:61-63`
- **What's wrong:** Hard-coded `DEFAULT_STREAMED_SYMBOLS` of 8 majors. When operator opens XAUUSD position, the bridge live override at `bridge.rs:640-677` doesn't fire because there's no tick. Should subscribe on demand from account-runtime poll.
- **Effort:** ~3 h.

### A.7 `volume_to_units` name conflates "wire/100" with "lots"

- **Severity:** Low (no behaviour bug; this is the function that misled A.2/A.3 authors)
- **File:** `crates/neoethos-app/src/app_services/ctrader_account.rs:885`
- **Fix:** Rename to `volume_wire_to_base_units` + comment explaining relationship to `lotSize`.
- **Effort:** ~20 min.

### A.8 Flutter docstrings mislabel volume as "centi-lot"

- **Severity:** Cosmetic
- **Files:** `backend_client.dart:490-491`, `orders.rs:104-105`
- **What's wrong:** There is no such cTrader unit. The value is wire-format cents of base currency. Replace with proto's wording.
- **Effort:** ~5 min.

### A.9 Daily-drawdown gate never resets on broker-day rollover

- **Severity:** HIGH (prop-firm compliance: a session across midnight reports wrong "daily" DD)
- **File:** `crates/neoethos-app/src/app_services/trading/session.rs:526, 565-566`
- **Current:** `self.day_start_equity = Some(self.day_start_equity.unwrap_or(runtime.trader.balance));`
- **What's wrong:** Set at connect time, then `unwrap_or` retains prior value forever. Risk gate at `risk_gate.rs:150-160` divides `(day_start_equity - account_equity) / day_start_equity`. After 3 trading days connected, "daily" DD becomes 3-day P&L. **FTMO/MyForexFunds/TFT explicitly require server-time midnight reset.**
- **Effort:** ~2 h (track `current_broker_day_id`; reset to current equity when it changes; same pattern for total DD using configured `initial_equity` only).

### A.10 Live-tick freshness window equals bridge refresh interval

- **Severity:** Low
- **File:** `bridge.rs:644-646` — `if freshness_ms <= 5_000`
- **Fix:** Tighten to ≤ 2_000 ms. The `STALE_THRESHOLD` const introduced in #148 should be used here.
- **Effort:** ~15 min.

---

## B. Backend/UI schema gaps

### B.1 AdvancedSettings save silently no-ops every knob outside 5-id whitelist

- **Severity:** CRITICAL (every UI change shows "Settings saved." but only 5 known IDs persist)
- **File:** `experiments/forex-flutter-ui/lib/screens/advanced_settings_screen.dart:347-365`
- **Current:**
  ```dart
  await ref.read(backendClientProvider).saveSettings(
        dataDir: edits['system.data_dir']?.toString(),
        newsCalendarEnabled: edits['news.calendar_enabled'] is bool ? ... : null,
        newsCalendarSource: edits['news.calendar_source']?.toString(),
        openaiModel: edits['ai.openai_model']?.toString(),
        newsTradingMode: edits['news.trading_mode']?.toString(),
      );
  // unknown knob ids are skipped silently; snackbar fires unconditionally
  ```
- **Root cause:** Backend's `POST /settings` is a typed DTO; Flutter knob catalog uses string IDs. No marshaller bridges them.
- **Right fix:** Either route through a `POST /settings/raw` (already exists), or add a generic `POST /settings/knob` accepting `{id, value}` pairs. Whatever the path, NEVER show success toast if some IDs were dropped.
- **Effort:** ~6 h.

### B.2 `Position.pnl_usd` field name misleading on non-USD accounts

- **Severity:** Low (number is correct, label wrong)
- **Files:** `account.rs:79`, `backend_client.dart:34, 43, 53`
- **What's wrong:** `netUnrealizedPnL` is in **deposit currency**, not USD. Owner's GBP account → "pnlUsd" is actually GBP.
- **Fix:** Rename to `pnl_account_ccy` on both sides; UI renders with existing `currency` field.
- **Effort:** ~45 min.

### B.3 `compute_pnl_pips` early-return zeroes pip count instead of returning None

- **Severity:** Medium
- **File:** `bridge.rs:719-722`
- **What's wrong:** `let Some(meta) = neoethos_core::symbol_metadata::resolve(name) else { return 0.0; };` — symbol not in table → silent 0. Live-tick path falls back to heuristic; broker-derived path falls to 0. Inconsistent.
- **Effort:** ~30 min.

### B.4 News config dead-code orphans (`poll_llm_news_sentiment` + 8 NewsConfig fields)

- **Severity:** Low
- **File:** `crates/neoethos-core/src/domain/news_filter.rs:117`
- **What's wrong:** Function only referenced by its own docs. Zero call sites. Several `NewsConfig` fields ride along. Settings UI may surface knobs that do nothing.
- **Effort:** ~2 h.

### B.5 No type-level distinction between Lots, BaseUnits, WireCents

- **Severity:** Cosmetic (design quality)
- **Fix:** Newtype wrappers `Lots(f64)`, `BaseUnits(f64)`, `WireCents(i64)` would make A.2/A.3 hard to commit.
- **Effort:** ~3 h.

### B.6 `/positions/close` ClosePositionBody docstring lies about unit

- **Severity:** Cosmetic (same as A.8).

### B.7 Live tick override never updates `pnl_usd`, only `pnl_pips`

- **Severity:** Low (documented limitation)
- **File:** `bridge.rs:631-639`
- **Effort:** ~2 h (compute live USD using `meta.pip_value_in_account` × pip count).

---

## C. Math, race & API drift

### C.1 `Position.volume` raw base-currency units exposed but unused by Flutter

- **Severity:** Cosmetic
- **Files:** `bridge.rs:689`, `backend_client.dart:27, 51`
- **Fix:** Drop from response, or document loudly.
- **Effort:** ~10 min.

### C.2 `lot_size.unwrap_or(10_000_000)` silent fallback masks missing broker metadata

- **Severity:** Medium
- **File:** `broker_api.rs:454`
- **What's wrong:** Fallback fine for standard FX. For indices, commodities (XAU `contract_size = 100`), crypto, 10M is grossly wrong. Should `bail!` instead. `risk_gate.rs::ctrader_protocol_volume_from_lots` already does this — inconsistent.
- **Effort:** ~15 min.

### C.3 (covered by A.6)

### C.4 OAuth expiration not enforced before placing trades

- **Severity:** Medium (Suspected, needs verification)
- **File:** `broker_api.rs:419, 542, 567`
- **What's wrong:** `resolve_creds()` returns whatever's in `secure_store`. If access token expired 30 s before click, order hits `CH_ACCESS_TOKEN_EXPIRED`. Should pre-emptively refresh if `expires_at - now < 60 s`.
- **Effort:** ~2 h.

### C.5 `STALE_THRESHOLD` constant exists but not used in bridge live-override gate

- **Severity:** Low
- **File:** `bridge.rs:646` literal `5_000`
- **Effort:** ~5 min.

### C.6 `freshness_ms` uses local-machine clock vs local receive time — naming misleading

- **Severity:** Cosmetic

### C.7 Account snapshot stale-window after repeated refresh failures (already mitigated #106)

- **Severity:** Low (flagged as potential refinement)

---

## Already-fixed (confirmed correct after today's commits)

- Volume placement (`broker_api.rs:455`): `volume_lots × lot_size_in_cents` — correct.
- Close-position volume scaling (`bridge.rs:603`): `p.volume × 100` — correct.
- `TRADING_BAD_VOLUME` error translation (`ctrader_errors.rs:117`) — distinct from `MARKET_CLOSED`.
- Currency hardcoded EUR (#144) — done.

## Not bugs (spec-conformant)

- `scale_price` for spot ticks (`live_spots_streamer.rs:501-505`): proto says all prices are 1/100000, so `raw/100_000` is right for all symbols including XAUUSD and JPY. `digits` param only used to round display.
- `scale_ctrader_money_int` (`ctrader_money.rs:76`): correctly handles per-entity `moneyDigits`.

---

## Summary

**3 Critical (in order):**

1. **A.1** — SL/TP magnitude formula off by 10× on 5-digit FX and 1000× on JPY/XAU. Money-critical. SL placed 10× further from entry than operator set, eroding prop-firm risk-per-trade discipline.
2. **A.2** — risk-gate `estimated_loss` overstated by `contract_size` (≈100,000× on FX). Gate rejects every trade with SL on FX, OR if disabled/env-tuned around, every trade sizes too large.
3. **B.1** — AdvancedSettings save silently no-ops everything outside 5-id whitelist while displaying success toast.

**2 High:**

- **A.3** — pip count off by `contract_size` whenever live-tick override is unavailable (weekends, non-major symbols).
- **A.9** — daily-drawdown gate never resets at broker-day rollover → prop-firm compliance failure on multi-day sessions.

**Total findings: 21.** Numbered by section + severity; line-referenced for direct action.

## Confidence

- **A.1, A.2, A.3, B.1, A.9** — High confidence, verified against proto field comments and current code.
- **A.4-A.10, B.2-B.7, C.1-C.3, C.5, C.6** — High confidence, mechanical code-read.
- **C.4** — Suspected, needs verification by reading the OAuth refresh code.
- **C.7** — Flagged as potential refinement; not a confirmed bug.
