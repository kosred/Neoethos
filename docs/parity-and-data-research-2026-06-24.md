# Research before the restructure (2026-06-24)

Sources: our code (`crates/neoethos-app`, `neoethos-core`, `neoethos-trader`)
+ cTrader Open API official docs (ctx7 `/websites/help_ctrader_open-api`).
Three groups, as requested.

---

## GROUP 1 — Parity: app ↔ cTrader API

**Question:** are the numbers we show the broker's own values, or our (possibly wrong) transforms?

### Authoritative — straight from the cTrader API, correctly scaled ✓
cTrader sends monetary values as integers with a `moneyDigits` exponent
(real = raw / 10^moneyDigits). We apply this (`scaled_money`). Confirmed against docs.
- **Balance** ← `ProtoOATrader.balance` (moneyDigits) ✓
- **Unrealized P/L ($)** ← `ProtoOAGetPositionUnrealizedPnLRes.netUnrealizedPnL` (moneyDigits) ✓
  — we DO call the official PnL endpoint per refresh, not a guess.
- **Used margin** ← `position.usedMargin` ✓
- **Entry price / SL / TP** ← `ProtoOAPosition` ✓
- **Symbol name** ← broker symbol catalog ✓

### Computed by us (standard accounting, but it's OURS not the API's)
- `equity = balance + unrealizedPnL` (cTrader defines it the same way)
- `free_margin = equity − used_margin`
- `pnl_pips` — cTrader gives only the **money** PnL; we derive pips
  (`account_pnl_to_pips`, overridden by live mid). Label it as derived.

### ⚠️ The ONE real divergence — VOLUME display
- cTrader wire `ProtoOATradeData.volume`, `stepVolume`, `lotSize` are all **in cents**.
- **Lots = volume_cents / lotSize_cents.** Docs confirm `lotSize` is "in cents".
- We show `volume_to_units(wire) = wire/100` = **units** (e.g. 117000), and the
  close needs wire cents (11 700 000 = 100×). cTrader's UI shows **1.17 lots**.
- **Root cause of "117000 vs 1.17"**: we never pull `ProtoOASymbol.lotSize`, so we
  can't show lots. **Fix (parity): server provides `volumeLots = volume_cents /
  lotSize` + the symbol's lotSize/stepVolume; UI shows lots like cTrader.**

**Verdict:** money is parity-correct (right endpoints + moneyDigits). Fix volume→lots
(needs lotSize from `ProtoOASymbol`), and label the few derived fields. Rule going
forward: **show the API's value; only transform when the API itself doesn't provide it,
and then mirror cTrader's own formula.**

---

## GROUP 2 — Data & file transparency (where everything lives)

Today NONE of these are visible/openable from the UI. Every artifact:

| What | Location (default) | Source |
|---|---|---|
| Engine config | `config.yaml` (app CWD) | editable in Advanced ✓ |
| Broker creds | `%AppData%\Roaming\neoethos\broker_credentials.toml` | OS user dir |
| OAuth token | secure store / keyring (+ legacy file fallback) | secret |
| Market data (Vortex) | `data/` (`system.data_dir`) | server download / import |
| Trained models | `models/` (`models.models_dir`) | Training |
| Discovered strategies | `cache/…live_portfolio.json`, `model_targets.json`, `cache/auto_loop/` | Discovery |
| Search cache | `cache/search`, general `cache/` | engine |
| Trade journal | `<data_dir>/journal/` (JSONL) | live/journal |
| Logs | `<data_dir()>/neoethos/logs` (or `LOG_DIR`) | runtime |
| AI (Codex) auth | `~/.codex/auth.json` | secret |

**Need:** a **Files / Storage** screen that lists each path with: resolved absolute
path, exists?, size, last-modified, item count, and an **"Open folder"** button
(Tauri `shell`/`opener`). For data downloads + user-imported data, show exactly where
it landed and let the user re-open/verify. (`/data/bootstrap` already returns
`dataDir` + symbols + fileCount + lastTouched — surface it as the anchor.)

---

## GROUP 3 — Autotrading / inference with EXISTING models + strategies

- The live engine (`neoethos-trader`) trades/replays from a **portfolio artifact**
  (`…live_portfolio.json`, real genes) + per-symbol `model_targets.json`; blends
  gene signal + ML (`blend_signal.rs`). `/autonomous/replay` = dry-run on history,
  `/autonomous/start` = live loop. Strategy Lab `promote` copies a validated set into
  `models/…/live_models`.
- **Gap:** no clear "pick an existing strategy/model → autotrade/infer" flow with
  provenance. The Autonomous screen starts by symbol/base only; it doesn't let you
  choose WHICH saved portfolio/model to run or show where those files are.
- **Need:** an **Autopilot/Inference** section that (a) lists available portfolios +
  trained model sets (from `cache/*live_portfolio.json` + `models/`), (b) shows each
  one's file path + symbols + metrics, (c) one click → replay (dry-run) or start live,
  (d) shows which artifact is currently driving the live loop.

---

## Proposed grouped restructure (for approval — NOT yet built)

Sidebar regrouped so each concern is one obvious place:
1. **Trade** — Dashboard · Markets · Market Watch · Positions · Account
2. **Autopilot** — Autonomous/Inference (pick existing portfolio+models, provenance) · Risk
3. **Research** — Discovery · Training · Strategy Lab · Intelligence
4. **Data & Files** — storage map (every path, size, open-folder) · Data download/import
5. **Desk** — Journal · News · AI Desk
6. **System** — Hardware · Advanced · Settings

Backend work this implies:
- Expose `lotSize`/`stepVolume` per symbol + `volumeLots` on positions (parity).
- A `storage/paths` endpoint (resolved paths + size/mtime/count) + an open-folder command.
- A `portfolios/list` (+ model sets) endpoint for the Autopilot picker.
- Surface the last 5 hidden endpoints (actions queue, indicators, chart/history,
  settings/presets, broker/timeframes).
