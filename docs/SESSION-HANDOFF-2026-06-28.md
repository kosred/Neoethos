# NeoEthos — Session Handoff (2026-06-28)

Single source of truth to continue WITHOUT losing the thread after a context refresh.
Read this first. Companion memory: `settings-consolidation-directive`, `tauri-migration-2026-06-22`.

---

## UPDATE (cont. 2026-06-29)
- **Settings consolidation cont.:** News gate + Data location sections + Help "Search tuning" section (commits f0ef9bc4, 4bf7deb2 earlier). Installer rebuilt OK (17:53, 37.2MB) after a disk-pressure build failure.
- **DISK-LEAK FOUND + FIXED (commit 24b7142f):** discovery mmaps the multi-TF cube to `%TEMP%/neoethos_feature_store/<sym>_<tf>_<PID>.fstore` — **M3 cube ≈ 12 GB each**. `delete_on_drop` cleans it on GRACEFUL exit, but FORCE-KILLING the run (we did, to swap engine builds) skips Drop → leak; accumulated ~29 GB. Reclaimed 28 GB (deleted the temp dir). Fix = startup sweep of orphan `.fstore` (Windows refuses deletion of live-mapped files → safe). **CLI rebuilt 18:11 with the fix. OPERATIONAL RULE: stop discovery via `cache/risky_stop.flag` (graceful), NOT force-kill.**
- **Discovery GA-tuned run RESULT (proof):** EURUSD alone produced **35 validated strategies** (H1=6, M30=8, M15=12, M5=9) vs **1** pre-tuning. Stopped at EURUSD M3 (the 12 GB unit) for disk. Strategies in `cache/auto_loop/`. Note: M3 base = ~12 GB transient disk while running (cleaned after, with the fix).
- **Latest builds:** installer `NeoEthos_0.5.0_x64-setup.exe` (17:53, has News/Data/Help + disk fix) — REINSTALL. CLI `target/release/neoethos-cli.exe` (18:11, leak-free).

## 0) IMMEDIATE GOAL
- **Today:** finish the in-flight UI/config changes (commit + build) so they're not lost.
- **Tomorrow:** put EXISTING discovered strategies into operation (Autopilot → demo → live) and stress-test the **Auto-Trading UI + backend** for bugs. (Tomorrow is a weekday → forex OPEN → live spots/candles/pips will actually flow.)

---

## 1) DONE THIS SESSION (committed, verified)
- **DLL permanent fix** — `tauri.conf.json` bundle.resources GLOBS `../../target/release/*.dll` → every native DLL auto-ships (catboost/xgboost/app_lib). Installer 37.2MB. (commit b25cfb7f)
- **GA anti-stagnation — VALIDATED 6× yield.** prefilter_top_k 50→120 (config) + stronger indicator-mutation (evolution_math.rs). EURUSD H1 went 1→**6** exported strategies; profitable genes 6→19; best gene PF 1.73→3.19. (commits ~GA-tuning, 6047fe78)
- **Risky walk-forward FILTER** — export the WF-passing gene subset instead of all-or-nothing reject (was 0 exports). (commit 302ac9f1)
- **WF criterion mode-aware** — risky = positive avg OOS + ≥60% folds; prop_firm/strict = full prop-firm rules. (commit 0d968bf6)
- **Settings consolidation slices #1/#2/#3/#5** (commits 6047fe78, 4bf7deb2 + backend): Settings now has **Discovery mode**, **Risky goal**, **Search tuning** (anti-stagnation knobs, friendly + described, writes config.yaml), **Compute** (auto/cpu/gpu), **Risk & sizing** (preset + limits); **Discovery screen shows the active-mode badge**. `/settings` DTO expanded (settings.rs).
- **#35 lockbox FOUNDATION** — `FeatureFrame::row_slice` (commit a0b5267e). Rest pending (see §3).
- **Data** — 13 pairs, 11yr M1 + higher TFs (no H2/H3, cTrader lacks them). Broker costs refreshed (92 symbols). Old strategies backed up: `cache/strategy_backup_2026-06-28`.
- **Latest installer built:** `target/release/bundle/nsis/NeoEthos_0.5.0_x64-setup.exe` (37.2MB, ~21:39). **Operator: REINSTALL to see all the above.**

## 2) RUNNING RIGHT NOW (detached)
- **Risky discovery** (GA-tuned engine): 13 pairs × H1,M30,M15,M5,M3 = 65 units. Log `cache/risky_discovery.log`. Stop: `touch cache/risky_stop.flag`. Resume: add `--resume`. Writes validated strategies → `cache/auto_loop/`. ~unit 3/65 at handoff, ~30min/unit (slow, CPU-bound). Mode = `system.trading_mode: risky` in config.yaml.

---

## 3) REMAINING — IN ORDER (the Settings-consolidation directive + levers)

### A. Finish Settings consolidation (operator mandate: "config that changes → Settings, with dropdowns + descriptions; config.yaml editing becomes form-based; only Discovery+Training stay outside")
1. **Data config → Settings** (default symbol/TF, data dir; keep the download ACTION on Data screen).
2. **News config → Settings** (calendar enabled/source, news_trading_mode dropdown).
3. **Advanced raw-YAML → friendly forms** — THE BIG ONE ("config.yaml with dropdowns"). Keep raw YAML only as a power-user fallback. Use the `/settings` DTO + knob-catalog. Add fields to SettingsDto/SettingsUpdateDto for any knob you surface (pattern below).
4. **Surface the tuning knobs in Help & Guide** (add a "Search tuning" section) AND the **TUI** (operator said UI first, TUI after).

### B. #35 Sealed lockbox (foundation done, finish carefully — VALIDATION-CRITICAL, do not rush)
- Env-driven `NEOETHOS_LOCKBOX_FRAC` (default 0.0 = OFF, safe). In `cmd_discover` (crates/neoethos-cli/src/main.rs ~line 1094, before `run_discovery_cycle`): split features+ohlcv via `FeatureFrame::row_slice` + an Ohlcv row-slice; run discovery on DEV only; eval the final portfolio on the sealed lockbox via `compute_discovery_forward_test_artifacts` (or in-memory backtest); gate/flag export on lockbox profitability. Verify row counts + that the lockbox was never seen by the search. Needs a CLI rebuild + a single-unit test run.

### C. Real bugs found (verify/fix tomorrow when market OPEN)
- **pips = +0.0 while P/L is real (+94£)** — cross-pair (EURUSD on GBP account) + no live tick → `compute_pnl_pips` returns 0. FIX: derive pips FX-free from entry vs last-known price (works even market-closed). bridge.rs ~736-815.
- **Live candles not updating** — was just SUNDAY (market closed). VERIFY Monday it streams; if not, debug the spot stream (it spawns fine, account 47367144 enabled, but spots=0 on weekend = expected).
- (optional) Mutation aggressiveness → config knob (currently a stronger hardcoded default).

### D. Other levers (background/compute)
- #2 live↔backtest parity (windowed feature recompute) — deep, needs live data.
- #5 more diversification (the discovery run is producing this now).
- #3 demo-gate is ALREADY wired (`app_services/live_gate.rs`, blocks live on real-money until ≥100 demo trades within 20%) — VERIFY it in the auto-trading test.

---

## 4) TOMORROW — put strategies into operation + test auto-trading
1. Stop/finish the risky discovery; review results in **Strategy Report** + **Strategy Lab** (validated strategies in `cache/auto_loop/`).
2. **Autopilot**: pick a portfolio → **Replay (dry-run)** → confirm it runs → **Start live on DEMO** (demo-gate allows demo). Watch the bar→signal→order loop.
3. **Stress-test Auto-Trading UI + backend**: order placement, position open/close, SL/TP modify, the demo forward-test gate (`/autonomous/gate`), live status, P/L + pips (now market-open), live candles.
4. Log any UI/backend issues → fix.

---

## 5) KEY KNOWLEDGE / GOTCHAS (so a fresh context doesn't relearn)
- **Risky vs PropFirm = `system.trading_mode` ("risky"/"growth"→Risky), NOT `models.discovery_mode`** (that only triggers Strict). Tell-tale of wrong-mode: log `F-305 ... PropFirm mode`.
- **Builds:** ALWAYS `npx tauri build` (cargo build does NOT embed the frontend). ~10 min. Frontend-only change still needs a rebuild. The CLI (`neoethos-cli`) is a SEPARATE binary — rebuild it (`cargo build --release -p neoethos-cli`) for engine changes used by discovery; **can't rebuild while the discovery exe is running (Windows file lock)** — stop it first.
- **Installed app:** `C:\Users\konst\AppData\Local\NeoEthos\neoethos-desktop.exe`. computer-use resolver only knows INSTALLED apps (was already granted). request_access needs the operator to click Approve (verbal "go ahead" can't bypass the OS dialog).
- **Data root:** `prepare_data_root` (lib.rs) → `NEOETHOS_USER_DATA_DIR` override (set to the repo for the dev) → else `%LOCALAPPDATA%\neoethos` (seeded from bundled resources). config/data/cache/models resolve relative to it.
- **Settings DTO pattern** (to surface any config knob): add field to `SettingsDto` + map in `dto_from_settings` + add `Option` to `SettingsUpdateDto` + write in `update_settings` (clamp) → `settings.save(config_path())`. File: `crates/neoethos-app/src/server/settings.rs`. Frontend: `SettingsUpdate` type in `desktop/src/api.ts` + control in `desktop/src/screens/Settings.tsx`.
- **Discovery internals:** `run_discovery_cycle` (discovery.rs); funnel = data→features→prefilter(top_k)→GA(50k cands)→profitable archive→quality→correlation→walk-forward→CPCV→export. GA fitness = `scoring::ga_fitness` (rewards monthly consistency). Mutation = `genetic/evolution_math.rs::mutate`. Stagnation/patience = `genetic/runtime_overrides.rs` (config `models.search_runtime`).
- **cTrader:** account 47367144 (Demo, enabled_for_execution). Historical bars + account-runtime work. Spots=0 was weekend. Canonical TFs exclude H2/H3. CANT_ROUTE earlier was on account-runtime multi-calls (transient).
- **Repo dir:** `C:\Users\konst\development\forex-ai`. Branch: master. Everything committed.
