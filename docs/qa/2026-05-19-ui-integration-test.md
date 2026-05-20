# UI Integration Test — v0.5.0 binary, 2026-05-19

**Binary under test:** `target/release/forex-app.exe` (v0.5.0 build,
2026-05-19 11:58, 106.4 MB).
**Operator:** kosred · **Method:** real button clicks via Windows-MCP
+ computer-use against the running binary, no rebuild.
**Question per row:** is this button actually wired to backend
behaviour, or only cosmetic?

Status legend:
- ✅ Wired correctly, works (with evidence)
- ⚠ Wired but partial (action runs but UI doesn't fully update)
- ❌ Cosmetic only — no backend action
- ⏸ Untestable in this session (needs specific market state, etc.)
- 🟡 Not yet exercised in this pass

---

## Step A — App boot + wizard
| Item | Status | Evidence |
|---|---|---|
| App launches at v0.5.0 build (target/release/forex-app.exe) | 🟡 | running pid 35176, MainWindowTitle "Forex AI - Pure Rust Terminal" |
| Wizard opens on first launch | 🟡 | wizard_state.json deleted before launch |
| Wizard Steps 1-11 end-to-end | 🟡 | |
| Wizard Apply transitions to workspace | 🟡 | |
| cTrader auto-connected after Apply | 🟡 | |

## Step B — Charts live real-time
| Item | Status | Evidence |
|---|---|---|

## Step C — Watchlist live
| Item | Status | Evidence |
|---|---|---|

## Step D — Order Ticket
| Item | Status | Evidence |
|---|---|---|

## Step E — Discovery
| Item | Status | Evidence |
|---|---|---|

## Step F — Training
| Item | Status | Evidence |
|---|---|---|

## Step G — AI Helper / Gemma
| Item | Status | Evidence |
|---|---|---|

## Step H — Remaining tabs (Trade Watch / News / Intelligence / Broker Setup / Data Bootstrap / Hardware / Risk / Settings / Runtime)
| Item | Status | Evidence |
|---|---|---|

## Step I — Top bar + status bar
| Item | Status | Evidence |
|---|---|---|

---

## Summary score

Will be filled at end of pass.
