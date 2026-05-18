# UI Test Playbook — 2026-05-18 evening handoff

**Status:** Σχεδιάστηκε πλήρως, αλλά το auto-execution μέσω
computer-use **μπλοκάρισε** γιατί απουσίαζες από το μηχάνημα και
το `request_access` dialog δεν εγκρίθηκε εντός του 180-sec window.
**Run this playbook χειροκίνητα όταν επιστρέψεις στις 19:00 EEST.**

Estimated total time: 28 λεπτά.

## Pre-flight (1 min)

```powershell
# 1. Disk safety
Get-PSDrive C | Select-Object @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}
# Should be > 80 GB. If < 30 GB, abort and free space first.

# 2. Verify Gemma model download completed
$gguf = "C:\Users\konst\development\forex-ai\resources\models\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
if (Test-Path $gguf) {
    "Model: $([math]::Round((Get-Item $gguf).Length / 1GB, 2)) GB"
} else {
    "Model NOT YET downloaded — run .\scripts\fetch-gemma-model.ps1"
}

# 3. Run the gemma crate tests
cd C:\Users\konst\development\forex-ai
cargo test -p forex-gemma --release
# Expected: 138+ tests pass, 0 fail.
```

## Launch the app (30 sec)

```powershell
# OPTION A — existing release binary (no gemma-helper)
Start-Process "C:\Users\konst\development\forex-ai\target\release\forex-app.exe"

# OPTION B — rebuild with gemma-helper feature (~30 min cold build)
# cd C:\Users\konst\development\forex-ai
# cargo build --release -p forex-app --features gemma-helper
```

## 1. Settings → Brokers → cTrader (5 min)

| # | Test | Expected | Result |
|---|------|----------|--------|
| 1.1 | Open Settings panel | Renders without errors | ☐ |
| 1.2 | Navigate to Brokers → cTrader | Section visible | ☐ |
| 1.3 | Client ID field | Pre-populated από `EMBEDDED_CTRADER_CLIENT_ID` | ☐ |
| 1.4 | Client Secret field | Pre-populated (masked) | ☐ |
| 1.5 | Redirect URI field | Pre-populated | ☐ |
| 1.6 | Environment toggle (Demo / Live) | Switches without crash | ☐ |
| 1.7 | "Start cTrader Login (Automatic)" | Browser ανοίγει σε OAuth URL | ☐ |
| 1.8 | "Start cTrader Auth" (manual) | Auth URL panel renders | ☐ |
| 1.9 | "Prepare Token Request" | POST payload preview | ☐ |
| 1.10 | Manual Code input + "Accept Code" | Δέχεται text χωρίς panic | ☐ |
| 1.11 | "Discover Accounts" χωρίς OAuth | Friendly error (όχι crash) | ☐ |
| 1.12 | "Restore Saved Session" | Είτε loads είτε "no session" | ☐ |
| 1.13 | "Clear Saved Session" | Clears + disables Restore | ☐ |
| 1.14 | "Create Demo Account" link | Ανοίγει browser → ctrader.com | ☐ |
| 1.15 | "Create Live Account" link | Ανοίγει browser → ctrader.com | ☐ |
| 1.16 | "Save Credentials to Disk" | Γράφει broker_credentials.toml | ☐ |
| 1.17 | Account targets (add/remove) | Add row, remove row, no crash | ☐ |

## 2. Settings → Brokers → DxTrade (3 min)

| # | Test | Expected | Result |
|---|------|----------|--------|
| 2.1 | Platform URL text edit | Accepts `https://demo.dx.trade` | ☐ |
| 2.2 | Username text edit | Accepts free text | ☐ |
| 2.3 | **Domain text edit** (NEW D3.1) | Accepts `default` | ☐ |
| 2.4 | Password text edit (masked) | No echo crash | ☐ |
| 2.5 | Account targets render | List visible | ☐ |
| 2.6 | Switch cTrader ⇄ DxTrade | Both panels render, no race | ☐ |

⚠ **Note:** Το Domain field είναι νέο (Phase D3.1, 2026-05-18). Αν το
wizard / settings UI **δεν** εκθέτει Domain text-edit, αυτό είναι
follow-up — υπάρχει στο `DxTradeBrokerSettings`, αλλά πιθανώς δεν
έχει wired-up egui row ακόμα.

## 3. Main app panels (8 min)

| # | Panel | Smoke test | Result |
|---|-------|------------|--------|
| 3.1 | Markets / Watchlist | Symbol grid renders (fallback OK) | ☐ |
| 3.2 | Dashboard | Top-level KPIs render | ☐ |
| 3.3 | Trade Watch | Positions table renders (empty OK) | ☐ |
| 3.4 | News panel | Renders χωρίς API key | ☐ |
| 3.5 | Order Ticket | Symbol/dir/vol/SL/TP clickable — **NO Submit** | ☐ |
| 3.6 | Discovery | Status panel renders | ☐ |
| 3.7 | Training | Status panel renders | ☐ |
| 3.8 | Intelligence / AI Insights | Renders χωρίς panic | ☐ |
| 3.9 | Broker Setup | Wizard step opens cleanly | ☐ |
| 3.10 | Runtime | Resource/perf panel | ☐ |
| 3.11 | Data Bootstrap | "Browse..." picker opens | ☐ |
| 3.12 | Hardware | Profile detection runs | ☐ |
| 3.13 | Risk | Risky Mode config + tier display | ☐ |
| 3.14 | Settings | Όλα τα sub-tabs accessible | ☐ |

## 4. Headless API integration (5 min)

```powershell
cd C:\Users\konst\development\forex-ai
.\target\release\forex-cli.exe --help

.\target\release\forex-app.exe --headless --auto-discovery
# Ctrl+C after 5 seconds.

.\target\release\forex-app.exe --headless --auto-training
# Ctrl+C after 5 seconds.

.\target\release\forex-app.exe --headless --auto-discovery --auto-training
# Ctrl+C after 5 seconds.
```

| # | Command | Exit clean? | Result |
|---|---------|-------------|--------|
| 4.1 | `forex-cli --help` | Shows usage | ☐ |
| 4.2 | `--headless --auto-discovery` (5 s) | No panic | ☐ |
| 4.3 | `--headless --auto-training` (5 s) | No panic | ☐ |
| 4.4 | `--headless` both jobs (5 s) | No deadlock | ☐ |

## 5. Disk safety check after (1 min)

```powershell
Get-PSDrive C | Select-Object @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}
Get-ChildItem -Path "C:\Users\konst\development\forex-ai" -Directory |
    Where-Object {$_.Name -in @('logs','cache','models','target','catboost_info','resources')} |
    ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -ErrorAction SilentlyContinue |
                 Measure-Object -Property Length -Sum).Sum / 1GB
        [PSCustomObject]@{ Dir = $_.Name; SizeGB = [math]::Round($size, 2) }
    } | Format-Table
```

## 6. Reporting

Συμπλήρωσε τα ☐ με ✓/✗/⚠ και σώσε ως
`docs/qa/2026-05-18-ui-full-test-report.md`. Για κάθε ✗/⚠ γράψε
σύντομη παράγραφο: Where (panel + button) / What (observed) /
Severity (Blocker/Major/Minor/Cosmetic) / Reproducible (Y/N).

Screenshots σε `docs/qa/screenshots/2026-05-18/`:

```powershell
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$bmp = New-Object Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height)
$gfx = [Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen(0, 0, 0, 0, $bmp.Size)
$bmp.Save("docs\qa\screenshots\2026-05-18\panel-name.png", [Drawing.Imaging.ImageFormat]::Png)
```

## 7. Known potential issues

- **DxTrade Domain field UI** — νέο πεδίο, μπορεί να μην έχει egui
  row ακόμα. One-line fix στο `crates/forex-app/src/ui/...`.
- **OrderSource::AiSuggested variant** — λείπει από
  `crates/forex-app/src/app_services/trading/mod.rs`. Το G6b
  (suggestion path) δεν τρέχει end-to-end μέχρι να μπει. ΟΧΙ
  blocker για το UI test.
- **gemma-helper feature** disabled by default — η τρέχουσα exe
  ΔΕΝ περιέχει τον helper. Για end-to-end test, ανοικοδόμηση με
  `--features gemma-helper`.
