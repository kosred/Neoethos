# scripts/run-7day-discovery.ps1
#
# Unattended 7-day "find strategies for everything, then train" run, launched
# detached so it survives the agent session. Two phases (operator's directive):
#   PHASE 1 — discovery ONLY, all symbols x all canonical TFs (auto-loop
#             --skip-training, resumable). A background timer writes a stop-flag
#             after 7 days so the loop exits gracefully after the current combo
#             (if it hasn't already finished all combos).
#   PHASE 2 — training for every (symbol, canonical-TF) that exists on disk.
#
# Each discovery also emits `<out>.live_portfolio.json` (the Phase-4 trader
# artifact), so after this run `neoethos-cli trader-replay --portfolio <...>`
# validates the autonomous trader on the freshly-found real genes.
#
# Logs: cache/run-7day.log  (tail it to watch progress).

$ErrorActionPreference = 'Continue'
$cli  = 'C:\Users\konst\development\forex-ai\target\release\neoethos-cli.exe'
$data = 'C:\Users\konst\development\forex-ai\data'
$repo = 'C:\Users\konst\development\forex-ai'
$flag = Join-Path $repo 'cache\auto_loop_stop.flag'
$ckpt = Join-Path $repo 'cache\auto_loop_checkpoint.json'
$log  = Join-Path $repo 'cache\run-7day.log'

Set-Location $repo
$env:NEOETHOS_BOT_DATA_ROOT = $data
# 2026-06-05 golden-mean: cap the GA's rayon pool to 8 of 12 threads so the laptop
# stays cool + usable (operator: "don't burn my computer"). Leaves 4 cores free.
$env:RAYON_NUM_THREADS = '8'
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'cache') | Out-Null
if (Test-Path $flag) { Remove-Item $flag -Force -ErrorAction SilentlyContinue }
if (Test-Path $ckpt) { Remove-Item $ckpt -Force -ErrorAction SilentlyContinue }  # fresh run
# 2026-06-05: clean any leftover out-of-core feature store. An M1 all-canonical
# combo's store can reach ~80 GB; a killed run leaks it (no Drop). Discovery
# Drop-deletes per combo, but pre-cleaning defends against disk-fill accumulation.
$fstore = Join-Path $env:TEMP 'neoethos_feature_store'
if (Test-Path $fstore) { Remove-Item $fstore -Recurse -Force -ErrorAction SilentlyContinue }

function Log($m) { "[{0}] {1}" -f (Get-Date -Format 'u'), $m | Out-File -FilePath $log -Append -Encoding UTF8 }

Log "================ 7-DAY RUN START ================"
Log "cli=$cli  data=$data"

# Background 7-day timer: writes the stop-flag so PHASE 1 exits after the current
# combo once 7 days elapse. (604800 s = 7 days.) If discovery finishes all combos
# sooner, auto-loop exits on its own and PHASE 2 starts early — that's fine.
Start-Job -Name 'stop7day' -ScriptBlock {
    Start-Sleep -Seconds 604800
    New-Item -ItemType File -Path $using:flag -Force | Out-Null
} | Out-Null
Log "Armed 7-day stop timer (writes $flag)."

# ── PHASE 1: discovery only, all symbols x canonical TFs ─────────────────────
Log "PHASE 1: discovery (auto-loop --skip-training, all symbols x canonical TFs)"
# 2026-06-05: skip M1/M3 bases — on this 6-core laptop a single M1 combo (5.27M
# bars) ran >130 min with no result (build + GA + CPCV validation too heavy). M5+
# bases (<=1M bars) finish in minutes-to-~45min. M1/M3 (noisiest scalping TFs)
# belong on the L40 VPS (real GPU + more cores).
& $cli auto-loop --skip-training --root $data --timeframes 'M5,M15,M30,H1,H4,H12,D1,W1,MN1' --stop-flag $flag *>> $log
Log "PHASE 1 complete (exit=$LASTEXITCODE)."

# ── PHASE 2: training for every (symbol, canonical-TF) on disk ───────────────
Log "PHASE 2: training all discovered combos"
$canon = @('M5','M15','M30','H1','H4','H12','D1','W1','MN1')  # M1/M3 skipped (laptop too slow; L40 VPS)
$symbols = Get-ChildItem $data -Directory -Filter 'symbol=*' -ErrorAction SilentlyContinue |
    ForEach-Object { $_.Name -replace '^symbol=','' }
foreach ($sym in $symbols) {
    $present = Get-ChildItem (Join-Path $data "symbol=$sym") -Directory -Filter 'timeframe=*' -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Name -replace '^timeframe=','' }
    $tfs = $canon | Where-Object { $present -contains $_ }
    foreach ($tf in $tfs) {
        Log "train $sym $tf"
        & $cli train --symbol $sym --base $tf *>> $log
    }
}

Get-Job -Name 'stop7day' -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue
Log "================ 7-DAY RUN COMPLETE ================"
