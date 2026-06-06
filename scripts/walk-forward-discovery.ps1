<#
.SYNOPSIS
  Walk-forward discovery for weak/low-RAM machines: slice a (huge) dataset by DATE into
  N training chunks + a held-out forward-test window, run discovery on each chunk
  (each fits RAM → no OOM), then forward-test every found portfolio on the holdout.

  This is the "3+3+3+2" pattern: e.g. 3 training chunks of 3 years each + a 2-year
  forward-test holdout. Each chunk is a separate sliced dataset, so M1 (5M+ rows)
  never loads all at once.

.NOTES
  Requires a CPU-ONLY build of neoethos-cli (NO gpu-vulkan) to avoid iGPU/wgpu OOM:
      cargo build --release -p neoethos-cli
  The chunks share a seen-signature file + the discovery ledger, so each chunk ADDS
  NEW diverse strategies (does not re-discover the previous chunk's) — a growing library.
  Robustness is judged by the holdout forward-test (trader-replay on the held-out window).

.EXAMPLE
  ./scripts/walk-forward-discovery.ps1 -Symbol EURUSD -Base M1 -EndDate 2025-12-31 `
      -ChunkYears 3 -NumChunks 3 -HoldoutYears 2 -Population 64
#>
[CmdletBinding()]
param(
  [string]$Symbol = "EURUSD",
  [string]$Base = "M1",
  [string]$Root = "data",
  [string]$WorkDir = "cache/walkforward",
  [string]$EndDate = "",                 # last bar date YYYY-MM-DD; "" = today (slice keeps whatever exists <= it)
  [int]$ChunkYears = 3,
  [int]$NumChunks = 3,
  [int]$HoldoutYears = 2,
  [int]$Population = 64,                  # weak-machine default; lower = less RAM/CPU
  [int]$Generations = 5000,              # time-bounded by MaxHours
  [double]$MaxHours = 0.5,
  [string]$Cli = "target/release/neoethos-cli.exe"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Cli)) { throw "CLI not found at $Cli — build it CPU-only: cargo build --release -p neoethos-cli" }
$end = if ([string]::IsNullOrWhiteSpace($EndDate)) { (Get-Date).ToString('yyyy-MM-dd') } else { $EndDate }
$endDt = [datetime]::ParseExact($end, 'yyyy-MM-dd', $null)
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$seenFile = Join-Path $WorkDir "$($Symbol)_$($Base).seen.bin"
$tag = "$($Symbol)_$($Base)"
Write-Host "=== Walk-forward discovery: $Symbol $Base | $NumChunks x ${ChunkYears}y train + ${HoldoutYears}y forward-test | end=$end ===" -ForegroundColor Cyan

# --- compute windows (backward from EndDate): training chunks then the holdout (most recent) ---
$holdoutStart = $endDt.AddYears(-$HoldoutYears)
$chunks = @()
for ($k = 1; $k -le $NumChunks; $k++) {
  # chunk 1 = oldest. The NumChunks*ChunkYears years immediately before the holdout.
  $cEnd   = $holdoutStart.AddYears(-($NumChunks - $k) * $ChunkYears)
  $cStart = $cEnd.AddYears(-$ChunkYears)
  $chunks += [pscustomobject]@{
    Idx   = $k
    From  = $cStart.ToString('yyyy-MM-dd')
    To    = $cEnd.ToString('yyyy-MM-dd')
    Dir   = (Join-Path $WorkDir "chunk$($k)")
    Out   = (Join-Path (Join-Path $WorkDir "chunk$($k)") "portfolio.json")
  }
}
$holdoutDir = Join-Path $WorkDir "holdout"

function Invoke-Cli { param([string[]]$CliArgs) & $Cli @CliArgs; if ($LASTEXITCODE -ne 0) { Write-Host "  (cli exit $LASTEXITCODE)" -ForegroundColor Yellow } }

# --- 1) training chunks: slice -> discover (CPU, capped, shared seen-file + ledger) ---
foreach ($c in $chunks) {
  Write-Host "`n--- CHUNK $($c.Idx): $($c.From) .. $($c.To) ---" -ForegroundColor Green
  Invoke-Cli @("slice-dataset","--symbol",$Symbol,"--base",$Base,"--root",$Root,"--out-root",$c.Dir,"--from-date",$c.From,"--to-date",$c.To)
  if (-not (Test-Path (Join-Path $c.Dir "symbol=$Symbol/timeframe=$Base/data.vortex"))) {
    Write-Host "  no data in this window — skipping" -ForegroundColor Yellow; continue
  }
  $env:NEOETHOS_BOT_PROP_SEEN_FILE = $seenFile   # shared across chunks → accumulate diverse, avoid re-discovery
  Invoke-Cli @("discover","--symbol",$Symbol,"--base",$Base,"--root",$c.Dir,"--out",$c.Out,
               "--population","$Population","--generations","$Generations")
}
Remove-Item Env:\NEOETHOS_BOT_PROP_SEEN_FILE -ErrorAction SilentlyContinue

# --- 2) holdout slice (the forward-test window, most recent HoldoutYears) ---
Write-Host "`n--- HOLDOUT (forward-test): $($holdoutStart.ToString('yyyy-MM-dd')) .. $end ---" -ForegroundColor Green
Invoke-Cli @("slice-dataset","--symbol",$Symbol,"--base",$Base,"--root",$Root,"--out-root",$holdoutDir,"--from-date",$holdoutStart.ToString('yyyy-MM-dd'),"--to-date",$end)

# --- 3) forward-test every found portfolio on the holdout (OOS) ---
Write-Host "`n=========== FORWARD-TEST (OOS on holdout) ===========" -ForegroundColor Cyan
foreach ($c in $chunks) {
  $lp = "$($c.Out).live_portfolio.json"
  if (Test-Path $lp) {
    Write-Host "`n[chunk $($c.Idx) -> holdout] $lp" -ForegroundColor Green
    Invoke-Cli @("trader-replay","--portfolio",$lp,"--root",$holdoutDir)
  } else {
    Write-Host "[chunk $($c.Idx)] no portfolio found (nothing passed the gates)" -ForegroundColor Yellow
  }
}
Write-Host "`n=== done. Library seen-file: $seenFile | ledgers: cache/search/ ===" -ForegroundColor Cyan
