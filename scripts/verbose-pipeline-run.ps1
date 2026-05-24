# scripts/verbose-pipeline-run.ps1 — extra-long Discovery + Training run
# under maximum verbosity so we can scan the log for bugs the UI smoke
# test masks. Designed for the "Phase 2 audit before UI work" — see #210.
#
# What it does:
#   1. Kills any running neoethos-app.exe so port 7423 is free.
#   2. Runs `target/release/neoethos-app.exe --headless --auto-discovery
#      --auto-training` with RUST_LOG cranked up to debug on the engine
#      crates that matter (search, models, data, app).
#   3. Tees stdout+stderr to `verbose-runs/<UTC>.log`.
#   4. When the process exits (or you Ctrl-C), summarises the log:
#        - line count
#        - per-level counts (ERROR / WARN / INFO)
#        - first 20 ERRORs / first 20 WARNs (with line numbers)
#        - any panic backtraces
#        - timing markers from the discovery / training side
#
# Usage:
#   .\scripts\verbose-pipeline-run.ps1                 # default 30 min budget
#   .\scripts\verbose-pipeline-run.ps1 -MaxMinutes 90  # extra long
#   .\scripts\verbose-pipeline-run.ps1 -NoTimeout      # let it run to completion
#
# Notes:
#   - The `--auto-discovery` + `--auto-training` flags pick the FIRST
#     symbol the discover_symbols() probe finds. With the current data
#     directory that is AUDUSD. To stress a specific pair, point
#     `--local-data-dir` at a directory containing only that pair OR
#     pre-emptively wire a CLI override (#194 lifted GA params to
#     env vars; symbol selection in headless mode is still hard-coded
#     to the first-on-disk match — file a follow-up if you need a flag).

[CmdletBinding()]
param(
    [int]$MaxMinutes = 30,
    [switch]$NoTimeout,
    [string]$RustLog = "neoethos_app=debug,neoethos_search=debug,neoethos_models=debug,neoethos_data=debug,info"
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$binary = Join-Path $repoRoot 'target\release\neoethos-app.exe'
$logDir = Join-Path $repoRoot 'verbose-runs'
$timestamp = (Get-Date -Format 'yyyy-MM-ddTHH-mm-ssZ').Replace(':', '-')
$logFile = Join-Path $logDir "$timestamp.log"

if (-not (Test-Path $binary)) {
    Write-Host "Release binary not found at $binary." -ForegroundColor Red
    Write-Host "Build first: cargo build --release -p neoethos-app" -ForegroundColor Yellow
    exit 1
}

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# 1. Make port 7423 free
Get-Process neoethos-app -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Killing existing neoethos-app pid=$($_.Id)" -ForegroundColor Yellow
    try { $_ | Stop-Process -Force } catch {}
}
Start-Sleep -Milliseconds 500

# 2. Launch under RUST_LOG verbose
$env:RUST_LOG = $RustLog
$env:RUST_BACKTRACE = "full"
$env:NEOETHOS_LAUNCHED_BY_FLUTTER = "1"  # suppress orphaned-double-click dialog

Write-Host "" -ForegroundColor Cyan
Write-Host "==> NeoEthos verbose pipeline run" -ForegroundColor Cyan
Write-Host "    binary    : $binary"
Write-Host "    log file  : $logFile"
Write-Host "    RUST_LOG  : $RustLog"
if ($NoTimeout) {
    Write-Host "    timeout   : (none, run until natural exit / Ctrl-C)"
} else {
    Write-Host "    timeout   : $MaxMinutes minutes"
}
Write-Host ""

$startTime = Get-Date

$args = @('--server', '--headless', '--auto-discovery', '--auto-training', '--config', 'config.yaml')

if ($NoTimeout) {
    # Foreground run — operator hits Ctrl-C to stop.
    Push-Location $repoRoot
    try {
        & $binary @args 2>&1 | Tee-Object -FilePath $logFile
    } finally {
        Pop-Location
    }
} else {
    Push-Location $repoRoot
    try {
        $process = Start-Process -FilePath $binary -ArgumentList $args `
            -RedirectStandardOutput "$logFile.out" -RedirectStandardError "$logFile.err" `
            -PassThru -NoNewWindow
        Write-Host "Started pid=$($process.Id). Waiting up to $MaxMinutes min..." -ForegroundColor Green
        $deadline = $startTime.AddMinutes($MaxMinutes)
        while (-not $process.HasExited) {
            if ((Get-Date) -gt $deadline) {
                Write-Host "Timeout reached — stopping process gracefully." -ForegroundColor Yellow
                $process | Stop-Process -Force
                break
            }
            Start-Sleep -Seconds 5
        }
        # Concat stdout + stderr into one log so the analyser doesn't
        # need to know about the split.
        Get-Content "$logFile.out", "$logFile.err" -ErrorAction SilentlyContinue | Set-Content $logFile -Encoding UTF8
        Remove-Item "$logFile.out", "$logFile.err" -ErrorAction SilentlyContinue
    } finally {
        Pop-Location
    }
}

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "==> Run finished. Elapsed: $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host ""

# 3. Post-run analysis
if (-not (Test-Path $logFile)) {
    Write-Host "Log file $logFile not produced. Did the process even start?" -ForegroundColor Red
    exit 1
}

$lines = Get-Content $logFile
$lineCount = $lines.Count
$errCount = ($lines | Select-String -Pattern '\bERROR\b' -SimpleMatch:$false).Count
$warnCount = ($lines | Select-String -Pattern '\bWARN\b' -SimpleMatch:$false).Count
$infoCount = ($lines | Select-String -Pattern '\bINFO\b' -SimpleMatch:$false).Count
$panicCount = ($lines | Select-String -Pattern 'panicked at|panic =').Count

Write-Host "==> Log summary  ($lineCount lines, $logFile)" -ForegroundColor Cyan
Write-Host "    ERRORs : $errCount"
Write-Host "    WARNs  : $warnCount"
Write-Host "    INFOs  : $infoCount"
Write-Host "    panics : $panicCount"
Write-Host ""

if ($errCount -gt 0) {
    Write-Host "==> First 20 ERROR lines" -ForegroundColor Red
    $lines | Select-String -Pattern '\bERROR\b' | Select-Object -First 20 | ForEach-Object {
        Write-Host ("  L{0,6}: {1}" -f $_.LineNumber, $_.Line.Trim())
    }
    Write-Host ""
}

if ($warnCount -gt 0) {
    Write-Host "==> First 20 WARN lines" -ForegroundColor Yellow
    $lines | Select-String -Pattern '\bWARN\b' | Select-Object -First 20 | ForEach-Object {
        Write-Host ("  L{0,6}: {1}" -f $_.LineNumber, $_.Line.Trim())
    }
    Write-Host ""
}

if ($panicCount -gt 0) {
    Write-Host "==> Panic backtraces" -ForegroundColor Red
    $lines | Select-String -Pattern 'panicked at|panic =' | ForEach-Object {
        Write-Host ("  L{0,6}: {1}" -f $_.LineNumber, $_.Line.Trim())
    }
    Write-Host ""
}

# Timing markers — anything we know about discovery/training boundaries
Write-Host "==> Pipeline milestones" -ForegroundColor Cyan
$lines | Select-String -Pattern 'auto-starting|job started|generation|portfolio strateg|training (started|completed|failed)|completed and|out of' -SimpleMatch:$false | Select-Object -First 30 | ForEach-Object {
    Write-Host ("  L{0,6}: {1}" -f $_.LineNumber, $_.Line.Trim())
}
Write-Host ""

Write-Host "Full log: $logFile" -ForegroundColor Gray
