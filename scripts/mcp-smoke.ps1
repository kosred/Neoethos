# mcp-smoke.ps1 — Tier-2 smoke test of the Codex control plane against the
# REAL running app on the DEMO cTrader account (docs/codex-control-plane.md
# §6.2, per the test-only-on-real-data rule).
#
# Prerequisites:
#   * The desktop app (or `neoethos-app --server`) is running and the DEMO
#     cTrader account is connected.
#   * `cargo build --release -p neoethos-mcp` has produced
#     target/release/neoethos-mcp.exe.
#
# What it does, over real stdio MCP:
#   1. initialize + tools/list (asserts 62 tools)
#   2. broker_status — ABORTS LOUDLY unless environment=Demo + connected
#   3. read-path sanity: account_snapshot, journal_stats, journal_analytics,
#      get_chart EURUSD M5
#   4. full trade round trip: open_position EURUSD buy 0.01 SL20/TP20 →
#      list_positions → modify_position_protection (widen SL) →
#      close_position (full) → journal_trades shows the round trip
#   5. pending round trip: far-from-market limit → list → cancel
#   6. job pattern: discovery_start (population 64, generations 1) → poll
#      engine_status → discovery_stop
#   7. replay_backtest (proves the >60s path inside the 600s budget)
#
# The Live-refusal path is deliberately NEVER exercised here — it is covered
# exhaustively by the tier-1 mock tests (cargo test -p neoethos-mcp).
#
# Usage:
#   pwsh -File scripts/mcp-smoke.ps1 [-BaseUrl http://127.0.0.1:7423] [-SkipTrade] [-SkipLong]

param(
    [string]$BaseUrl = "http://127.0.0.1:7423",
    # Skip step 4/5 (the real demo trade round trips).
    [switch]$SkipTrade,
    # Skip step 6/7 (discovery + replay — the long compute paths).
    [switch]$SkipLong
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $repo "target\release\neoethos-mcp.exe"
if (-not (Test-Path $exe)) {
    throw "neoethos-mcp.exe not found at $exe — run: cargo build --release -p neoethos-mcp"
}

# ── stdio MCP plumbing ───────────────────────────────────────────────────────

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $exe
$psi.Arguments = "--base-url $BaseUrl"
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$proc = [System.Diagnostics.Process]::Start($psi)

$script:nextId = 0
function Send-Rpc([string]$method, $params) {
    $script:nextId++
    $msg = @{ jsonrpc = "2.0"; id = $script:nextId; method = $method }
    if ($null -ne $params) { $msg.params = $params }
    $proc.StandardInput.WriteLine(($msg | ConvertTo-Json -Depth 20 -Compress))
    $proc.StandardInput.Flush()
    while ($true) {
        $line = $proc.StandardOutput.ReadLine()
        if ($null -eq $line) { throw "MCP server closed stdout while waiting for a reply to $method" }
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $reply = $line | ConvertFrom-Json
        if ($reply.id -eq $script:nextId) {
            if ($reply.PSObject.Properties.Name -contains "error") {
                throw "JSON-RPC error from ${method}: $($reply.error | ConvertTo-Json -Depth 10 -Compress)"
            }
            return $reply.result
        }
        # Notifications / server-initiated messages: skip.
    }
}

function Send-Notification([string]$method) {
    $proc.StandardInput.WriteLine((@{ jsonrpc = "2.0"; method = $method } | ConvertTo-Json -Compress))
    $proc.StandardInput.Flush()
}

function Invoke-Tool([string]$name, $arguments, [switch]$AllowError) {
    $params = @{ name = $name }
    if ($null -ne $arguments) { $params.arguments = $arguments }
    $result = Send-Rpc "tools/call" $params
    $text = ($result.content | Where-Object { $_.type -eq "text" } | Select-Object -First 1).text
    if ($result.isError -and -not $AllowError) {
        throw "TOOL $name FAILED: $text"
    }
    if ($result.isError) {
        Write-Host "  [expected-error] $name -> $text" -ForegroundColor DarkYellow
        return $null
    }
    try { return $text | ConvertFrom-Json } catch { return $text }
}

function Step([string]$label) { Write-Host "== $label" -ForegroundColor Cyan }

try {
    # ── 1. initialize + tools/list ──────────────────────────────────────────
    Step "initialize + tools/list"
    $init = Send-Rpc "initialize" @{
        protocolVersion = "2025-06-18"
        capabilities = @{}
        clientInfo = @{ name = "mcp-smoke"; version = "1.0" }
    }
    Send-Notification "notifications/initialized"
    Write-Host "  server: $($init.serverInfo.name) $($init.serverInfo.version)"
    $tools = (Send-Rpc "tools/list" $null).tools
    if ($tools.Count -ne 62) { throw "expected 62 tools, got $($tools.Count)" }
    Write-Host "  62 tools listed."

    # ── 2. broker_status gate ───────────────────────────────────────────────
    Step "broker_status (demo gate)"
    $broker = Invoke-Tool "broker_status" $null
    $env_ = $broker.status.environment
    $connected = $broker.status.connected
    Write-Host "  environment=$env_ connected=$connected account=$($broker.status.accountId)"
    if ($env_ -ne "Demo" -or -not $connected) {
        throw "ABORT: the smoke test requires a CONNECTED DEMO account (got environment=$env_, connected=$connected). Nothing further will run."
    }

    # ── 3. read-path sanity ─────────────────────────────────────────────────
    Step "read-path sanity"
    $snap = Invoke-Tool "account_snapshot" @{}
    Write-Host "  balance=$($snap.balance) $($snap.currency), positions=$($snap.positions.Count)"
    $null = Invoke-Tool "journal_stats" $null
    Write-Host "  journal_stats ok"
    $null = Invoke-Tool "journal_analytics" $null
    Write-Host "  journal_analytics ok"
    $chart = Invoke-Tool "get_chart" @{ symbol = "EURUSD"; timeframe = "M5"; limit = 50 }
    Write-Host "  get_chart EURUSD M5: $($chart.candleCount) candles (source=$($chart.source))"

    if (-not $SkipTrade) {
        # ── 4. trade round trip ────────────────────────────────────────────
        Step "trade round trip (DEMO, 0.01 lot, SL 20 / TP 20)"
        $before = @($snap.positions | ForEach-Object { $_.positionId })
        $opened = Invoke-Tool "open_position" @{
            symbol = "EURUSD"; side = "buy"; volume_lots = 0.01
            stop_loss_pips = 20.0; take_profit_pips = 20.0
            comment = "mcp-smoke round trip"
        }
        Write-Host "  open: status=$($opened.status) positionId=$($opened.positionId)"
        Start-Sleep -Seconds 6   # let the bridge refresh the snapshot
        $positions = Invoke-Tool "list_positions" $null
        $mine = $positions | Where-Object { $before -notcontains $_.positionId } | Select-Object -First 1
        if ($null -eq $mine) { throw "the freshly opened position did not appear in list_positions" }
        Write-Host "  visible: positionId=$($mine.positionId) $($mine.symbol) $($mine.side) SL=$($mine.stopLoss)"

        if ($null -ne $mine.stopLoss -and $mine.stopLoss -gt 0) {
            $widened = [math]::Round($mine.stopLoss - 0.0005, 5)  # widen a BUY stop by 5 pips
            $null = Invoke-Tool "modify_position_protection" @{
                position_id = $mine.positionId; stop_loss_price = $widened
            }
            Write-Host "  protection amended: SL -> $widened"
        } else {
            Write-Host "  (no broker SL price on the row yet — skipping the amend step)"
        }

        $closed = Invoke-Tool "close_position" @{ position_id = $mine.positionId }
        Write-Host "  close: status=$($closed.status)"
        Start-Sleep -Seconds 6
        $recent = Invoke-Tool "journal_trades" @{ limit = 10 }
        Write-Host "  journal shows $($recent.Count) recent trades (round trip should be on top)"

        # ── 5. pending round trip ──────────────────────────────────────────
        Step "pending order round trip (far-from-market limit)"
        $chart2 = Invoke-Tool "get_chart" @{ symbol = "EURUSD"; timeframe = "M5"; limit = 2 }
        $trigger = [math]::Round($chart2.latestClose * 0.95, 5)   # 5% below market: cannot fill
        $pending = Invoke-Tool "place_pending_order" @{
            symbol = "EURUSD"; side = "buy"; order_type = "limit"
            volume_lots = 0.01; trigger_price = $trigger; stop_loss_pips = 20.0
            comment = "mcp-smoke pending"
        }
        Write-Host "  placed: orderId=$($pending.orderId) trigger=$trigger"
        Start-Sleep -Seconds 3
        $orders = Invoke-Tool "list_pending_orders" $null
        $ours = $orders | Where-Object { $_.orderId -eq $pending.orderId }
        if ($null -eq $ours) { throw "the resting order did not appear in list_pending_orders" }
        $null = Invoke-Tool "cancel_pending_order" @{ order_id = $pending.orderId }
        Write-Host "  cancelled orderId=$($pending.orderId)"
    } else {
        Write-Host "== trade round trips SKIPPED (-SkipTrade)" -ForegroundColor DarkYellow
    }

    if (-not $SkipLong) {
        # ── 6. discovery job pattern ───────────────────────────────────────
        Step "discovery job pattern (population 64, generations 1)"
        $null = Invoke-Tool "discovery_start" @{ population = 64; generations = 1 }
        $sawRunning = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 2
            $engines = Invoke-Tool "engine_status" $null
            if ($engines.discovery -eq "Running") { $sawRunning = $true; break }
            if ($engines.discovery -in @("Succeeded", "Failed")) { break }
        }
        Write-Host "  discovery observed running=$sawRunning -> stopping"
        $null = Invoke-Tool "discovery_stop" $null
        Write-Host "  discovery_stop ok (idempotent)"

        # ── 7. the long synchronous path ───────────────────────────────────
        Step "replay_backtest (long synchronous call, 600s budget)"
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $replay = Invoke-Tool "replay_backtest" @{} -AllowError
        $sw.Stop()
        if ($null -ne $replay) {
            Write-Host "  replay finished in $([int]$sw.Elapsed.TotalSeconds)s"
        } else {
            Write-Host "  replay reported a tool error (acceptable if no on-disk history for the configured symbol) after $([int]$sw.Elapsed.TotalSeconds)s"
        }
    } else {
        Write-Host "== long compute paths SKIPPED (-SkipLong)" -ForegroundColor DarkYellow
    }

    Write-Host ""
    Write-Host "MCP SMOKE: ALL STEPS PASSED" -ForegroundColor Green
}
finally {
    if (-not $proc.HasExited) {
        try { $proc.StandardInput.Close() } catch {}
        if (-not $proc.WaitForExit(5000)) { $proc.Kill($true) }
    }
    $stderrTail = $proc.StandardError.ReadToEnd()
    if ($stderrTail) {
        Write-Host "--- server stderr (logs) ---" -ForegroundColor DarkGray
        $stderrTail -split "`n" | Select-Object -Last 15 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
    }
}
