# find-hyperstack-creds.ps1
#
# Diagnostic helper που ψάχνει Hyperstack API key + VM info σε ΟΛΑ
# τα γνωστά Windows-side locations. Δεν τυπώνει το ίδιο το key ή
# την τιμή — μόνο πού βρέθηκε και το προφίλ του (length, last 4
# chars), ώστε ο operator να ξέρει αν χρειάζεται να δώσει νέο.
#
# Output:
#   - env vars present (HYPERSTACK_*, NEXGEN_*, SSH-related)
#   - .env στο repo root + κάθε .env σε ~/.config/hyperstack/
#   - Docker containers ονόματος *hyperstack* / *mcp*
#   - SSH config entries που πιθανώς δείχνουν στο VM
#   - PowerShell history searches για keywords
#
# Δεν τροποποιεί τίποτα. Pure read.

[CmdletBinding()]
param()

function Show($label, $value) {
    if ([string]::IsNullOrEmpty($value)) {
        Write-Host "  ⚠ $label : <empty>" -ForegroundColor Yellow
    } else {
        $len = $value.Length
        $last4 = if ($len -ge 4) { $value.Substring($len - 4) } else { '????' }
        Write-Host "  ✓ $label : found (len=$len, ends with $last4)" -ForegroundColor Green
    }
}
function Section($s) { Write-Host "`n=== $s ===" -ForegroundColor Cyan }

Section 'Environment variables'
foreach ($name in @(
    'HYPERSTACK_API_KEY','HYPERSTACK_VM_ID','HYPERSTACK_VM_IP','HYPERSTACK_VM_USER',
    'NEXGEN_API_KEY','NEXGENCLOUD_API_KEY','INFRAHUB_API_KEY',
    'HF_TOKEN','HUGGINGFACE_TOKEN'
)) {
    $v = [Environment]::GetEnvironmentVariable($name, 'Process')
    if ([string]::IsNullOrEmpty($v)) { $v = [Environment]::GetEnvironmentVariable($name, 'User') }
    if ([string]::IsNullOrEmpty($v)) { $v = [Environment]::GetEnvironmentVariable($name, 'Machine') }
    Show $name $v
}

Section '.env files'
$envCandidates = @(
    "$PSScriptRoot\..\.env",
    "$env:USERPROFILE\.env",
    "$env:USERPROFILE\.config\hyperstack\config",
    "$env:APPDATA\hyperstack\config",
    "$env:LOCALAPPDATA\hyperstack\config"
)
foreach ($p in $envCandidates) {
    $resolved = try { (Resolve-Path $p -ErrorAction Stop).Path } catch { $p }
    if (Test-Path $resolved) {
        Write-Host "  ✓ $resolved" -ForegroundColor Green
        $hsKey = (Get-Content $resolved | Select-String -Pattern '^\s*(HYPERSTACK_API_KEY|NEXGEN_API_KEY|API_KEY)\s*=').Matches.Count
        Write-Host "    Hyperstack-shaped lines: $hsKey" -ForegroundColor Gray
    } else {
        Write-Host "  · $resolved (missing)" -ForegroundColor DarkGray
    }
}

Section 'Docker containers'
$dockerPresent = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerPresent) {
    Write-Host "  ⚠ docker CLI not on PATH" -ForegroundColor Yellow
} else {
    try {
        docker ps -a --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' 2>$null | ForEach-Object {
            if ($_ -match '(?i)hyperstack|mcp|nexgen') {
                Write-Host "  ✓ $_" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "  ⚠ docker ps failed: $_" -ForegroundColor Yellow
    }
}

Section 'Hyperstack MCP HTTP endpoint'
foreach ($port in @(8080, 8081, 3000, 4000)) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:$port/" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        Write-Host "  ✓ localhost:$port responding (status $($r.StatusCode))" -ForegroundColor Green
    } catch {
        Write-Host "  · localhost:$port no response" -ForegroundColor DarkGray
    }
}

Section 'SSH config'
$sshConfig = "$env:USERPROFILE\.ssh\config"
if (Test-Path $sshConfig) {
    $entries = (Get-Content $sshConfig | Select-String -Pattern '^Host\s+(.+)$').Matches |
        ForEach-Object { $_.Groups[1].Value }
    Write-Host "  ✓ $sshConfig" -ForegroundColor Green
    $entries | ForEach-Object { Write-Host "    Host: $_" -ForegroundColor Gray }
    $hsHosts = $entries | Where-Object { $_ -match '(?i)hyper|l40|gpu|vps|nexgen' }
    if ($hsHosts) {
        Write-Host "  ⚠ Possible Hyperstack-related hosts:" -ForegroundColor Yellow
        $hsHosts | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    }
} else {
    Write-Host "  · $sshConfig (missing)" -ForegroundColor DarkGray
}

Section 'PowerShell history'
$histFile = "$env:USERPROFILE\AppData\Roaming\Microsoft\Windows\PowerShell\PSReadLine\ConsoleHost_history.txt"
if (Test-Path $histFile) {
    $hits = Get-Content $histFile | Select-String -Pattern '(?i)hyperstack|nexgen|infrahub|l40' |
        Select-Object -Last 10
    if ($hits) {
        Write-Host "  ✓ Recent matching commands:" -ForegroundColor Green
        $hits | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    } else {
        Write-Host "  · No hyperstack-related commands in history" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  · history file missing" -ForegroundColor DarkGray
}

Section 'Summary'
$missing = @()
if (-not $env:HYPERSTACK_API_KEY -and -not (Test-Path "$PSScriptRoot\..\.env")) {
    $missing += "HYPERSTACK_API_KEY (set env var or create .env at repo root)"
}
if ($missing.Count -eq 0) {
    Write-Host "  ✓ All required credentials seem available." -ForegroundColor Green
    Write-Host "  Next: .\scripts\release-on-vps.ps1" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Missing:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "  How to set HYPERSTACK_API_KEY:" -ForegroundColor Cyan
    Write-Host "    `$env:HYPERSTACK_API_KEY = 'your-key-here'  # current session only"
    Write-Host "    [Environment]::SetEnvironmentVariable('HYPERSTACK_API_KEY','your-key','User')  # persistent"
    Write-Host "    Or create $PSScriptRoot\..\.env with line: HYPERSTACK_API_KEY=your-key"
}
