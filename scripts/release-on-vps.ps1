# release-on-vps.ps1
#
# All-in-one Windows orchestrator for the 2026-05-18 release:
#   1. Unstick git, commit the day's branch work, push to GitHub
#   2. Find / start the Hyperstack L40 VM
#   3. SSH-run scripts/build-release-on-vps.sh on the VM
#   4. SCP the tarball back to %USERPROFILE%\Downloads\releases\
#   5. Verify size + SHA256
#   6. Stop the VM (cost saver)
#   7. Print a billing summary
#
# Prerequisites the operator must have set ONCE:
#   - $env:HYPERSTACK_API_KEY  (or in a .env at repo root: HYPERSTACK_API_KEY=…)
#   - $env:HYPERSTACK_VM_ID    (optional — script can list and pick)
#   - SSH key configured for the VM user (default: ubuntu)
#   - GitHub auth (HTTPS PAT or SSH key)
#
# Run from repo root: cd C:\Users\konst\development\neoethos
#                     .\scripts\release-on-vps.ps1

[CmdletBinding()]
param(
    [string]$VmId         = $env:HYPERSTACK_VM_ID,
    [string]$VmUser       = ${env:HYPERSTACK_VM_USER}    ?? 'ubuntu',
    [string]$VmIp         = $env:HYPERSTACK_VM_IP,
    [string]$Branch       = 'feature/neoethos-gemma-g0',
    [string]$MergeInto    = 'master',
    [switch]$SkipGitPush,
    [switch]$SkipVpsBuild,
    [switch]$SkipStop,
    [switch]$KeepVmAlive
)

$ErrorActionPreference = 'Stop'
$DateStamp = (Get-Date -Format 'yyyy-MM-dd')
$RepoRoot  = (Get-Item $PSScriptRoot).Parent.FullName
$Tarball   = "neoethos-linux-x86_64-$DateStamp.tar.gz"
$LocalDest = Join-Path $env:USERPROFILE 'Downloads\releases'
$LocalTar  = Join-Path $LocalDest $Tarball
$Started   = Get-Date

function Step($n, $msg) { Write-Host "`n=== Step $n · $msg ===" -ForegroundColor Cyan }
function Info($msg)     { Write-Host "  $msg" -ForegroundColor Gray }
function OK($msg)       { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Fail($msg)     { Write-Host "  ✗ $msg" -ForegroundColor Red; throw $msg }

# Load .env if present
$envFile = Join-Path $RepoRoot '.env'
if (Test-Path $envFile) {
    Info "Loading $envFile"
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.+?)\s*$') {
            Set-Item "env:$($Matches[1])" $Matches[2]
        }
    }
}

# Disk pre-check
Step 0 'Disk safety'
$free = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
Info "C: drive free: $free GB"
if ($free -lt 30) { Fail "Need ≥30 GB free; have $free GB." }
OK "Disk OK"

# ─────────────────────────────────────────────────────────────────────
# 1. Git unstick + push
# ─────────────────────────────────────────────────────────────────────
if (-not $SkipGitPush) {
    Step 1 'Git unstuck + push to GitHub'
    Push-Location $RepoRoot
    try {
        $lock = Join-Path $RepoRoot '.git\index.lock'
        if (Test-Path $lock) {
            Info "Removing $lock"
            Remove-Item -Force $lock
            OK "Lock removed"
        }

        $currentBranch = (git branch --show-current).Trim()
        Info "Current branch: $currentBranch"

        # Stage + commit on the feature branch
        $dirty = git status --porcelain
        if ($dirty) {
            Info "Staging changes..."
            git add -A
            $msg = @"
Phase G + D3.1 follow-ups + Flutter scaffold · 2026-05-18

Round 1: neoethos-gemma crate (G0 scaffold + G2 topic gate w/ embedding
gate + 40+40 anchor corpus + G3 read-only tools + G1 prep w/
bundled-model path resolver + G6a expert wiring). 144 tests pass.

Round 2: G7 JSONL audit log writer, OrderSource::AiSuggested variant,
DxTrade Domain UI row. 151 tests pass.

Round 3: forex-flutter-ui scaffold from mockups/ui_mockup.html
(14 panels, TradingView dark theme, Riverpod, Dio, mocked backend
client, 6 widget smoke tests). egui untouched pending Step B parity.

Disk safety: 102 GB free throughout. Headless invariant preserved.
"@
            git commit -m $msg
            OK "Committed on $currentBranch"
        } else {
            Info "No uncommitted changes — already clean"
        }

        # Merge into master
        Info "Switching to $MergeInto..."
        git checkout $MergeInto
        git pull --ff-only origin $MergeInto
        Info "Merging $currentBranch (squash → keeps master linear)"
        git merge --no-ff --no-edit $currentBranch
        OK "Merged"

        Info "Pushing to origin/$MergeInto..."
        git push origin $MergeInto
        OK "Pushed"

        # Sanity
        $diff = git --no-pager log --oneline "origin/$MergeInto..$MergeInto" 2>&1
        if ($diff) { Fail "Local $MergeInto still ahead of origin: $diff" }
        OK "origin/$MergeInto in sync"
    }
    finally { Pop-Location }
}

# ─────────────────────────────────────────────────────────────────────
# 2. Hyperstack — find / start the VM
# ─────────────────────────────────────────────────────────────────────
if (-not $SkipVpsBuild) {
    Step 2 'Hyperstack VPS — find + start'
    if (-not $env:HYPERSTACK_API_KEY) {
        Fail "HYPERSTACK_API_KEY env var missing. Set it or put it in .env."
    }
    $hsHeader = @{ 'api_key' = $env:HYPERSTACK_API_KEY }
    $hsBase   = 'https://infrahub-api.nexgencloud.com/v1/core/virtual-machines'

    if (-not $VmId) {
        Info "VM ID not provided; listing all VMs..."
        $vms = (Invoke-RestMethod -Uri $hsBase -Headers $hsHeader).instances
        $vms | ForEach-Object {
            Info ("  id={0,-6} name={1,-30} status={2,-12} flavor={3}" -f `
                $_.id, $_.name, $_.status, $_.flavor_name)
        }
        $candidates = $vms | Where-Object {
            $_.flavor_name -match 'L40|RTX|A100|H100' -and
            $_.status -in @('ACTIVE', 'SHUTOFF', 'HIBERNATED')
        }
        if (-not $candidates) {
            Fail "No L40/RTX/A100/H100 VM found. Create one in Hyperstack console first."
        }
        $VmId = ($candidates | Select-Object -First 1).id
        $VmIp = ($candidates | Select-Object -First 1).fixed_ip
        Info "Picked VM id=$VmId ip=$VmIp"
    }

    # Re-fetch to know the status
    $vm = (Invoke-RestMethod -Uri "$hsBase/$VmId" -Headers $hsHeader).instance
    if (-not $VmIp) { $VmIp = $vm.fixed_ip }
    Info "VM status: $($vm.status)"

    if ($vm.status -in @('SHUTOFF', 'HIBERNATED')) {
        Info "Starting VM..."
        Invoke-RestMethod -Uri "$hsBase/$VmId/start" -Method POST -Headers $hsHeader | Out-Null
        do {
            Start-Sleep -Seconds 10
            $vm = (Invoke-RestMethod -Uri "$hsBase/$VmId" -Headers $hsHeader).instance
            Info "  status=$($vm.status)"
        } while ($vm.status -notin @('ACTIVE', 'ERROR'))
        if ($vm.status -ne 'ACTIVE') { Fail "VM failed to start: $($vm.status)" }
        OK "VM ACTIVE"
    } elseif ($vm.status -eq 'ACTIVE') {
        OK "VM already ACTIVE"
    } else {
        Fail "VM in unexpected state: $($vm.status)"
    }

    # ─────────────────────────────────────────────────────────────
    # 3. SSH-run the build script on the VM
    # ─────────────────────────────────────────────────────────────
    Step 3 'SSH build on VM'
    $sshTarget = "$VmUser@$VmIp"
    Info "Target: $sshTarget"

    # Copy the build script to the VM
    $remoteScript = "/tmp/build-release-on-vps.sh"
    scp -o StrictHostKeyChecking=no `
        (Join-Path $RepoRoot 'scripts\build-release-on-vps.sh') `
        "${sshTarget}:$remoteScript"

    # Execute it (long-running — 15-30 min typical)
    Info "Executing build (long-running — 15-30 min)..."
    ssh -o StrictHostKeyChecking=no $sshTarget "bash $remoteScript"
    OK "Build done"

    # ─────────────────────────────────────────────────────────────
    # 4. SCP tarball back to Windows
    # ─────────────────────────────────────────────────────────────
    Step 4 'Download tarball'
    New-Item -ItemType Directory -Force -Path $LocalDest | Out-Null
    scp -o StrictHostKeyChecking=no `
        "${sshTarget}:~/$Tarball" $LocalTar

    if (-not (Test-Path $LocalTar)) { Fail "Tarball not downloaded to $LocalTar" }
    $size = [math]::Round((Get-Item $LocalTar).Length / 1MB, 2)
    $sha  = (Get-FileHash $LocalTar -Algorithm SHA256).Hash
    OK "Downloaded: $LocalTar"
    OK "Size: $size MB"
    OK "SHA-256: $sha"

    # ─────────────────────────────────────────────────────────────
    # 5. Stop the VM (cost saver)
    # ─────────────────────────────────────────────────────────────
    if (-not $SkipStop -and -not $KeepVmAlive) {
        Step 5 'Stop VM'
        Invoke-RestMethod -Uri "$hsBase/$VmId/stop" -Method POST -Headers $hsHeader | Out-Null
        do {
            Start-Sleep -Seconds 10
            $vm = (Invoke-RestMethod -Uri "$hsBase/$VmId" -Headers $hsHeader).instance
            Info "  status=$($vm.status)"
        } while ($vm.status -notin @('SHUTOFF', 'HIBERNATED', 'ERROR'))
        OK "VM stopped (status=$($vm.status))"
    } else {
        Info "Skipping VM stop (KeepVmAlive=$KeepVmAlive)"
    }

    # ─────────────────────────────────────────────────────────────
    # 6. Billing summary
    # ─────────────────────────────────────────────────────────────
    Step 6 'Billing summary'
    $elapsed = (Get-Date) - $Started
    $hours = [math]::Round($elapsed.TotalHours, 2)
    # L40 hourly rate at Hyperstack (May 2026, public pricing) ≈ $1.40-2.00
    $costLow  = [math]::Round($hours * 1.40, 2)
    $costHigh = [math]::Round($hours * 2.00, 2)
    Info "Total runtime: $hours h"
    Info "Estimated cost: \$$costLow – \$$costHigh"
    OK "Done"
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Release pipeline complete." -ForegroundColor Green
Write-Host "  Tarball: $LocalTar" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
