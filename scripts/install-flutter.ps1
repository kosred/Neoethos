# install-flutter.ps1
#
# Idempotent Flutter SDK installer για Windows. Δοκιμάζει με σειρά:
#   1. winget (αν είναι διαθέσιμο)
#   2. scoop  (αν είναι διαθέσιμο)
#   3. manual download από storage.googleapis.com
#
# Μετά verify με `flutter --version` + `flutter doctor` + εκτελεί
# `flutter config --enable-windows-desktop` ώστε το desktop target
# να είναι έτοιμο για `flutter create .` στο
# experiments/forex-flutter-ui/.
#
# Disk: ~3 GB. Αν C: < 5 GB free, abort.

[CmdletBinding()]
param(
    [string]$InstallDir = "$env:LOCALAPPDATA\flutter",
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Step($n, $msg) { Write-Host "`n=== Step $n · $msg ===" -ForegroundColor Cyan }
function Info($msg)     { Write-Host "  $msg" -ForegroundColor Gray }
function OK($msg)       { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Fail($msg)     { Write-Host "  ✗ $msg" -ForegroundColor Red; throw $msg }

# Disk safety
Step 0 'Disk safety'
$free = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
Info "C: drive free: $free GB"
if ($free -lt 5) { Fail "Need ≥5 GB free; have $free GB." }
OK "Disk OK"

# Already installed?
$existing = Get-Command flutter -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    OK "Flutter already on PATH: $($existing.Source)"
    Info (flutter --version 2>&1 | Select-String 'Flutter' | Select-Object -First 1)
    return
}

# --- Try winget ---
Step 1 'Try winget'
$winget = Get-Command winget -ErrorAction SilentlyContinue
if ($winget) {
    Info "winget found; installing Flutter.Flutter..."
    try {
        winget install --id=Flutter.Flutter -e --silent --accept-package-agreements --accept-source-agreements 2>&1 | Out-Host
        $f = Get-Command flutter -ErrorAction SilentlyContinue
        if ($f) {
            OK "winget install succeeded"
            $installedVia = 'winget'
            $true
        } else {
            Warn "winget exited 0 but flutter not on PATH"
        }
    } catch {
        Warn "winget failed: $_"
    }
} else {
    Info "winget not available"
}

# --- Try scoop ---
if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
    Step 2 'Try scoop'
    $scoop = Get-Command scoop -ErrorAction SilentlyContinue
    if ($scoop) {
        Info "scoop found; installing flutter..."
        try {
            scoop bucket add extras 2>&1 | Out-Null
            scoop install flutter 2>&1 | Out-Host
            $f = Get-Command flutter -ErrorAction SilentlyContinue
            if ($f) {
                OK "scoop install succeeded"
                $installedVia = 'scoop'
            }
        } catch {
            Warn "scoop failed: $_"
        }
    } else {
        Info "scoop not available"
    }
}

# --- Manual download fallback ---
if (-not (Get-Command flutter -ErrorAction SilentlyContinue)) {
    Step 3 'Manual download from storage.googleapis.com'

    if (-not (Test-Path $InstallDir)) {
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    }

    # Discover the latest stable release URL via releases_windows.json
    $manifestUrl = 'https://storage.googleapis.com/flutter_infra_release/releases/releases_windows.json'
    Info "Fetching release manifest..."
    $manifest = Invoke-RestMethod -Uri $manifestUrl -UseBasicParsing
    $stableHash = $manifest.current_release.stable
    $stableRelease = $manifest.releases | Where-Object { $_.hash -eq $stableHash } | Select-Object -First 1
    if (-not $stableRelease) { Fail "Could not resolve latest stable Flutter release." }
    $zipUrl = "https://storage.googleapis.com/flutter_infra_release/releases/$($stableRelease.archive)"
    $zipName = Split-Path $stableRelease.archive -Leaf
    $zipPath = Join-Path $env:TEMP $zipName

    Info "Latest stable: $($stableRelease.version)"
    Info "Download URL : $zipUrl"
    Info "Target dir   : $InstallDir"

    if (-not (Test-Path $zipPath) -or $Force) {
        Info "Downloading (~700 MB zip)..."
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
    } else {
        Info "Zip already present at $zipPath (use -Force to re-download)"
    }

    Info "Extracting to $InstallDir ..."
    Expand-Archive -Path $zipPath -DestinationPath $InstallDir -Force
    # The zip contains a top-level `flutter/` directory.
    $flutterBin = Join-Path $InstallDir 'flutter\bin'
    if (-not (Test-Path (Join-Path $flutterBin 'flutter.bat'))) {
        Fail "Extraction failed; flutter.bat not found at $flutterBin"
    }

    # Add to PATH (user-level, persistent)
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath -notlike "*$flutterBin*") {
        Info "Adding $flutterBin to user PATH..."
        $newPath = if ([string]::IsNullOrEmpty($userPath)) { $flutterBin } else { "$userPath;$flutterBin" }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    }
    $env:Path = "$env:Path;$flutterBin"
    OK "Manual install complete: $flutterBin"
    $installedVia = 'manual'
}

# --- Verify ---
Step 4 'Verify'
$flutter = Get-Command flutter -ErrorAction Stop
$version = & flutter --version 2>&1
OK "$($flutter.Source)"
$version | ForEach-Object { Info $_ }

Step 5 'Enable Windows desktop'
flutter config --enable-windows-desktop | Out-Host

Step 6 'flutter doctor'
flutter doctor 2>&1 | Out-Host

Step 7 'Bootstrap forex-flutter-ui'
$flutterUi = Join-Path $PSScriptRoot '..\experiments\forex-flutter-ui' | Resolve-Path
Push-Location $flutterUi
try {
    if (-not (Test-Path 'windows')) {
        Info "Running flutter create . --platforms windows..."
        flutter create . --platforms windows --org com.neoethos 2>&1 | Out-Host
    }
    Info "flutter pub get..."
    flutter pub get 2>&1 | Out-Host
    OK "forex-flutter-ui bootstrapped"
} finally { Pop-Location }

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  Flutter ready." -ForegroundColor Green
Write-Host "  Next: cd experiments\forex-flutter-ui ; flutter test" -ForegroundColor Green
Write-Host "        then: flutter run -d windows" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
