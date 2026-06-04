# scripts/build-installer.ps1
#
# One-command "produce a downloadable Setup.exe" pipeline.
#
# Steps:
#   1. Verify NSIS (makensis.exe) is available on PATH.
#   2. Run make-release-bundle.ps1 to populate dist/NeoEthos/.
#   3. Run makensis on installer/neoethos.nsi → dist/NeoEthos-Setup-*.exe.
#
# After this script completes, the deliverable is ONE file:
#   dist\NeoEthos-Setup-0.4.41.exe
#
# That's what the end-user downloads + double-clicks. The installer:
#   - Asks where to install (default: %ProgramFiles%\NeoEthos)
#   - Drops everything from dist/NeoEthos/ into the install dir
#   - Creates Start Menu shortcut → NeoEthos.exe
#   - Creates Desktop shortcut → NeoEthos.exe (optional, ticked by default)
#   - Adds NeoEthos to Add/Remove Programs
#   - Offers to launch NeoEthos.exe on the final wizard page
#
# End-user NEVER sees `bin\neoethos-app.exe`. The `bin/` folder ships
# with Hidden+System attributes; File Explorer hides it by default.
#
# Prerequisites:
#   - NSIS installed: https://nsis.sourceforge.io/Download
#     (winget install NSIS.NSIS  OR  choco install nsis)
#   - The `cargo build --release` + `flutter build windows --release`
#     artefacts must already exist (the bundle script verifies this).

[CmdletBinding()]
param(
    # 'release' (default) or 'debug'. Debug builds a slower, less-compressed
    # bundle that's still useful for end-to-end smoke testing the installer
    # wizard itself.
    [string]$Profile = 'release'
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Write-Host "NeoEthos installer build" -ForegroundColor Cyan
Write-Host "  repo root : $repoRoot"
Write-Host "  profile   : $Profile"
Write-Host ""

# ── 1. NSIS check ────────────────────────────────────────────────────────────
$makensis = Get-Command makensis -ErrorAction SilentlyContinue
if (-not $makensis) {
    Write-Host "makensis.exe not found on PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install NSIS via one of:" -ForegroundColor Yellow
    Write-Host "  winget install NSIS.NSIS"
    Write-Host "  choco install nsis"
    Write-Host "  https://nsis.sourceforge.io/Download (manual)"
    Write-Host ""
    Write-Host "After install, re-open the terminal so PATH picks up makensis." -ForegroundColor Yellow
    throw "NSIS is required to build the installer."
}
Write-Host "[OK] NSIS found: $($makensis.Source)" -ForegroundColor Green

# ── 2. Run the bundle script ─────────────────────────────────────────────────
$bundleScript = Join-Path $repoRoot 'scripts\make-release-bundle.ps1'
if (-not (Test-Path $bundleScript)) {
    throw "make-release-bundle.ps1 not found at $bundleScript"
}

Write-Host ""
Write-Host "==> Running make-release-bundle.ps1..." -ForegroundColor Cyan
& $bundleScript -Profile $Profile
# 2026-05-26: `$LASTEXITCODE` is only set by native executables. PowerShell
# scripts (like make-release-bundle.ps1) leave it at its previous value if
# they don't `exit` explicitly - meaning a stale non-zero code from earlier
# in the parent script triggered a false failure. Guard with `-gt 0` AND
# allow `$null` (= never set in this run, i.e. clean) to pass.
if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -gt 0) {
    throw "make-release-bundle.ps1 failed with exit code $LASTEXITCODE"
}

# ── 3. Compile the NSIS installer ────────────────────────────────────────────
$nsiScript = Join-Path $repoRoot 'installer\neoethos.nsi'
if (-not (Test-Path $nsiScript)) {
    throw "installer/neoethos.nsi not found at $nsiScript"
}

Write-Host ""
Write-Host "==> Compiling installer/neoethos.nsi..." -ForegroundColor Cyan

# /V3 = info-level output (less noisy than the default /V4 trace).
# /NOCD prevents makensis from changing dir; we already have absolute paths
# in the .nsi via `..\` references and want them resolved relative to the
# .nsi's location which is the default behaviour when not /NOCD'd, so let
# makensis cd into installer/.
# Note: the .nsi uses `..\LICENSE` etc - relative to installer/ - so we
# cd there explicitly and let makensis resolve from cwd.
Push-Location (Split-Path $nsiScript -Parent)
try {
    & $makensis.Source '/V3' (Split-Path $nsiScript -Leaf)
    if ($LASTEXITCODE -ne 0) { throw "makensis failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

# ── 4. Verify output ─────────────────────────────────────────────────────────
$installerExe = Join-Path $repoRoot 'dist\NeoEthos-Setup-0.4.41.exe'
if (-not (Test-Path $installerExe)) {
    throw "Installer was not produced at $installerExe - check makensis output above for errors."
}
$installerSize = (Get-Item $installerExe).Length
$sizeMB = [math]::Round($installerSize / 1MB, 1)

Write-Host ""
Write-Host "[OK] Installer built: $installerExe" -ForegroundColor Green
Write-Host "     Size: $sizeMB MB"
Write-Host ""
Write-Host "End-user workflow:" -ForegroundColor Cyan
Write-Host "  1. Download NeoEthos-Setup-0.4.41.exe (one file)."
Write-Host "  2. Double-click to install. Follow the wizard."
Write-Host "  3. Launch via Start Menu or Desktop shortcut."
Write-Host "  4. Never see bin\neoethos-app.exe - it's hidden."
