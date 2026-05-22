# scripts/make-release-bundle.ps1
#
# Build a self-contained "double-click and go" folder for NeoEthos.
# Output: `dist/NeoEthos/` containing
#
#   forex_flutter_ui.exe        ← the UI (this is what the user clicks)
#   flutter_windows.dll         ← Flutter engine DLL
#   data/                       ← Flutter assets bundle
#   neoethos-app.exe            ← Rust backend (auto-spawned by Flutter)
#   config.yaml                 ← backend settings
#   assets/branding/*.png       ← icons / splash
#   resources/models/           ← optional: Gemma GGUF lives here
#
# Operator workflow: copy the whole `dist/NeoEthos/` folder anywhere,
# double-click `forex_flutter_ui.exe`. The Flutter `BackendSupervisor`
# finds the co-located `neoethos-app.exe`, spawns it on port 7423, and
# the UI talks to it. No installer required for testing.
#
# The packaged NSIS installer (cargo-packager → installer_no_paid_certs_strategy.md)
# wraps this same layout in an .exe installer that adds Start-menu
# shortcuts; this script is the dev-mode equivalent of "what the
# installer drops on disk".

[CmdletBinding()]
param(
    [string]$Profile = 'release',     # 'release' or 'debug'
    [string]$Destination = 'dist\NeoEthos'
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$dst = Join-Path $repoRoot $Destination

Write-Host "Building NeoEthos release bundle from $repoRoot"
Write-Host "  profile     : $Profile"
Write-Host "  destination : $dst"

# 1. Verify the artefacts exist. We don't (re)build here — the operator
#    runs `cargo build -p neoethos-app --release` and
#    `flutter build windows --release` first; this script just collates.
$rustExe = Join-Path $repoRoot ("target\$Profile\neoethos-app.exe")
$flutterProfileDir = if ($Profile -eq 'release') { 'Release' } else { 'Debug' }
$flutterDir = Join-Path $repoRoot ("experiments\forex-flutter-ui\build\windows\x64\runner\$flutterProfileDir")

foreach ($p in @($rustExe, $flutterDir)) {
    if (-not (Test-Path $p)) {
        throw "Required artefact missing: $p`nBuild with: cargo build -p neoethos-app --$Profile + flutter build windows --$Profile"
    }
}

# 2. Clean previous bundle.
if (Test-Path $dst) {
    Write-Host "Cleaning previous bundle at $dst"
    Remove-Item -Recurse -Force $dst
}
New-Item -ItemType Directory -Force -Path $dst | Out-Null

# 3. Copy Flutter bits (forex_flutter_ui.exe + DLL + data/).
Write-Host "Copying Flutter shell from $flutterDir"
Copy-Item -Path (Join-Path $flutterDir 'forex_flutter_ui.exe') -Destination $dst
Copy-Item -Path (Join-Path $flutterDir 'flutter_windows.dll')  -Destination $dst
Copy-Item -Path (Join-Path $flutterDir 'data') -Destination $dst -Recurse

# 4. Copy Rust backend side-by-side. BackendSupervisor's first lookup
#    is "next to the Flutter exe" — this is exactly what makes that
#    work in production.
Write-Host "Copying neoethos-app.exe from $rustExe"
Copy-Item -Path $rustExe -Destination $dst

# 5. Copy config.yaml + branding. config.yaml is what the backend
#    loads as `Settings::from_yaml("config.yaml")` from its CWD;
#    BackendSupervisor pins the CWD to the dir that contains it.
Copy-Item -Path (Join-Path $repoRoot 'config.yaml') -Destination $dst
Copy-Item -Path (Join-Path $repoRoot 'assets') -Destination $dst -Recurse
Copy-Item -Path (Join-Path $repoRoot 'LICENSE') -Destination $dst
Copy-Item -Path (Join-Path $repoRoot 'README.md') -Destination $dst -ErrorAction SilentlyContinue

# 6. Carve out a place for the Gemma GGUF. The AI Helper screen tells
#    the user to drop the model file here; pre-creating the dir makes
#    the install hint actionable.
New-Item -ItemType Directory -Force -Path (Join-Path $dst 'resources\models') | Out-Null

# 7. Quick verification.
$bundleExe = Join-Path $dst 'forex_flutter_ui.exe'
$bundleBackend = Join-Path $dst 'neoethos-app.exe'
$bundleConfig = Join-Path $dst 'config.yaml'

Write-Host ""
Write-Host "Bundle contents:"
Get-ChildItem $dst | Format-Table Name, @{Name='Size'; Expression={if ($_.PSIsContainer) { 'dir' } else { $_.Length }}} -AutoSize

Write-Host ""
Write-Host "Sanity check:"
Write-Host ("  forex_flutter_ui.exe exists : " + (Test-Path $bundleExe))
Write-Host ("  neoethos-app.exe exists     : " + (Test-Path $bundleBackend))
Write-Host ("  config.yaml exists          : " + (Test-Path $bundleConfig))
Write-Host ""
Write-Host "Done. Double-click $bundleExe to launch."
