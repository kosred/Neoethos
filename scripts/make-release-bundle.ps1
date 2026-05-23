# scripts/make-release-bundle.ps1
#
# Build a self-contained "double-click and go" folder for NeoEthos.
# Output: `dist/NeoEthos/` containing
#
#   NeoEthos.exe                ← the ONE entry point (user double-clicks this)
#   flutter_windows.dll         ← Flutter engine DLL (alongside the UI exe)
#   data/                       ← Flutter assets bundle
#   config.yaml                 ← backend settings
#   bin/
#     neoethos-app.exe          ← Rust backend (hidden from operator;
#                                 auto-spawned by the UI exe)
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
    [string]$Destination = 'dist\NeoEthos',
    # When -Lite is set, the bundle ships WITHOUT the Gemma GGUF —
    # the NSIS installer downloads it at install time via
    # `scripts/gemma_model_install.nsh`. Total bundle drops from
    # ~5.5 GB back to ~250 MB so the installer .exe stays under
    # GitHub Releases' 2 GB per-asset cap. Default behaviour (no
    # -Lite) is the "Full" offline-testing bundle that includes
    # the GGUF inline — handy for local AI Helper verification
    # without internet access during install.
    [switch]$Lite
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

# 3. Copy Flutter bits. The Flutter build output uses whichever name
#    the CMakeLists.txt's `BINARY_NAME` was set to. We pinned that to
#    "NeoEthos", so the source filename is NeoEthos.exe; if a previous
#    build under the old name is still in the dir we accept either.
Write-Host "Copying Flutter shell from $flutterDir"
$srcUiExe = Join-Path $flutterDir 'NeoEthos.exe'
if (-not (Test-Path $srcUiExe)) {
    # Fallback to the pre-rename binary name.
    $srcUiExe = Join-Path $flutterDir 'forex_flutter_ui.exe'
}
if (-not (Test-Path $srcUiExe)) {
    throw "Could not find NeoEthos.exe or forex_flutter_ui.exe in $flutterDir - re-run flutter build windows --$Profile."
}
Copy-Item -Path $srcUiExe -Destination (Join-Path $dst 'NeoEthos.exe')
Copy-Item -Path (Join-Path $flutterDir 'flutter_windows.dll') -Destination $dst
Copy-Item -Path (Join-Path $flutterDir 'data') -Destination $dst -Recurse

# 4. Copy Rust backend into a `bin/` subfolder so it stays out of the
#    operator's way. BackendSupervisor's lookup #1 is exactly this
#    `<exe-dir>/bin/neoethos-app.exe` path — the operator sees one
#    NeoEthos.exe at the top of the bundle and nothing else.
$binDir = Join-Path $dst 'bin'
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
Write-Host "Copying neoethos-app.exe from $rustExe into bin/"
Copy-Item -Path $rustExe -Destination $binDir

# 5. Copy config.yaml + branding. config.yaml is what the backend
#    loads as `Settings::from_yaml("config.yaml")` from its CWD;
#    BackendSupervisor pins the CWD to the dir that contains it.
Copy-Item -Path (Join-Path $repoRoot 'config.yaml') -Destination $dst
Copy-Item -Path (Join-Path $repoRoot 'assets') -Destination $dst -Recurse
Copy-Item -Path (Join-Path $repoRoot 'LICENSE') -Destination $dst
Copy-Item -Path (Join-Path $repoRoot 'README.md') -Destination $dst -ErrorAction SilentlyContinue

# 6. Bundle the Gemma GGUF if it's already on disk under the repo's
#    `resources/models/`. The user explicitly asked that the model
#    ship inside the bundle so the operator never has to download it
#    by hand ("κακος το εχουμε αφησει στο χρηστη"). If the GGUF is
#    missing we still create the empty slot and warn loudly — the
#    bundle is still usable for chart/exec/discovery, but AI Helper +
#    News will show "rebuild with --features gemma-backend OR run
#    scripts/fetch-gemma-model.ps1" until the file lands.
$bundleModelDir = Join-Path $dst 'resources\models'
New-Item -ItemType Directory -Force -Path $bundleModelDir | Out-Null

if ($Lite) {
    Write-Host "Lite mode: skipping GGUF copy. NSIS installer will fetch it during install." -ForegroundColor Cyan
    # We still copy the LICENSE-gemma so end users see the upstream
    # attribution even before the download lands.
    $gemmaLicenseSrc = Join-Path $repoRoot 'LICENSE-gemma'
    if (Test-Path $gemmaLicenseSrc) {
        Copy-Item -Path $gemmaLicenseSrc -Destination $bundleModelDir
    }
} else {
    $repoModelDir = Join-Path $repoRoot 'resources\models'
    $ggufFiles = if (Test-Path $repoModelDir) {
        Get-ChildItem -Path $repoModelDir -Filter '*.gguf' -File -ErrorAction SilentlyContinue
    } else { @() }

    if ($ggufFiles.Count -gt 0) {
        foreach ($gguf in $ggufFiles) {
            $sizeGB = [math]::Round($gguf.Length / 1GB, 2)
            Write-Host ("Copying GGUF: {0} ({1} GB)" -f $gguf.Name, $sizeGB) -ForegroundColor Green
            Copy-Item -Path $gguf.FullName -Destination $bundleModelDir
        }
        # License compliance: Gemma is Google's license and requires
        # the license file + attribution to travel with the weights.
        $gemmaLicenseSrc = Join-Path $repoRoot 'LICENSE-gemma'
        if (Test-Path $gemmaLicenseSrc) {
            Copy-Item -Path $gemmaLicenseSrc -Destination $bundleModelDir
        } else {
            # No license file yet — emit a placeholder so the bundle is
            # never shipped without notice. Compliance scaffolding lives
            # in the repo as `LICENSE-gemma`; document the missing-file
            # case here loudly.
            Write-Warning "LICENSE-gemma not found at $gemmaLicenseSrc — bundle ships without Gemma license attribution. Fix before publishing."
        }
    } else {
        Write-Warning ("No .gguf found in {0}. AI Helper + News will be disabled in this bundle. To include them, run scripts/fetch-gemma-model.ps1 and re-bundle (or use -Lite to defer to NSIS install-time fetch)." -f $repoModelDir)
    }
}

# 7. Quick verification.
$bundleExe = Join-Path $dst 'NeoEthos.exe'
$bundleBackend = Join-Path $dst 'bin\neoethos-app.exe'
$bundleConfig = Join-Path $dst 'config.yaml'
$bundleGguf = Get-ChildItem -Path $bundleModelDir -Filter '*.gguf' -ErrorAction SilentlyContinue | Select-Object -First 1

Write-Host ""
Write-Host "Bundle contents:"
Get-ChildItem $dst | Format-Table Name, @{Name='Size'; Expression={if ($_.PSIsContainer) { 'dir' } else { $_.Length }}} -AutoSize

# Total bundle size — relevant because the GGUF makes it jump from ~250 MB
# to ~5.5 GB; the user should see this stat before publishing.
$totalBytes = (Get-ChildItem $dst -Recurse -File | Measure-Object -Property Length -Sum).Sum
$totalGB = [math]::Round($totalBytes / 1GB, 2)

Write-Host ""
Write-Host "Sanity check:"
Write-Host ("  NeoEthos.exe exists       : " + (Test-Path $bundleExe))
Write-Host ("  bin/neoethos-app.exe      : " + (Test-Path $bundleBackend))
Write-Host ("  config.yaml exists        : " + (Test-Path $bundleConfig))
Write-Host ("  Gemma GGUF bundled        : " + ($null -ne $bundleGguf))
if ($bundleGguf) {
    Write-Host ("    → {0} ({1} GB)" -f $bundleGguf.Name, [math]::Round($bundleGguf.Length / 1GB, 2))
}
Write-Host ("  Total bundle size         : {0} GB" -f $totalGB)
Write-Host ""
Write-Host "Done. Double-click $bundleExe to launch."
