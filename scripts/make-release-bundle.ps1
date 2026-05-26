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
#   (no model files — AI Helper runs against a ChatGPT subscription)
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

# 3. Copy Flutter bits. The Flutter build output uses whichever name
#    the CMakeLists.txt's `BINARY_NAME` was set to. We pinned that to
#    "NeoEthos", so the source filename is NeoEthos.exe; if a previous
#    build under the old name is still in the dir we accept either.
Write-Host "Copying Flutter shell from $flutterDir"
$srcUiExe = Join-Path $flutterDir 'NeoEthos.exe'
if (-not (Test-Path $srcUiExe)) {
    # Flutter's CMakeLists pins BINARY_NAME as lowercase "neoethos" today;
    # earlier rev was "NeoEthos". Either is fine — we rename to NeoEthos.exe
    # at the destination.
    $srcUiExe = Join-Path $flutterDir 'neoethos.exe'
}
if (-not (Test-Path $srcUiExe)) {
    # Fallback to the pre-rename binary name.
    $srcUiExe = Join-Path $flutterDir 'forex_flutter_ui.exe'
}
if (-not (Test-Path $srcUiExe)) {
    throw "Could not find NeoEthos.exe / neoethos.exe / forex_flutter_ui.exe in $flutterDir - re-run flutter build windows --$Profile."
}
Copy-Item -Path $srcUiExe -Destination (Join-Path $dst 'NeoEthos.exe')
# 2026-05-26 fix: previously only `flutter_windows.dll` was copied, but
# `flutter build windows --release` also emits one DLL per Flutter plugin
# (e.g. `url_launcher_windows_plugin.dll`). Loading the main exe with a
# missing plugin DLL exits with 0xC0000135 STATUS_DLL_NOT_FOUND before
# the BackendSupervisor ever runs — the UI window appears for ~50 ms and
# disappears with no error popup. Bundle every *.dll the Flutter build
# produced so plugins resolve at startup.
Copy-Item -Path (Join-Path $flutterDir '*.dll') -Destination $dst
Copy-Item -Path (Join-Path $flutterDir 'data') -Destination $dst -Recurse

# 4. Copy Rust backend into a `bin/` subfolder so it stays out of the
#    operator's way. BackendSupervisor's lookup #1 is exactly this
#    `<exe-dir>/bin/neoethos-app.exe` path — the operator sees one
#    NeoEthos.exe at the top of the bundle and nothing else.
$binDir = Join-Path $dst 'bin'
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
Write-Host "Copying neoethos-app.exe from $rustExe into bin/"
Copy-Item -Path $rustExe -Destination $binDir

# 4b. Tree-models runtime DLLs (task #236, 2026-05-25). The catboost-rust
#     and xgboost-sys crates download pre-built native libraries during
#     `cargo build --release`; the build-cargo-release.ps1 script has
#     already copied them next to neoethos-app.exe. We bundle them into
#     `bin/` so the dynamic loader finds them via the standard
#     "same-dir-as-exe" search step when CatBoostExpert / XGBoostExpert
#     initialise at runtime.
#
#     End-user contract: NO native runtime install required. CatBoost
#     + XGBoost + LightGBM (static-linked) all ship inside this bundle.
$dllSourceDir = Split-Path -Parent $rustExe
foreach ($dll in @('catboostmodel.dll', 'xgboost.dll')) {
    $src = Join-Path $dllSourceDir $dll
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $binDir -Force
        $sizeMB = [math]::Round((Get-Item $src).Length / 1MB, 1)
        Write-Host "  + $dll ($sizeMB MB) → bin/"
    } else {
        # catboost.dll absent = the build script didn't run or the
        # cargo build didn't include `tree-models`. xgboost.dll absent
        # is usually fine (static-linked variant). We warn but don't
        # fail — the operator can still ship without these and the
        # ensemble runs with the local-fallback tree experts.
        Write-Warning "$dll not found next to $rustExe — make sure scripts\build-cargo-release.ps1 ran with the gpu-vulkan + tree-models features. Bundle will be missing this expert."
    }
}

# 4a. Operator-confusion guard. The `bin/` folder ships a clear
#     "do not run" marker AND has the Hidden + System attributes set
#     so File Explorer hides it by default. Real-world report: even
#     the project owner double-clicked the backend exe by accident
#     and saw the binary "die silently" because cwd lacks config.yaml.
#     The Win32 help dialog in `show_double_click_help_dialog_if_orphaned`
#     catches the curious user who still drills in; this attribute +
#     README catches the casual user before they even see the file.
$donotRunPath = Join-Path $binDir 'DO-NOT-RUN.txt'
@'
⚠️  DO NOT RUN ANY .EXE IN THIS FOLDER  ⚠️

This folder contains the NeoEthos BACKEND SERVER.
It is meant to be auto-started by the NeoEthos app, NOT clicked directly.

WHAT TO DO:
→ Go back one folder up to "NeoEthos\"
→ Double-click "NeoEthos.exe" (250 KB, the small one with the icon)
→ The app will auto-start everything in this folder for you

If you double-click "neoethos-app.exe" directly:
- No window appears (the backend has no UI of its own)
- The backend tries to start but has no config in this dir
- You see "nothing happens" — but the process either errored out
  silently or is running invisibly on port 7423

This folder is normally HIDDEN from File Explorer.
You're only seeing it because "Show hidden files" is enabled.
'@ | Set-Content -Path $donotRunPath -Encoding UTF8

# Set Hidden + System attribute so default File Explorer settings
# don't show this folder at all. The operator only sees NeoEthos.exe
# + config.yaml + assets/ at the bundle root.
(Get-Item $binDir -Force).Attributes = 'Hidden, System, Directory'
Write-Host "Marked bin/ as Hidden+System; dropped DO-NOT-RUN.txt marker."

# 5. Copy config.yaml + branding. config.yaml is what the backend
#    loads as `Settings::from_yaml("config.yaml")` from its CWD;
#    BackendSupervisor pins the CWD to the dir that contains it.
Copy-Item -Path (Join-Path $repoRoot 'config.yaml') -Destination $dst
Copy-Item -Path (Join-Path $repoRoot 'assets') -Destination $dst -Recurse
Copy-Item -Path (Join-Path $repoRoot 'LICENSE') -Destination $dst
Copy-Item -Path (Join-Path $repoRoot 'README.md') -Destination $dst -ErrorAction SilentlyContinue

# 6. (Gemma GGUF bundling removed — AI Helper now uses a ChatGPT
#    subscription via neoethos-codex OAuth, no local model file is
#    needed. The bundle stays ~250 MB regardless.)

# 7. Quick verification.
$bundleExe = Join-Path $dst 'NeoEthos.exe'
$bundleBackend = Join-Path $dst 'bin\neoethos-app.exe'
$bundleConfig = Join-Path $dst 'config.yaml'

Write-Host ""
Write-Host "Bundle contents:"
Get-ChildItem $dst | Format-Table Name, @{Name='Size'; Expression={if ($_.PSIsContainer) { 'dir' } else { $_.Length }}} -AutoSize

$totalBytes = (Get-ChildItem $dst -Recurse -File | Measure-Object -Property Length -Sum).Sum
$totalMB = [math]::Round($totalBytes / 1MB, 2)

Write-Host ""
Write-Host "Sanity check:"
Write-Host ("  NeoEthos.exe exists       : " + (Test-Path $bundleExe))
Write-Host ("  bin/neoethos-app.exe      : " + (Test-Path $bundleBackend))
Write-Host ("  config.yaml exists        : " + (Test-Path $bundleConfig))
Write-Host ("  Total bundle size         : {0} MB" -f $totalMB)
Write-Host ""
Write-Host "Done. Double-click $bundleExe to launch."
