# scripts/build-cargo-release.ps1
#
# Build neoethos-app.exe with ALL ML features enabled - tree-models
# (LightGBM/XGBoost/CatBoost/sklears) + gpu-vulkan (wgpu Vulkan backend).
#
# **End-user contract**: this script bakes EVERY native dependency into
# the produced binary + sidecar DLLs. The end-user installs ONE installer
# (.exe) and opens a broker account - NOTHING else (no CMake, no MSVC, no
# LLVM, no Vulkan SDK, no CUDA toolkit). All build-time toolchains are
# the developer's problem, not the user's.
#
# Build-time prerequisites on this machine (the DEVELOPER's box):
#   - Visual Studio Build Tools 2019+ with "Desktop development with C++"
#     workload (provides cl.exe, link.exe, nmake.exe + Windows SDK)
#   - LLVM/Clang installed at C:\Program Files\LLVM (provides libclang.dll
#     that `bindgen` uses to parse the LightGBM + CatBoost C headers)
#   - CMake 3.18+ on PATH (used by lightgbm3-sys to build the vendored
#     LightGBM C++ source - `vendor/lightgbm3-sys/lightgbm/`)
#   - Vulkan SDK 1.3+ at C:\VulkanSDK\<version>\ (for gpu-vulkan feature
#     - provides vulkan-1.lib + glslc.exe for shader compilation)
#   - Internet access for the catboost-rust and xgb crates to download
#     their pre-built native binaries from github.com/catboost/catboost
#     and the xgboost upstream release tags (~80 MB total, cached)
#
# Output:
#   target\release\neoethos-app.exe         - the binary
#   target\release\catboostmodel.dll        - CatBoost runtime (sidecar)
#   target\release\xgboost.dll              - XGBoost runtime (sidecar, if built)
#
# Next steps after this script:
#   scripts\make-release-bundle.ps1 → produces dist\NeoEthos\
#   scripts\build-installer.ps1     → produces dist\NeoEthos-Setup-*.exe
#
# Both downstream scripts already know to pick up the DLLs sitting next
# to neoethos-app.exe and bundle them into the installer.

[CmdletBinding()]
param(
    # Cargo feature flags to enable. Default: gpu-vulkan covers the
    # cross-vendor GPU path (NVIDIA + AMD + Intel iGPU via wgpu). The
    # tree-models feature is wired through neoethos-app's default
    # dependency spec on neoethos-models so we don't have to enumerate
    # it here.
    [string]$Features = 'gpu-vulkan',

    # Set to skip the actual cargo build - useful for verifying the
    # env-detection plumbing without paying the 10-25 min build cost.
    [switch]$SkipBuild,

    # Override Vulkan SDK install dir. Default: auto-detect newest
    # version under C:\VulkanSDK\.
    [string]$VulkanSdkDir,

    # Override LLVM install dir. Default: C:\Program Files\LLVM.
    [string]$LlvmDir = 'C:\Program Files\LLVM'
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Write-Host "NeoEthos cargo release build" -ForegroundColor Cyan
Write-Host "  repo root : $repoRoot"
Write-Host "  features  : $Features"
Write-Host ""

# ── 1. Locate Visual Studio Build Tools (provides MSVC + nmake) ──────────────
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw @"
vswhere.exe not found at $vswhere

vswhere is Microsoft's standard "where is Visual Studio installed?" utility
and ships with every VS install. If it's missing, no VS install exists.

Install Visual Studio Build Tools 2019 or 2022:
  winget install Microsoft.VisualStudio.2022.BuildTools `
    --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools"

That installs cl.exe + link.exe + nmake.exe + Windows SDK headers
without the full VS IDE. Then re-run this script.
"@
}

Write-Host "==> Locating VS install via vswhere..." -ForegroundColor Cyan
# **2026-05-25 fix**: query the Workload ID (`VCTools`) rather than
# the specific VS2022-only Component ID (`VC.Tools.x86.x64`). The
# workload ID is the same for VS 2019 + VS 2022 BuildTools, so the
# script auto-discovers either. Also add `-prerelease` so beta VS
# installs are picked up if that's all the user has.
$vsInstall = & $vswhere -latest -prerelease `
    -requires Microsoft.VisualStudio.Workload.VCTools `
    -property installationPath

# **2026-05-25 fallback**: VS 2019 BuildTools manual installs sometimes
# omit the Workload manifest from vswhere's index (the workload runs
# but isn't catalogued). If vswhere comes back empty, scan the standard
# install roots directly. This is the same fallback the cc-rs crate
# uses internally.
#
# **2026-05-25 — preview-version guardrail**: explicitly EXCLUDE the VS
# 2026 Insiders path (`...\Microsoft Visual Studio\18\BuildTools`). Its
# MSVC 14.51.36231 ships vectorized STL symbols
# (`__std_find_trivial_1`, `__std_rotate`, `_Thrd_sleep_for`, etc.) that
# the released VC runtime DLLs DO NOT export yet. Mixing it with any
# downstream linker that resolves against the stable runtime produces
# `LNK2019 unresolved external` for every vectorized algorithm
# instantiation in lightgbm3-sys + xgboost. Stick to stable 2022 BuildTools.
if (-not $vsInstall) {
    Write-Host "vswhere returned nothing - falling back to filesystem probe" -ForegroundColor Yellow
    # Ordered newest-stable-first: 2022 stable wins over 2019 stable.
    # Insiders/preview VS18 path is INTENTIONALLY OMITTED — see comment above.
    $candidates = @(
        'C:\Program Files\Microsoft Visual Studio\2022\BuildTools',
        'C:\Program Files\Microsoft Visual Studio\2022\Community',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\Community',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise'
    )
    foreach ($candidate in $candidates) {
        if (Test-Path (Join-Path $candidate 'VC\Auxiliary\Build\vcvars64.bat')) {
            $vsInstall = $candidate
            Write-Host "[OK] Found VS at: $vsInstall (filesystem probe)" -ForegroundColor Green
            break
        }
    }
}

# **2026-05-25 — also reject the preview VS18 path if vswhere returned it**.
# `vswhere -latest -prerelease` happily picks the Insiders install. The
# subsequent vcvars64.bat injects 14.51.36231 LIB paths and the link
# step fails on vectorized STL symbols. If we're handed that path, fall
# back to the filesystem probe so we get a stable VS 2022/2019 instead.
if ($vsInstall -and $vsInstall -match '\\Microsoft Visual Studio\\18\\') {
    Write-Host "vswhere returned VS 2026 Insiders ($vsInstall)" -ForegroundColor Yellow
    Write-Host "  → its MSVC 14.51 preview vectorized-STL symbols break LightGBM link." -ForegroundColor Yellow
    Write-Host "  → falling back to stable VS 2022/2019 probe..." -ForegroundColor Yellow
    $vsInstall = $null
    $stableRoots = @(
        'C:\Program Files\Microsoft Visual Studio\2022\BuildTools',
        'C:\Program Files\Microsoft Visual Studio\2022\Community',
        'C:\Program Files\Microsoft Visual Studio\2022\Professional',
        'C:\Program Files\Microsoft Visual Studio\2022\Enterprise',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\Community',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional',
        'C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional',
        'C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise'
    )
    foreach ($candidate in $stableRoots) {
        if (Test-Path (Join-Path $candidate 'VC\Auxiliary\Build\vcvars64.bat')) {
            $vsInstall = $candidate
            Write-Host "[OK] Found stable VS at: $vsInstall" -ForegroundColor Green
            break
        }
    }
}
if (-not $vsInstall) {
    throw @"
vswhere found no VS install with the MSVC C++ toolset (component
Microsoft.VisualStudio.Component.VC.Tools.x86.x64).

You may have Visual Studio installed but WITHOUT the "Desktop development
with C++" workload. Open Visual Studio Installer and add it, or run:

  winget install Microsoft.VisualStudio.2022.BuildTools `
    --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools"
"@
}
Write-Host "[OK] VS install at: $vsInstall" -ForegroundColor Green

# ── 2. Load vcvars64.bat env into this PowerShell session ────────────────────
# Trick: run vcvars64.bat inside cmd.exe, dump the resulting env with
# `set`, then ingest each KEY=VAL line into PowerShell's env: drive.
# This is the canonical way to "activate" MSVC inside a non-cmd shell.
$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars (VS install layout is unexpected)."
}

Write-Host "==> Loading MSVC env via vcvars64.bat..." -ForegroundColor Cyan
# Use `cmd /c` to chain vcvars64 then `set` - captures the populated
# environment AFTER vcvars has done its thing. The `>nul` swallows the
# "Environment initialized for: ..." banner so we only get KEY=VAL lines.
$envDump = & cmd.exe /c "`"$vcvars`" >nul && set"
$envCount = 0
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        $envCount++
    }
}
Write-Host "[OK] $envCount env vars loaded (PATH/INCLUDE/LIB/etc.)" -ForegroundColor Green

# Sanity-check the key tools made it onto PATH.
$nmake = Get-Command nmake -ErrorAction SilentlyContinue
$cl = Get-Command cl -ErrorAction SilentlyContinue
if (-not $nmake -or -not $cl) {
    throw "vcvars64 ran but cl.exe / nmake.exe still not on PATH - VS install is broken."
}
Write-Host "  cl.exe    : $($cl.Source)"
Write-Host "  nmake.exe : $($nmake.Source)"

# ── 3. Set LIBCLANG_PATH for bindgen ─────────────────────────────────────────
# bindgen (used by lightgbm3-sys and catboost-rust to parse C headers
# into Rust FFI bindings) looks up libclang.dll via this env var.
if (-not (Test-Path (Join-Path $LlvmDir 'bin\libclang.dll'))) {
    throw @"
libclang.dll not found at $LlvmDir\bin\libclang.dll

The `bindgen` crate needs libclang.dll to parse the LightGBM + CatBoost
C headers. Install LLVM from:

  winget install LLVM.LLVM

The default install path is C:\Program Files\LLVM. If you put it
elsewhere, pass -LlvmDir <path> to this script.
"@
}
$env:LIBCLANG_PATH = Join-Path $LlvmDir 'bin'
Write-Host "[OK] LIBCLANG_PATH = $env:LIBCLANG_PATH" -ForegroundColor Green

# ── 4. Set VULKAN_SDK for the gpu-vulkan feature ─────────────────────────────
if ($Features -match 'gpu-vulkan') {
    if (-not $VulkanSdkDir) {
        # Auto-detect newest install under C:\VulkanSDK\<version>\.
        $candidates = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        $VulkanSdkDir = ($candidates | Select-Object -First 1).FullName
    }
    if (-not $VulkanSdkDir -or -not (Test-Path (Join-Path $VulkanSdkDir 'Lib\vulkan-1.lib'))) {
        throw @"
Vulkan SDK not found.

The gpu-vulkan feature links against vulkan-1.lib + uses glslc.exe to
compile shaders. Install via:

  winget install KhronosGroup.VulkanSDK

Default install path is C:\VulkanSDK\<version>\. If you put it
elsewhere, pass -VulkanSdkDir <path> to this script.
"@
    }
    $env:VULKAN_SDK = $VulkanSdkDir
    Write-Host "[OK] VULKAN_SDK = $env:VULKAN_SDK" -ForegroundColor Green
}

# ── 5. Sanity check CMake (lightgbm3-sys uses it) ────────────────────────────
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    throw @"
cmake.exe not found on PATH after loading vcvars64.

Install CMake via:
  winget install Kitware.CMake

Then re-open the terminal so PATH picks up cmake.exe.
"@
}
Write-Host "[OK] cmake.exe : $($cmake.Source)" -ForegroundColor Green

if ($SkipBuild) {
    Write-Host ""
    Write-Host "[OK] Environment configured. Skipping build (-SkipBuild specified)." -ForegroundColor Yellow
    return
}

# ── 6. cargo build ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> cargo build --release -p neoethos-app --features $Features" -ForegroundColor Cyan
Write-Host "    (cold cache: ~20-30 minutes. Most of it is LightGBM C++ build."
Write-Host "    Warm cache: ~30 seconds. Coffee time on first run.)"
Write-Host ""

Push-Location $repoRoot
try {
    & cargo build --release -p neoethos-app --features $Features
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

# ── 7. Copy sidecar DLLs next to neoethos-app.exe ────────────────────────────
# The catboost-rust and xgboost-sys build.rs scripts download their
# native libraries into OUT_DIR (deep under target/release/build/). We
# need them sitting next to neoethos-app.exe at runtime, otherwise
# Windows' dynamic loader can't find them when LoadLibrary fires
# during the CatBoostExpert / XGBoostExpert init paths.
$releaseDir = Join-Path $repoRoot 'target\release'

Write-Host ""
Write-Host "==> Copying runtime DLLs next to neoethos-app.exe..." -ForegroundColor Cyan

$dst = Join-Path $releaseDir 'catboostmodel.dll'
# 2026-05-26 fix: search ONLY under `target/release/build/` (the
# catboost-rust build.rs OUT_DIR) so a stale copy already next to
# neoethos-app.exe from a previous run does not match itself first
# and trigger "Cannot overwrite the item ... with itself".
$catboostSearchRoot = Join-Path $releaseDir 'build'
$catboostDll = if (Test-Path $catboostSearchRoot) {
    Get-ChildItem $catboostSearchRoot -Recurse -Filter 'catboostmodel.dll' `
        -ErrorAction SilentlyContinue | Select-Object -First 1
} else { $null }
if ($catboostDll) {
    if ($catboostDll.FullName -ieq $dst) {
        Write-Host "  [skip] catboostmodel.dll already in place at $dst" -ForegroundColor DarkGray
    } else {
        Copy-Item -Path $catboostDll.FullName -Destination $dst -Force
        $sizeMB = [math]::Round((Get-Item $dst).Length / 1MB, 1)
        Write-Host "  [OK] catboostmodel.dll ($sizeMB MB) → $dst" -ForegroundColor Green
    }
} elseif (Test-Path $dst) {
    # Build did not produce a fresh DLL but a stale one exists next to
    # neoethos-app.exe — keep it (warm-cache re-runs hit this path).
    $sizeMB = [math]::Round((Get-Item $dst).Length / 1MB, 1)
    Write-Host "  [keep] catboostmodel.dll ($sizeMB MB) already at $dst (no fresh build output)" -ForegroundColor DarkGray
} else {
    Write-Warning "catboostmodel.dll not found anywhere under target/release/build/ - CatBoost expert (catboost + catboost_alt) will refuse to load at runtime."
}

# XGBoost: the xgboost-sys crate with `use_prebuilt_xgb` may produce
# a static .lib (static-linked, no DLL needed) or a dynamic .dll.
# Pick up the .dll if it exists.
$xgbDst = Join-Path $releaseDir 'xgboost.dll'
# 2026-05-26 fix: same self-copy guard as catboostmodel.dll above.
$xgbSearchRoot = Join-Path $releaseDir 'build'
$xgbDll = if (Test-Path $xgbSearchRoot) {
    Get-ChildItem $xgbSearchRoot -Recurse -Filter 'xgboost.dll' `
        -ErrorAction SilentlyContinue | Select-Object -First 1
} else { $null }
if ($xgbDll) {
    if ($xgbDll.FullName -ieq $xgbDst) {
        Write-Host "  [skip] xgboost.dll already in place at $xgbDst" -ForegroundColor DarkGray
    } else {
        Copy-Item -Path $xgbDll.FullName -Destination $xgbDst -Force
        $sizeMB = [math]::Round((Get-Item $xgbDst).Length / 1MB, 1)
        Write-Host "  [OK] xgboost.dll ($sizeMB MB) → $xgbDst" -ForegroundColor Green
    }
} elseif (Test-Path $xgbDst) {
    $sizeMB = [math]::Round((Get-Item $xgbDst).Length / 1MB, 1)
    Write-Host "  [keep] xgboost.dll ($sizeMB MB) already at $xgbDst (no fresh build output)" -ForegroundColor DarkGray
} else {
    Write-Host "  [info] xgboost.dll not found - assuming static-link build (the usual prebuilt case)."
}

# ── 8. Report ────────────────────────────────────────────────────────────────
$exe = Join-Path $releaseDir 'neoethos-app.exe'
if (-not (Test-Path $exe)) {
    throw "Build claimed success but $exe is missing."
}
$exeSizeMB = [math]::Round((Get-Item $exe).Length / 1MB, 1)

Write-Host ""
Write-Host "[OK] Release build complete." -ForegroundColor Green
Write-Host "     neoethos-app.exe : $exe ($exeSizeMB MB)"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. scripts\make-release-bundle.ps1   (collates Flutter UI + Rust backend)"
Write-Host "  2. scripts\build-installer.ps1       (produces dist\NeoEthos-Setup-*.exe)"
Write-Host ""
Write-Host "End-user prerequisites: NONE (modulo a broker / cTrader demo account)." -ForegroundColor Green
