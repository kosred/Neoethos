# scripts/fetch-gemma-model.ps1
#
# One-time download of the bundled Gemma 4 E4B Uncensored Aggressive GGUF.
# Run BEFORE building the NSIS / debian installer — the packager looks for
# the .gguf file at the path below.
#
# Disk requirement: ~5 GB free for the model; additional ~20 GB free for
# the release build itself. Aborts if C: drive free space is below 50 GB.

[CmdletBinding()]
param(
    [string]$Quant = 'Q4_K_M',  # Q4_K_M (5.0 GB) or Q5_K_M (5.4 GB)
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$Repo = 'HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive'
$Filename = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-$Quant.gguf"
$Url = "https://huggingface.co/$Repo/resolve/main/$Filename"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$Target = Join-Path $RepoRoot "resources/models/$Filename"
$TargetDir = Split-Path $Target -Parent

if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
}

# Disk safety pre-check
$FreeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
Write-Host "C: drive free: $FreeGB GB" -ForegroundColor Cyan
if ($FreeGB -lt 50) {
    throw "C: drive has only $FreeGB GB free. Need >= 50 GB for safe model + cargo build."
}

if ((Test-Path $Target) -and -not $Force) {
    $size = (Get-Item $Target).Length / 1GB
    Write-Host "Model already at $Target ($([math]::Round($size, 2)) GB). Use -Force to re-download." -ForegroundColor Yellow
    exit 0
}

Write-Host "Downloading $Filename..." -ForegroundColor Green
Write-Host "  From: $Url"
Write-Host "  To:   $Target"
Write-Host ""

$ProgressPreference = 'SilentlyContinue'  # speeds up Invoke-WebRequest
$tmpPath = "$Target.tmp"
try {
    Invoke-WebRequest -Uri $Url -OutFile $tmpPath -UseBasicParsing
    Move-Item -Force $tmpPath $Target
} finally {
    if (Test-Path $tmpPath) { Remove-Item $tmpPath -Force }
}

$sizeGB = [math]::Round((Get-Item $Target).Length / 1GB, 2)
$sha = (Get-FileHash -Algorithm SHA256 -Path $Target).Hash
Write-Host ""
Write-Host "Downloaded successfully:" -ForegroundColor Green
Write-Host "  Size:   $sizeGB GB"
Write-Host "  SHA256: $sha"
Write-Host "  Path:   $Target"
