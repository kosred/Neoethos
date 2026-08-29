param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]] $CargoArguments
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$probe = Join-Path ([System.IO.Path]::GetTempPath()) "neoethos-resolve-host-$PID.exe"
$probePdb = [System.IO.Path]::ChangeExtension($probe, '.pdb')
$previousLocation = Get-Location

try {
    Set-Location -LiteralPath $repo
    & rustc --edition 2024 -D warnings scripts/build/resolve_host.rs -o $probe
    if ($LASTEXITCODE -ne 0) {
        throw "build-host resolver compilation failed with exit code $LASTEXITCODE"
    }

    $hostEvidence = @(& $probe)
    if ($LASTEXITCODE -ne 0) {
        throw "build-host resolver failed with exit code $LASTEXITCODE"
    }

    $plan = @{}
    foreach ($line in $hostEvidence) {
        if ($line -match '^([^=]+)=(.*)$' -and $Matches[1] -ne 'gpu_device') {
            $plan[$Matches[1]] = $Matches[2]
        }
    }
    $availableThreads = $plan['available_parallelism']
    $workerLimit = $plan['automatic_worker_limit']
    $acceleratorMode = $plan['accelerator_mode']
    $cudaArchitectures = $plan['cuda_architectures']
    if ($availableThreads -notmatch '^[1-9][0-9]*$' -or
        $workerLimit -notmatch '^[1-9][0-9]*$') {
        throw "invalid build-host plan: available=$availableThreads workers=$workerLimit mode=$acceleratorMode cuda_architectures=$cudaArchitectures"
    }

    $env:CARGO_BUILD_JOBS = $workerLimit
    if ($acceleratorMode -eq 'cpu_only' -and $cudaArchitectures -eq 'none') {
        Remove-Item Env:NEOETHOS_CUDA_ARCHS -ErrorAction SilentlyContinue
    }
    elseif ($acceleratorMode -eq 'nvidia' -and
        $cudaArchitectures -match '^[1-9][0-9]*(;[1-9][0-9]*)*$') {
        $env:NEOETHOS_CUDA_ARCHS = $cudaArchitectures
    }
    else {
        throw "invalid accelerator selection in build-host plan: mode=$acceleratorMode cuda_architectures=$cudaArchitectures"
    }
    $hostEvidence | Write-Output
    & cargo @CargoArguments
    $cargoExitCode = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $previousLocation
    Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $probePdb -Force -ErrorAction SilentlyContinue
}

exit $cargoExitCode
