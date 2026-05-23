$ErrorActionPreference = 'Stop'

$packageArgs = @{
    packageName    = 'neoethos'
    softwareName   = 'neoethos*'
    fileType       = 'exe'
    silentArgs     = '/S'
    validExitCodes = @(0)
}

$uninstall = Get-UninstallRegistryKey -SoftwareName $packageArgs.softwareName |
    Select-Object -First 1

if ($uninstall -and $uninstall.UninstallString) {
    $packageArgs.file = $uninstall.UninstallString.Trim('"')
    Uninstall-ChocolateyPackage @packageArgs
}
