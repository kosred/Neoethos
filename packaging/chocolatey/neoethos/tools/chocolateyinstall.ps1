$ErrorActionPreference = 'Stop'

$packageArgs = @{
    packageName    = 'neoethos'
    fileType       = 'exe'
    url64bit       = 'https://github.com/kosred/neoethos/releases/download/v0.4.20/neoethos_0.4.20_x64-setup.exe'
    softwareName   = 'neoethos*'
    checksum64     = ''
    checksumType64 = 'sha256'
    silentArgs     = '/S'
    validExitCodes = @(0)
}

Install-ChocolateyPackage @packageArgs
