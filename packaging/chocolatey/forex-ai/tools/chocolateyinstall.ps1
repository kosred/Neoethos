# chocolateyinstall.ps1 — Chocolatey install hook for forex-ai.
#
# Spec: https://docs.chocolatey.org/en-us/create/functions/install-chocolateyzippackage
# Strategy ref: docs/audits/research/installer_no_paid_certs_strategy.md §1.5.2
#
# Downloads the official GitHub Releases tarball, verifies the SHA-256, and
# extracts the binaries under $toolsDir so Chocolatey's shim-gen can wire up
# `forex-app` and `forex-cli` onto the user's PATH.

$ErrorActionPreference = 'Stop'

$toolsDir = "$(Split-Path -parent $MyInvocation.MyCommand.Definition)"

$packageArgs = @{
    packageName    = 'forex-ai'
    fileType       = 'exe'
    url64bit       = 'https://github.com/kosred/forex-ai/releases/download/v0.4.14/forex-app_0.4.14_x64-setup.exe'
    checksum64     = 'DB83A13345AD653501F9C66C894C1B7C8A0D477AA2FBFE4C119BCD442DC8E834'
    checksumType64 = 'sha256'
    silentArgs     = '/S'
    validExitCodes = @(0)
}

# Install-ChocolateyPackage downloads the NSIS installer, verifies SHA-256,
# and runs it silently. Aborts on checksum mismatch.
Install-ChocolateyPackage @packageArgs

# Optional: register an Add/Remove Programs entry pointing at the bin dir.
# Chocolatey ships its own ARP entry for the package itself, so this is
# usually unnecessary; left as a no-op stub for future use.




