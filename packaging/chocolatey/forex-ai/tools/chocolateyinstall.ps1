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
    unzipLocation  = $toolsDir
    url64bit       = 'https://github.com/kosred/forex-ai/releases/download/v0.4.8/forex-ai-v0.4.8-windows-x86_64-setup.exe'
    checksum64     = 'E759C4BA7E124250A22D34AD1757403E39ECDF4EF011A5B47C1C8BA138198090'
    checksumType64 = 'sha256'
}

# Install-ChocolateyZipPackage handles download + SHA-256 verification +
# extraction in one call. If the checksum does not match it aborts.
Install-ChocolateyZipPackage @packageArgs

# Optional: register an Add/Remove Programs entry pointing at the bin dir.
# Chocolatey ships its own ARP entry for the package itself, so this is
# usually unnecessary; left as a no-op stub for future use.

