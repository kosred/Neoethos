# chocolateyuninstall.ps1 — Chocolatey uninstall hook for forex-ai.
#
# Spec: https://docs.chocolatey.org/en-us/create/functions/uninstall-chocolateyzippackage
# Strategy ref: docs/audits/research/installer_no_paid_certs_strategy.md §1.5.2
#
# Removes the extracted zip artefacts. Chocolatey's shim-gen automatically
# unregisters the forex-app and forex-cli shims because the underlying
# binaries are gone after `Uninstall-ChocolateyZipPackage` completes.
#
# This hook does NOT delete user data (config, cache, logs) — those live
# under %APPDATA%/forex-ai and %LOCALAPPDATA%/forex-ai per
# installer_infrastructure_spec.md §8.2. Wizard §10 (migration / coexistence)
# expects user data to survive package uninstall.

$ErrorActionPreference = 'Stop'

$packageArgs = @{
    packageName = 'forex-ai'
    # Must match the URL used in chocolateyinstall.ps1 so Chocolatey can map
    # the package back to its extracted contents.
    zipFileName = 'forex-ai-v0.4.7-windows-x86_64.zip'
}

Uninstall-ChocolateyZipPackage @packageArgs
