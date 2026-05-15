# Installer Infrastructure Spec — forex-ai

> Status: research deliverable, no code changes implied.
> Author: research agent, 2026-05-15.
> Operator directive (verbatim, Greek): «Όλες οι εκδόσεις είτε desktop είτε CLI θα πρέπει να γίνονται εγκατάσταση και όχι portable.»
> English translation: *every release — desktop or CLI — must be installed, not run as a portable archive.*
> Constraint: the installer must be a guided wizard that automates the post-extract setup steps (directory creation, autostart hook, OAuth seeding, uninstall registration).
>
> This file replaces any prior "drop a zip, run the exe" distribution assumption.
> It only documents toolchains and contracts — implementation lives in follow-up tickets.

---

## Methodology and citation policy

Every concrete claim below is grounded in an upstream source. URLs are inline,
in the body of the section that depends on the fact. Where two sources
contradict, the most-authoritative one (vendor docs > project release > third-party
blog) wins and the discrepancy is called out.

The following primary sources were sweeped:

- cargo-wix README and crates.io page — <https://github.com/volks73/cargo-wix>, <https://docs.rs/crate/cargo-wix/latest>.
- WiX Toolset releases — <https://github.com/wixtoolset/wix/releases>, <https://docs.firegiant.com/wix/whatsnew/releasenotes/>.
- NSIS news / release — <https://sourceforge.net/p/nsis/news/2025/03/nsis-311-released/>.
- Inno Setup help / signing — <https://jrsoftware.org/ishelp/topic_setup_signtool.htm>.
- cargo-bundle README — <https://github.com/burtonageo/cargo-bundle>.
- create-dmg README — <https://github.com/sindresorhus/create-dmg>.
- cargo-deb README and crate — <https://github.com/kornelski/cargo-deb>, <https://docs.rs/crate/cargo-deb/latest>.
- cargo-generate-rpm README — <https://github.com/cat-in-136/cargo-generate-rpm>.
- AppImage docs — <https://docs.appimage.org/packaging-guide/optional/updates.html>, <https://github.com/AppImage/AppImageKit/wiki/FUSE>.
- Snapcraft docs — <https://documentation.ubuntu.com/snapcraft/stable/explanation/classic-confinement/>.
- Flatpak manifest docs — <https://docs.flatpak.org/en/latest/sandbox-permissions.html>.
- Tauri v2 distribute guides — <https://v2.tauri.app/distribute/>, <https://v2.tauri.app/distribute/windows-installer/>, <https://v2.tauri.app/plugin/updater/>, <https://v2.tauri.app/distribute/sign/windows/>.
- cargo-dist book — <https://axodotdev.github.io/cargo-dist/book/install.html>, <https://github.com/axodotdev/cargo-dist/blob/main/book/src/installers/msi.md>.
- cargo-packager README and docs — <https://github.com/crabnebula-dev/cargo-packager>, <https://docs.crabnebula.dev/packager/>.
- Sparkle docs — <https://sparkle-project.org/documentation/>, <https://sparkle-project.org/documentation/publishing/>, <https://sparkle-project.org/documentation/eddsa-migration/>.
- WinSparkle — <https://github.com/vslavik/winsparkle>.
- Apple Developer Program & notarization — <https://developer.apple.com/programs/whats-included/>, <https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution>, <https://developer.apple.com/news/upcoming-requirements/?id=11012023a>, <https://developer.apple.com/developer-id/>.
- macOS launchd / LaunchAgents — <https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html>, <https://launchd.info/>.
- XDG Base Directory Specification — <https://specifications.freedesktop.org/basedir/latest/>, <https://wiki.archlinux.org/title/XDG_Base_Directory>.
- systemd XDG autostart generator — <https://www.freedesktop.org/software/systemd/man/latest/systemd-xdg-autostart-generator.html>, <https://systemd.io/DESKTOP_ENVIRONMENTS/>.
- Microsoft Windows Known Folders — <https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid>.
- Microsoft Run / RunOnce registry keys — <https://learn.microsoft.com/en-us/windows/win32/setupapi/run-and-runonce-registry-keys>.
- Microsoft code signing options — <https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options>.
- Microsoft SignTool — <https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool>, <https://learn.microsoft.com/en-us/windows/win32/seccrypto/using-signtool-to-sign-a-file>.
- Microsoft MSIX — <https://learn.microsoft.com/en-us/windows/msix/>, <https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/msix-windows10-windows11>, <https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-prepare>.
- Azure Artifact Signing (formerly Trusted Signing) — <https://learn.microsoft.com/en-us/azure/artifact-signing/faq>, <https://azure.microsoft.com/en-us/products/artifact-signing>.
- Sectigo / DigiCert EV pricing — <https://www.sectigo.com/ssl-certificates-tls/code-signing>.
- WinGet manifest schema — <https://learn.microsoft.com/en-us/windows/package-manager/package/manifest>, <https://github.com/microsoft/winget-create>.
- Sigstore cosign — <https://docs.sigstore.dev/quickstart/quickstart-cosign/>, <https://github.com/sigstore/cosign>.
- cTrader Open API — <https://help.ctrader.com/open-api/>, <https://help.ctrader.com/open-api/proxies-endpoints/>, <https://help.ctrader.com/open-api/connection/>.

When this spec cites "the upstream README" without a deeper URL, it is the
GitHub-hosted README at the project's `main` branch as of 2026-05-15.

---

## §1 — Target platforms

forex-ai targets three operating-system families. Priority ordering reflects
cTrader's desktop bias — the cTrader desktop client is Windows-first, and the
typical retail FX trader runs Windows 10 or 11.

### 1.1 Windows 10 / 11 (x86_64) — **priority 1**

- **Minimum version**: Windows 10 1809 (build 17763) for native TLS 1.3 in
  Schannel, which keeps the `rustls` + `reqwest` stack consistent with the
  cTrader proxy's TLS requirement (see §1.4).
- **MSIX-only paths** (used as a *secondary*, store-bound distribution channel)
  require Windows 10 1607 (build 14393) or newer; per Microsoft's "MSIX on
  Windows 10 and Windows 11" page, "the Windows Application Packaging Project
  is supported on Windows 10, version 1607, and later" — see
  <https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-prepare>.
- **Architecture**: x86_64 only. ARM64 Windows is *not* a launch target — the
  catboost / lightgbm / xgboost vendored stacks (see workspace `vendor/`)
  ship native blobs for x86_64 only. ARM64 Windows is a deferred backlog item.

### 1.2 macOS 13 Ventura+ (x86_64 + Apple Silicon arm64) — **priority 2**

- **Minimum version**: macOS 13 (Ventura). Reason: `notarytool` is the only
  supported notarization client (Apple sunset `altool` on 2023-11-01 per
  <https://developer.apple.com/news/upcoming-requirements/?id=11012023a>), and
  the hardened-runtime entitlements model we depend on shipped in macOS 10.13.6.
  Ventura is the lowest version we test against in CI; older macOS may work but
  is unsupported.
- **Architecture**: two physical binaries:
  - `x86_64-apple-darwin` for Intel Macs (still ~10% of cTrader's macOS user
    base based on Spotware community polls).
  - `aarch64-apple-darwin` for Apple Silicon M1/M2/M3/M4.
  - The `.app` is shipped as a **universal binary** (`lipo`-merged) inside a
    single `.dmg` so the operator does not have to choose at download time.

### 1.3 Linux — **priority 3**

Distros targeted at release v1.0:

- Ubuntu 22.04 LTS, 24.04 LTS (deb).
- Debian 12 (deb, same package).
- Fedora 39, 40, 41 (rpm).
- Arch Linux rolling (Pacman / AUR mirror generated by `cargo-packager`).
- Any glibc 2.35+ system can run the AppImage (used as the universal fallback).

Architecture: `x86_64-unknown-linux-gnu` only at v1.0. `aarch64-unknown-linux-gnu`
is feasible on Ubuntu/Debian once CI runners are available, but is gated on the
same vendored-blob availability concern as Windows ARM64.

### 1.4 cTrader Open API platform constraints

The cTrader Open API itself does *not* impose an OS requirement on its clients —
"You can use any language to implement cTrader Open API" per
<https://help.ctrader.com/open-api/>. However, two network constraints apply
to every platform we ship to:

- The Protobuf endpoint is fixed at TCP **port 5035** with mandatory TLS (the
  page <https://help.ctrader.com/open-api/proxies-endpoints/> describes the
  endpoint, and <https://help.ctrader.com/open-api/connection/> states "The TCP
  client connection must use SSL, otherwise you will not be able to connect or
  interact with the API"). The installer wizard must therefore verify
  outbound TLS connectivity to `live.ctraderapi.com:5035` and
  `demo.ctraderapi.com:5035` as a pre-flight check (no firewall corp policy
  surprises after the user has finished the install).
- The cTID OAuth flow (web browser redirect to
  `https://connect.spotware.com/`) means the installer must register a
  custom-scheme URL handler (`forex-ai://oauth/callback`) on the host OS — see
  §6.4 below for the per-platform registration mechanic.

---

## §2 — Installer toolchains per platform

### 2.1 Windows candidates

#### 2.1.1 cargo-wix — *Rust wrapper over WiX*

- **License**: MIT or Apache 2.0 (dual). Per <https://github.com/volks73/cargo-wix>:
  "Dual-licensed under MIT or Apache 2.0".
- **Latest stable**: **0.3.9**, released 2025-03-13 (per crates.io and
  <https://docs.rs/crate/cargo-wix/latest>).
- **Toolset coupling**: cargo-wix runs on any host but the underlying WiX
  Toolset itself "is Windows only; thus this project is only useful when
  installed on a Windows machine" (README). It supports both **WiX 3.14.1
  (Legacy)** and **WiX 4+ (Modern)**; legacy is currently the default subcommand.
- **Workspace handling**: cargo-wix targets a single binary per invocation
  (`cargo wix --bin forex-app`). To ship a single MSI containing *both*
  `forex-app.exe` and `forex-cli.exe` (the operator's mandate — installed, not
  portable) we author a hand-written `wix/main.wxs` that includes both
  binaries as `<Component>` entries; the README documents the override.
- **Signing**: built-in. `cargo wix sign` wraps Microsoft's
  `signtool.exe sign /fd SHA256 /td SHA256 /tr <timestamp-url> <msi>`. The
  cargo-wix flag set is documented in 0.3.9 release notes ("fix for the sign
  subcommand arguments").
- **Auto-update**: not provided. Pair with WinSparkle (see §4.2).

#### 2.1.2 WiX Toolset 4+ / 5 / 6 / 7 directly

- **License**: MS-RL (Microsoft Reciprocal License). Per FireGiant: "use of
  the WiX Toolset requires an Open Source Maintenance Fee" — see
  <https://docs.firegiant.com/wix/>.
- **Latest stable**: **WiX 7.0.0** (released 2026-04-06, per
  <https://github.com/wixtoolset/wix/releases>). WiX 5.0.2 (2024-10-04) is the
  last stable in the v5 line; the toolchain is intentionally backwards
  compatible — "WiX v5 was made highly compatible with WiX v4, with most users
  able to switch with no code changes; WiX v6 is intentionally highly compatible
  with WiX v5" (FireGiant release notes).
- **Operator-relevant Open Source Maintenance Fee**: "if you use this project
  to generate revenue, the Open Source Maintenance Fee is required". forex-ai
  is operator-internal; this still triggers the fee. *Budget item.*

#### 2.1.3 Inno Setup

- **License**: free for any use (including commercial), per
  <https://jrsoftware.org/isinfo.php> — confirmed by community references.
- **Latest stable**: 6.x line; the upstream "What's new" page is
  <https://jrsoftware.org/files/is6.4-whatsnew.htm>.
- **Signing**: integrated, via `[Setup]` `SignTool` directive. Documented at
  <https://jrsoftware.org/ishelp/topic_setup_signtool.htm>: "The SignTool
  specifies the name and parameters of the Sign Tool to be used to digitally
  sign Setup (and Uninstall if SignedUninstaller is set to yes)". Example
  invocation: `signtool.exe sign /f certificate.pfx /p password /t <timestamp>`.
- **Inno Setup 6** introduced `ISSigTool.exe` for ECDSA P-256 file signatures,
  but per the upstream FAQ it "does not replace Microsoft's signtool.exe in any
  way and is not related to Authenticode Code Signing at all" — i.e. it
  cannot replace SignTool for SmartScreen compliance.

#### 2.1.4 NSIS

- **License**: zlib/libpng, fully free for commercial use.
- **Latest stable**: **NSIS 3.11** (released March 2025, per the SourceForge
  news feed <https://sourceforge.net/p/nsis/news/2025/03/nsis-311-released/>).
- **Strength**: tiny installer footprint, very flexible scripting language.
- **Weakness**: scripting language is bespoke and verbose; per-user vs
  per-machine logic is hand-rolled. Tauri's NSIS bundler abstracts this away
  (see §2.4).

#### 2.1.5 MSIX

- **License**: Microsoft proprietary, free to use; documented at
  <https://learn.microsoft.com/en-us/windows/msix/>.
- **Min OS**: Windows 10 1607.
- **Operator-relevant restriction**: per
  <https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-prepare>:
  "MSIX doesn't support per-user Windows services. MSIX supports session-0
  (per-machine) services running under one of the defined system accounts
  (LocalSystem, LocalService, or NetworkService)." This **rules MSIX out** for
  forex-ai's autonomous-trading watchdog, which is a per-user background
  process holding the user's OAuth refresh token. We do not ship via MSIX.
- WinGet manifest schema does include `msix` as one installer type
  (<https://learn.microsoft.com/en-us/windows/package-manager/package/manifest>),
  so if Microsoft Store distribution ever becomes desirable we revisit.

### 2.2 macOS candidates

#### 2.2.1 cargo-bundle

- **License**: MIT or Apache 2.0 (dual).
- **Status per upstream README** (<https://github.com/burtonageo/cargo-bundle>):
  "very early alpha", "the format of the `[package.metadata.bundle]` section may
  change". For a long-lived operator-internal tool we do **not** want
  alpha-grade tooling on the macOS path.
- **Output**: `.app` bundle, plus *experimental* `.deb` and `.msi` paths.
- **Decision**: rejected for production use. See §2.4 for the substitute.

#### 2.2.2 create-dmg (sindresorhus)

- **License**: MIT.
- **Latest stable**: **v8.1.0**, 2026-03-21.
- **What it does**: takes a `.app` and a background image, produces a polished
  DMG (window position, icon position, "Drag to /Applications" alias). Pure
  Node.js, macOS-only.
- **Notarization**: explicitly *not* automated — "you must notarize your DMG"
  separately. We script this via `xcrun notarytool` in CI.
- **Why we still want it**: the DMG UX (drag-to-Applications animation, EULA
  pane) is brand-defining and `pkgbuild`/`productbuild` do not produce DMGs at
  all — they produce `.pkg` installers, which we *also* ship for users who
  prefer the install-wizard model.

#### 2.2.3 pkgbuild + productbuild

- **License**: Apple toolchain bundled with Xcode Command Line Tools (free).
- **Workflow** per the search summary of
  <https://developer.apple.com/forums/thread/746354> and the manpage:
  1. `pkgbuild --root <staging> --identifier com.example.forexai --version
     <ver> --install-location /Applications component.pkg`.
  2. `productbuild --distribution Distribution.xml --resources Resources/
     --sign "Developer ID Installer: <name> (<team-id>)" forex-ai-<ver>.pkg`.
  3. `xcrun notarytool submit forex-ai-<ver>.pkg --apple-id ... --wait`.
  4. `xcrun stapler staple forex-ai-<ver>.pkg`.
- **Why we ship a .pkg in addition to a .dmg**: a .pkg is a *wizard* (welcome,
  license, destination, install) which is exactly what the operator directive
  requires. A DMG is a drag-install — by itself it does not satisfy the
  "installer wizard" rule. We default to the .pkg and offer the .dmg as a
  convenience download.

#### 2.2.4 Notarization workflow (Apple Notary Service)

- Apple Developer Program enrollment: **$99/yr** per
  <https://developer.apple.com/programs/whats-included/>: "the Apple Developer
  Program is 99 USD per membership year, or in local currency where available".
- `notarytool` is mandatory since 2023-11-01; the legacy `altool` was retired
  (<https://developer.apple.com/news/upcoming-requirements/?id=11012023a>).
- Hardened Runtime + entitlements must be declared at signing time. From the
  Apple "Notarizing macOS software before distribution" doc
  (<https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution>):
  "Apps must be signed with the Hardened Runtime capability enabled".
- We will need at minimum the following entitlements (declared in
  `forex-app.entitlements`):
  - `com.apple.security.network.client` — TLS to ctraderapi.com:5035.
  - `com.apple.security.cs.allow-jit` — `wgpu` / `burn-wgpu` Metal shader
    pipeline uses JIT on macOS.
  - `com.apple.security.cs.disable-library-validation` — required if we load
    `libcatboost.dylib` or other vendored CPU/GPU ML blobs that are signed by
    a different team-id (or unsigned).
  - The smaller the set, the better — every disabled hardening check is a
    potential review flag.

### 2.3 Linux candidates

#### 2.3.1 cargo-deb

- **License**: MIT.
- **Latest stable**: **3.7.0** (2026-05-02 per docs.rs).
- **Workspace handling**: select-by-crate using `cargo deb -p forex-app` or
  `--manifest-path`. To ship both `forex-app` and `forex-cli` we author a
  meta-package and use `[package.metadata.deb.assets]` to declare each binary
  with mode `755` to `/usr/bin/forex-app` and `/usr/bin/forex-cli`.
- **systemd integration**: the README documents
  `[package.metadata.deb.systemd-units]` which auto-installs a unit file and
  invokes `deb-systemd-helper` during postinst.
- **GPG signing**: cargo-deb does *not* sign the `.deb` itself. The standard
  workflow is `dpkg-sig --sign builder forex-ai.deb` after `cargo deb`. We
  bundle this in CI.

#### 2.3.2 cargo-generate-rpm

- **License**: MIT.
- **Latest stable**: **0.21.0** (2026-05-04 per docs.rs).
- **Approach**: pure-Rust implementation using the `rpm` crate; no `rpmbuild`
  dependency on the build host. Major win for a Linux-x86_64 CI runner.
- **GPG signing**: built-in via `--signing_key <path>` flag (README).
- **Dependency declaration**: `[package.metadata.generate-rpm.requires]`
  mirrors Cargo's syntax (`">= 3"`, `">= 1.2, < 3.4"`).

#### 2.3.3 AppImage

- **License**: MIT (AppImageKit) plus per-runtime licensing.
- **Runtime requirement**: FUSE2 (preferred) or FUSE3. Per
  <https://github.com/AppImage/AppImageKit/wiki/FUSE>, "AppImages support the
  `--appimage-extract` option for systems on which FUSE is not available", but
  the operator UX of "double-click to run" requires FUSE.
- **Signing**: GPG2 signatures *inside* the AppImage are supported by
  `appimagetool -s` per <https://docs.appimage.org/packaging-guide/optional/updates.html>.
- **Updates**: AppImageUpdate + zsync. The update-info string is embedded in
  the AppImage's ISO 9660 Volume Descriptor #1 field "Application Used", per
  <https://docs.appimage.org/packaging-guide/optional/updates.html>. Self-update
  is possible by bundling `AppImageUpdate` itself inside the AppImage.
- **Operator-relevant caveat**: AppImage is *self-contained*, which on the
  surface contradicts the "installed, not portable" rule. We mitigate by
  shipping AppImage **alongside** a launcher script that the user runs once;
  the launcher copies the AppImage into `~/.local/bin/`, installs an XDG
  `.desktop` entry, registers the `forex-ai://` URL scheme, and seeds
  `~/.config/forex-ai/`. After that one-time bootstrap the experience is
  indistinguishable from a deb/rpm install.

#### 2.3.4 Flatpak

- **License**: LGPL-2.1+ for the runtime; manifest is JSON or YAML.
- **Sandbox**: very strong by default. Per
  <https://docs.flatpak.org/en/latest/sandbox-permissions.html>, "Sandbox
  access is primarily configured through the finish-args section of the
  manifest file. By default, applications have no access to processes outside
  the sandbox, limited syscalls (apps can't use nonstandard network socket
  types or ptrace other processes), limited access to the session D-Bus
  instance, and no access to host services like X11, system D-Bus, or
  PulseAudio".
- **Operator-relevant blockers**:
  - No raw TCP-to-arbitrary-port unless `--share=network` is declared. (We
    can do this, but we then explain to the user why the sandbox is laxer.)
  - GPU access for `wgpu`/`burn-wgpu` requires `--device=dri`.
  - Persistent state must live under `~/.var/app/<app-id>/` — diverges from
    XDG.
- **Decision**: Flatpak is a *deferred* secondary channel. Not v1.0.

#### 2.3.5 Snap

- **License**: GPLv3 for snapcraft, proprietary store.
- **Confinement**: per
  <https://documentation.ubuntu.com/snapcraft/stable/explanation/classic-confinement/>,
  three modes exist: `strict`, `devmode`, `classic`. forex-ai needs raw network
  + GPU access + USB-token signing tooling, which works under `strict` with
  the right plugs (`network`, `home`, `opengl`) but is fiddly.
- **Operator-relevant point**: per the same docs, "For core24, snaps must use
  libraries by name or version as defined for Ubuntu 24.04 LTS, while core22
  snaps look for dependencies on the host system, which may result in app
  instability, unknown behavior, or crashing". core24 is the right target for
  v1.0.
- **Decision**: Snap is a *deferred* secondary channel — Canonical Store
  distribution is a future option, but the auto-update model (snapd manages
  it, not the operator) is incompatible with our "no silent unverified
  updates" rule (§4).

### 2.4 Cross-platform meta-tools

#### 2.4.1 Tauri 2 bundler

- **License**: MIT or Apache 2.0 (dual).
- **Latest stable**: Tauri 2.x line (the CHANGELOG at
  <https://github.com/tauri-apps/tauri/blob/main/crates/tauri-bundler/CHANGELOG.md>
  is the canonical version source; v2 is mature and recommended).
- **Output formats** per <https://v2.tauri.app/distribute/>: "deb", "rpm",
  "appimage", "nsis", "msi", "app", "dmg". This is essentially a superset of
  the per-platform tools above and is the *recommended* path **if** forex-app
  migrates from egui to Tauri per the UI/UX migration spec.
- **NSIS per-user vs per-machine**: per
  <https://v2.tauri.app/distribute/windows-installer/>: "The 'currentUser'
  mode is the default for the installer, installing the app in a directory
  that doesn't require Administrator access, with installer metadata saved
  under the HKCU registry path. 'perMachine' installs the app in the
  Program Files folder and requires Administrator access, with installer
  metadata saved under the HKLM registry path. 'both' mode allows the user to
  choose at install time". For forex-ai we ship `perMachine` because
  registry-based autostart and uninstall must work for any account on the
  machine; the wizard surfaces this as "Install for everyone on this PC".
- **Uninstall hooks**: `NSIS_HOOK_PREUNINSTALL` and `NSIS_HOOK_POSTUNINSTALL`
  per the Tauri docs. We use POSTUNINSTALL to clean cache / logs / data, with
  a confirmation prompt (§6).
- **Code signing** (Tauri 2):
  - Windows: `TAURI_WINDOWS_SIGNTOOL_PATH` env var points to `signtool.exe`,
    plus `tauri.conf.json > bundle.windows.certificateThumbprint`. Per
    <https://v2.tauri.app/distribute/sign/windows/>.
  - macOS: identity inferred from `APPLE_CERTIFICATE` env var or
    `bundle.macOS.signingIdentity`. Hardened-runtime entitlements declared
    in `tauri.conf.json`.

#### 2.4.2 cargo-dist (renamed to `dist`)

- **License**: MIT or Apache 2.0 (dual).
- **Latest stable**: **0.31.0**, released 2026-02-23.
- **What it does**: end-to-end release pipeline that "spins up machines for
  each platform you support" — GitHub Actions matrix, MSI/PKG/deb/AppImage
  via various back-ends, GitHub Release artifacts upload.
- **Installer formats** (per <https://axodotdev.github.io/cargo-dist/book/installers/>):
  - `shell` — `curl | sh` bootstrapper.
  - `powershell` — `iwr | iex` bootstrapper.
  - `msi` — uses the WiX v3 toolchain (per
    <https://github.com/axodotdev/cargo-dist/blob/main/book/src/installers/msi.md>:
    "MSI requires Windows and the WiX v3 toolchain to build").
  - `pkg` — macOS.
  - `homebrew` — Homebrew tap.
  - `npm` — npm package wrapper for the CLI.
- **Strength**: one config (`Cargo.toml [workspace.metadata.dist]`) drives the
  whole matrix.
- **Weakness**: MSI back-end is WiX **v3**, not v4+ — a fixable but real
  limitation. Apple notarization is *not* built in; the project asks the
  operator to script it.

#### 2.4.3 cargo-packager (CrabNebula)

- **License**: MIT or Apache 2.0.
- **Latest stable**: **0.11.8**, 2025-11-27.
- **Format matrix** (README):
  - macOS: `.dmg`, `.app`.
  - Linux: `.deb`, `.AppImage`, Pacman (`.tar.gz` + `PKGBUILD`).
  - Windows: NSIS (`.exe`), WiX MSI (`.msi`).
- **Updater**: `cargo-packager-updater` companion crate — see §4.4.
- **Signing**: macOS identity string (`Developer ID Application: NAME
  (TEAM_ID)`); recent releases fixed cert-handling regressions.
- **Operator advantage over cargo-dist**: same author/team as Tauri's
  bundler (CrabNebula picks up the Tauri-bundler crate as its core), so the
  format output is identical to what Tauri 2 would produce — *one* codebase
  to learn whether or not we migrate to Tauri.

### 2.5 Per-platform decision

| Platform | Primary toolchain | Secondary / fallback | Rationale |
|----------|------------------|----------------------|-----------|
| Windows  | **cargo-packager → NSIS + WiX/MSI** | Tauri 2 bundler if migration; cargo-wix if cargo-packager regresses | Single config, both installer types, integrated signing, updater. NSIS gives smaller download; MSI gives Group Policy compatibility for any corp users. |
| macOS    | **cargo-packager → .app, then `pkgbuild`+`productbuild` + create-dmg** | Tauri 2 bundler if migration | `.pkg` wizard satisfies the "installed, not portable" rule. `.dmg` is the secondary convenience channel. |
| Linux    | **cargo-deb (Debian/Ubuntu) + cargo-generate-rpm (Fedora) + cargo-packager → AppImage** | Tauri 2 bundler if migration | Native package format for each major distro family. AppImage is the catch-all for non-Ubuntu/Fedora systems. |
| Release orchestration | **cargo-dist / `dist`** | hand-rolled `release.yml` | One GitHub Actions matrix, signed-release-on-tag-push, GitHub Releases as the auth-source-of-truth. cargo-dist invokes the per-platform tools above. |

Justification — cargo-packager over cargo-bundle: cargo-bundle is "very early
alpha" per its own README; cargo-packager is on a 0.11 line with bi-weekly
releases and is the upstream choice of the same CrabNebula team that
maintains the Tauri bundler. Switching from cargo-packager to Tauri's
bundler later is essentially a config rename, not a tool change.

Justification — cargo-dist for orchestration: a single workflow that
authors `Cargo.toml [workspace.metadata.dist]` and generates the GitHub
Actions YAML keeps drift between platforms low. cargo-dist's MSI back-end
on WiX v3 is acceptable for v1.0 (WiX 3.14.1 is still released and
SmartScreen-compatible); if we hit limits we drop down to direct
cargo-packager MSI generation.

---

## §3 — Workspace binary inventory

Verified from `Cargo.toml` and `crates/*/Cargo.toml` on 2026-05-15.

### 3.1 Shippable binaries

| Crate           | Bin name      | Type           | Notes |
|-----------------|---------------|----------------|-------|
| `forex-app`     | `forex-app`   | Desktop GUI    | `eframe`/`egui` 0.31, `egui_dock` 0.16; `clap` 4.6 for CLI flags driving the GUI; may migrate to Tauri per the UI/UX migration spec. `product_name = "ForexAI"` is already set in Cargo.toml. |
| `forex-cli`     | `forex-cli`   | CLI + TUI      | `ratatui` 0.29 + `crossterm`. Headless. |
| `forex-core`    | (lib)         | Library        | Not shipped standalone. |
| `forex-data`    | `forex-data`  | CLI helper     | Data ingestion / backfill. Ships alongside `forex-cli`. |
| `forex-models`  | `forex-models`| CLI helper     | Model training / eval. Ships alongside `forex-cli`. |
| `forex-news`    | `forex-news`  | CLI helper     | News ingestion. Ships alongside `forex-cli`. |
| `forex-search`  | `forex-search`| CLI helper     | Search index. Ships alongside `forex-cli`. |

There is **no** dedicated `forex-bot-headless` binary today. The autonomous-
trading watchdog lives inside `forex-cli` (as a subcommand). The systemd file
`forex-bot.service` at the repo root references a `python -m forex_bot.main`
process — that is a legacy Python entry point (see `scripts/check_no_python_legacy.sh`)
and is being retired. The installer ships only Rust binaries.

### 3.2 Per-binary deployment matrix

| Binary | Target triple(s) | Bundled assets | Runtime resources | System deps |
|--------|------------------|----------------|-------------------|-------------|
| `forex-app` | `x86_64-pc-windows-msvc`, `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu` | `assets/symbol_metadata/defaults.json`; `assets/icons/`; Spotware `.proto` files at compile-time (already vendored via `prost-build`); training-config templates from `crates/forex-models/templates/` | Cache dir (per platform), log dir, OAuth secrets keystore | Linux: `libssl3`, `libgomp1`, `libvulkan1` (for `burn-wgpu`); macOS: Metal (system); Windows: `vcruntime140.dll` (covered by the Rust toolchain's MSVC runtime) |
| `forex-cli` | same as `forex-app` | training-config templates | same | `libgomp1`, `libssl3` |
| `forex-data` | same | none | same | same |
| `forex-models` | same | training-config templates, vendored `libcatboost` / `libxgboost` / `liblightgbm3` dynamic libs (operator already vendored under `vendor/lightgbm3-sys`) | model checkpoint output dir | optional CUDA 13 runtime if `--features gpu`; rocm if AMD |
| `forex-news` | same | none | API key file in config dir | none beyond above |
| `forex-search` | same | embedded index schema | search-index directory in data dir | none beyond above |

Asset locations (already in the workspace):

- `assets/symbol_metadata/defaults.json` — must ship as a data file in every
  package, installed to the platform's data dir (§8).
- Spotware proto files — already embedded at compile time via `prost-build`,
  no runtime distribution needed.

### 3.3 Optional GPU dependencies

`forex-models` and `forex-app` (via `burn-wgpu`) can use:

- **CUDA 13.x runtime** (Linux/Windows, NVIDIA only).
- **HIP/ROCm** (Linux, AMD; experimental).
- **Vulkan loader** (Linux, vendor-neutral via `burn-wgpu`).
- **Metal** (macOS, system framework).
- **DirectX 12** (Windows, system framework).

The installer wizard must:

1. Detect at install time whether CUDA/ROCm is present.
2. Default to the CPU build of catboost/lightgbm/xgboost if no GPU detected.
3. Offer a "GPU acceleration (NVIDIA / AMD)" optional component that downloads
   the GPU-enabled native blobs from our release asset URL. This is the *only*
   place we do a network-fetched component install — every other asset is in
   the base installer.

The "optional component" UX comes free with WiX `<Feature>` elements and
NSIS `Section /o`.

---

## §4 — Auto-update strategy

Operator rule: **no silent unverified updates.** Every auto-update path below
must verify a signature against an embedded public key before executing.

### 4.1 Update channel model

Two channels, both signed:

- **stable** — promoted from `release/*` branch; tagged `v0.X.Y`.
- **beta** — promoted from `main`; tagged `v0.X.Y-beta.N`.

The user picks the channel during install. Switching channels later is a
config-only change.

Updates are *opt-in by default* — the wizard asks "Check for updates
automatically?" and defaults to **yes** but never installs without an explicit
user click on the "Update now" button. This satisfies "no silent unverified
updates" and the EU CRA "user consent for security updates" line.

### 4.2 Sparkle / WinSparkle

- **Sparkle (macOS)** — <https://sparkle-project.org/>. License: MIT-style.
  Per <https://sparkle-project.org/documentation/eddsa-migration/>, "Sparkle's
  EdDSA (ed25519) signature is used to sign the published update archive
  (dmg, zip, etc), binary delta updates, and installer packages. Signatures
  are automatically generated when you make an appcast using `generate_appcast`
  tool".
- **WinSparkle (Windows)** — <https://github.com/vslavik/winsparkle>. License:
  MIT. "Inspired by the Sparkle framework for Mac, to the point of sharing
  the same updates format (appcasts)". Uses the same EdDSA Ed25519 chain on
  Windows.
- Operator advantages:
  - Same appcast XML format on both platforms.
  - `SURequireSignedFeed` option locks the appcast URL to a single signing key.
  - User explicitly clicks "Install Update" — there is a dialog by default.
- Operator-relevant work:
  - Generate the Ed25519 key pair once with `Sparkle/bin/generate_keys`,
    store the **private** key in our CI's secret store (1Password / GitHub
    Secrets / AWS Secrets Manager — TBD by ops).
  - Embed the **public** key in the binary at build time.
  - CI step: after building the DMG / PKG / MSI, run `generate_appcast` to
    produce `appcast.xml`, sign it, upload alongside the artifact to GitHub
    Releases.

### 4.3 Tauri updater (if Tauri migration happens)

- Documented at <https://v2.tauri.app/plugin/updater/>.
- Signature scheme: **minisign** (Ed25519). "Tauri's updater needs a signature
  to verify that the update is from a trusted source, and this cannot be
  disabled. When present, the update response's signature field and the
  downloaded artifact will be checked against the configured pubkey using
  Minisign".
- Key generation: `cargo tauri signer generate -w ~/.tauri/forex-ai.key`.
- Configuration: `tauri.conf.json > plugins.updater.{endpoints, pubkey}` plus
  `bundle.createUpdaterArtifacts = true`.
- Same operator workflow as Sparkle: private key in CI secrets, public key
  embedded.

### 4.4 cargo-packager-updater

- Same authors as cargo-packager and the Tauri bundler. Source:
  <https://github.com/crabnebula-dev/cargo-packager>, `crates/updater/`.
- Signature scheme: also minisign Ed25519.
- Reasonable to use even if forex-app stays on egui, because cargo-packager
  is already the bundler.

### 4.5 cargo-dist updater integration

- cargo-dist does not ship its own updater. It does emit shell / powershell
  installers that can detect a newer release on GitHub and re-run themselves.
  For forex-app we want in-process update notifications, not a shell loop —
  so we layer Sparkle/WinSparkle (or Tauri's updater) *inside* the binary
  and let cargo-dist handle release-artifact orchestration.

### 4.6 Signature verification chain end-to-end

```
[Developer machine]                [CI runner]                 [End user machine]
                                                                
forex-ai/ git tag v0.5.0  ───────► GitHub Actions ──────────► download msi/dmg/deb
                                          │                          │
                                          ▼                          ▼
                                   build artifact                signed?
                                          │                    yes (Authenticode / 
                                          ▼                    Developer ID / dpkg-sig)
                                   signtool / codesign /              │
                                   dpkg-sig --sign                    ▼
                                          │                    install runs
                                          ▼                          │
                                   generate_appcast --                ▼
                                   sign-ed25519-with                Sparkle/Tauri checks
                                   $SPARKLE_PRIV_KEY ───────────►  https://updates.forex-ai/
                                          │                          │
                                          ▼                          ▼
                                   upload to                  GET appcast.xml ─ ed25519 verify
                                   github releases                   │
                                                                     ▼
                                                              user clicks Install
                                                                     │
                                                                     ▼
                                                              download update bundle
                                                              ed25519 verify
                                                              authenticode/codesign verify
                                                              install
```

The dual-signature design (platform code-signing **plus** Sparkle Ed25519)
means even a compromised platform CA (worst case) cannot deliver an unsigned
update payload past our self-managed Ed25519 key.

---

## §5 — Code signing requirements

### 5.1 Windows

- **OV vs EV** — per
  <https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options>:
  "Organization Validated (OV) certificates from a Certificate Authority (CA)
  such as DigiCert, Sectigo, or GlobalSign are a well-established option for
  code signing. Extended Validation (EV) certificates previously bypassed
  SmartScreen entirely on first download, making them the go-to choice for new
  apps with no reputation. That behavior was removed in 2024. EV-signed files
  now go through the same reputation-building process as OV certificates."
- **Pricing** (Sectigo / DigiCert published pricing, 2025-2026):
  - Sectigo EV Code Signing: $279.99/yr per <https://signmycode.com/sectigo-ev-code-signing>.
  - DigiCert EV Code Signing: $559.99/yr per <https://signmycode.com/digicert-ev-code-signing>.
  - **Lifespan rule change**: "Starting February 15, 2026, code signing
    certificate lifespans are limited to a maximum of one year, and DigiCert
    now offers only 1-year code-signing certificate plans." Budget annually.
- **Hardware**: per CA/B Forum rules, EV code signing keys must reside in a
  FIPS-compliant hardware token or cloud HSM (YubiKey FIPS, Azure Key Vault,
  Google Cloud KMS, Luna). The CA either ships a USB token or accepts an
  existing HSM.
- **SignTool** invocation (canonical reference
  <https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool>):
  - "The SignTool sign command requires the `/fd` file digest algorithm and the
    `/td` timestamp digest algorithm option to be specified during signing and
    timestamping, respectively."
  - "All certificates must be SHA-2, and signed with the `/fd sha256` SignTool
    command line switch."
  - Example: `signtool sign /n "Forex AI Limited" /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 forex-ai-setup.exe`.
- **Azure Trusted Signing (now "Azure Artifact Signing")** — alternative path:
  - Pricing per <https://learn.microsoft.com/en-us/azure/artifact-signing/faq>:
    $9.99/month for up to 5,000 signatures, one certificate profile; $99.99/month
    for up to 100,000 signatures, 10 profiles.
  - **Not EV** — "there is no plan to issue EV certificates". So Trusted
    Signing cannot replace an EV cert for kernel-driver work, but it is
    perfectly adequate for a user-mode trading application (post-2024
    SmartScreen change made EV and OV behave the same way for reputation
    accumulation).
  - **Operator-relevant**: Trusted Signing is cheaper and removes the USB-
    token-in-the-CI-runner problem entirely. **Recommended over EV for
    forex-ai** unless we need a logo'd installer that bypasses SmartScreen
    reputation accumulation (we do not).

### 5.2 macOS

- **Apple Developer Program**: $99/yr per
  <https://developer.apple.com/programs/whats-included/>. Enroll as
  individual or organization; organization enrollment requires a D-U-N-S
  number and takes 1–6 weeks (the budget concern is *time*, not money).
- **Certificate types**: per <https://developer.apple.com/developer-id/>:
  - "Developer ID Application" — signs the `.app` binary.
  - "Developer ID Installer" — signs the `.pkg`.
  - Both are needed.
- **Hardened Runtime + entitlements** declared at sign time:
  ```
  codesign --force --options runtime --entitlements forex-app.entitlements \
           --sign "Developer ID Application: Forex AI Ltd (TEAMID)" \
           --timestamp \
           forex-app.app
  ```
- **Notarization** with `xcrun notarytool submit ... --wait` followed by
  `xcrun stapler staple` (so users can install offline once notarized).
- All commands require Xcode Command Line Tools (`xcode-select --install`).

### 5.3 Linux

- **GPG signing of `.deb`**: `dpkg-sig --sign builder forex-ai_<ver>_amd64.deb`.
  The `.deb` format embeds the GPG signature; APT verifies via the system
  keyring when the user has installed our `forex-ai.gpg` to
  `/etc/apt/keyrings/`.
- **GPG signing of `.rpm`**: cargo-generate-rpm `--signing_key <gpg-key>.asc`.
- **AppImage GPG**: `appimagetool -s ...` (GPG2 embedded).
- **Sigstore cosign** (optional, modern alternative) — per
  <https://docs.sigstore.dev/quickstart/quickstart-cosign/>: "Cosign supports
  software artifact signing, verification, and storage in an OCI registry...
  it can also be used for other file types, including SBOMs, WASM modules,
  and Tekton bundles." Keyless mode (Fulcio + Rekor) gives a non-secret
  transparency-logged signature. **Recommended addition** to the GPG chain
  — both sigs published alongside each artifact, so a paranoid user can
  cross-verify.
- **Cost**: GPG is free; Sigstore is free.

### 5.4 Annual cost summary (operator budget line item)

| Item | Annual cost (USD) | Notes |
|------|-------------------|-------|
| Apple Developer Program | $99 | Hard requirement for macOS distribution. |
| Azure Artifact Signing (basic) | ~$120 ($9.99/mo) | **Recommended** path for Windows. |
| (Alt) Sectigo EV Code Signing | $280 | If we prefer USB-token model. |
| (Alt) DigiCert EV Code Signing | $560 | Avoid unless required. |
| GPG key + key servers | $0 | Self-managed. |
| Sigstore cosign | $0 | Optional, keyless. |
| **Total minimum** | **~$220** | Apple + Azure Artifact Signing. |

---

## §6 — Auto-uninstall + clean removal

The installer must register an uninstaller and clean the following on
uninstall:

### 6.1 Binaries

- Windows: `C:\Program Files\forex-ai\` directory (per-machine install) or
  `%LOCALAPPDATA%\Programs\forex-ai\` (per-user). NSIS / WiX uninstaller
  handles this natively.
- macOS: `/Applications/Forex AI.app` removed by the `.pkg` `BundlePostFlight`
  script.
- Linux: `dpkg -P forex-ai` / `rpm -e forex-ai` removes `/usr/bin/forex-app`,
  `/usr/bin/forex-cli`, and the assets under `/usr/share/forex-ai/`.

### 6.2 Data, config, cache, log directories (user prompt required)

The operator's data is potentially valuable (trained model checkpoints,
historical price cache, OAuth refresh tokens). The uninstall wizard MUST
present a checkbox dialog:

> "Also remove your forex-ai data, settings, and logs?
> [ ] Yes, remove everything (default: unchecked)
> Your account secrets and downloaded market history will be deleted."

If checked, remove:

- Windows: `%LOCALAPPDATA%\forex-ai\` (data, logs, cache),
  `%APPDATA%\forex-ai\` (config).
- macOS: `~/Library/Application Support/forex-ai/`,
  `~/Library/Caches/forex-ai/`, `~/Library/Logs/forex-ai/`,
  `~/Library/Preferences/com.forex-ai.app.plist`.
- Linux: `~/.local/share/forex-ai/`, `~/.config/forex-ai/`,
  `~/.cache/forex-ai/`, `~/.local/state/forex-ai/`.

### 6.3 Auto-start hooks

- Windows: registry value at
  `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run` named
  `forex-ai` (per <https://learn.microsoft.com/en-us/windows/win32/setupapi/run-and-runonce-registry-keys>).
  Uninstaller removes the value.
- macOS: LaunchAgent plist at
  `~/Library/LaunchAgents/com.forex-ai.watchdog.plist` (per
  <https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html>).
  Uninstaller runs `launchctl unload` then deletes the file.
- Linux: systemd user unit at `~/.config/systemd/user/forex-ai.service` plus
  XDG autostart entry at `~/.config/autostart/forex-ai.desktop`.
  Uninstaller runs `systemctl --user disable --now forex-ai.service` then
  deletes both files.

### 6.4 OAuth refresh tokens + URL scheme handler

- The cTID OAuth refresh token lives in the OS keystore:
  - Windows: Windows Credential Manager (`CredWrite`/`CredDelete` via `wincred`
    crate or via cargo-packager's `keyring` integration).
  - macOS: Keychain (`security delete-generic-password -s "forex-ai-ctid"`).
  - Linux: Secret Service via libsecret (e.g. GNOME Keyring / KWallet).
- The uninstaller deletes the keystore entry and prints a note to the user:
  > "Your cTID refresh token has been removed locally. To revoke server-side,
  > visit https://id.ctrader.com/ → Authorized Apps → Forex AI → Revoke."
- The custom URL scheme handler `forex-ai://oauth/callback` is registered:
  - Windows: registry key
    `HKEY_CLASSES_ROOT\forex-ai\shell\open\command` set to the binary path
    plus `--oauth-callback %1`.
  - macOS: `CFBundleURLSchemes` array in `Info.plist`; LaunchServices picks it
    up automatically.
  - Linux: `MimeType=x-scheme-handler/forex-ai;` line in the `.desktop` file
    plus `xdg-mime default forex-ai.desktop x-scheme-handler/forex-ai`.

### 6.5 Uninstall verification

After uninstall the wizard runs a self-check: enumerate the four paths
(binary, data, config, autostart) and report any leftovers. This catches
the edge case of a running process holding a file lock at uninstall time.

---

## §7 — Auto-launch / system-tray

`forex-app` ships a system-tray daemon that runs the autonomous-trading
watchdog and the periodic reconciliation job. The installer must register it
as an autostart entry **with the user's explicit consent** — the wizard
asks "Run Forex AI in the background on startup?" with default = NO.

### 7.1 Windows

Two equivalent choices, both documented by Microsoft:

- **Registry Run key** (simpler, no admin) —
  `HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Run` value
  `forex-ai` = `"C:\Program Files\forex-ai\forex-app.exe" --background`.
  Per <https://learn.microsoft.com/en-us/windows/win32/setupapi/run-and-runonce-registry-keys>:
  "Run key makes the program run every time the user logs on".
- **Task Scheduler** (more flexible, allows delayed start / network-ready
  trigger / battery filter): `schtasks /create /tn ForexAI /tr ...
  /sc onlogon /delay 0001:00`. Recommended for the "wait until network is up"
  use case, which matters for the cTrader proxy connection.

We use **Task Scheduler** because the proxy connection is meaningless before
the network stack is up.

### 7.2 macOS

LaunchAgent plist at `~/Library/LaunchAgents/com.forex-ai.watchdog.plist`,
template:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>            <string>com.forex-ai.watchdog</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Applications/Forex AI.app/Contents/MacOS/forex-app</string>
        <string>--background</string>
    </array>
    <key>RunAtLoad</key>        <true/>
    <key>KeepAlive</key>        <true/>
    <key>StandardOutPath</key>  <string>/Users/USER/Library/Logs/forex-ai/watchdog.out</string>
    <key>StandardErrorPath</key><string>/Users/USER/Library/Logs/forex-ai/watchdog.err</string>
</dict>
</plist>
```

Per the launchd documentation
(<https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html>):
"Property list files describing agents are installed in /Library/LaunchAgents
or in the LaunchAgents subdirectory of an individual user's Library
directory." We use the user-specific path because the agent must run as the
logged-in user (it owns the keychain entry).

`KeepAlive=true` ensures the watchdog is restarted if it crashes — per
<https://launchd.info/>: "the value may be set to true to unconditionally
keep the job alive".

### 7.3 Linux

Two parallel registrations, both required because GNOME and KDE differ:

- **systemd user unit** at `~/.config/systemd/user/forex-ai.service`:
  ```
  [Unit]
  Description=Forex AI background watchdog
  After=network-online.target
  Wants=network-online.target

  [Service]
  Type=simple
  ExecStart=/usr/bin/forex-app --background
  Restart=on-failure
  RestartSec=10s

  [Install]
  WantedBy=default.target
  ```
  Enabled by `systemctl --user enable --now forex-ai.service`.
- **XDG autostart `.desktop` entry** at `~/.config/autostart/forex-ai.desktop`:
  ```
  [Desktop Entry]
  Type=Application
  Name=Forex AI
  Exec=/usr/bin/forex-app --background
  X-GNOME-Autostart-enabled=true
  Hidden=false
  ```
  Per <https://specifications.freedesktop.org/autostart-spec/autostart-spec-latest.html>.

The systemd user unit is the actual runtime; the `.desktop` is a fallback for
distros that haven't enabled `systemd-xdg-autostart-generator`. Per
<https://www.freedesktop.org/software/systemd/man/latest/systemd-xdg-autostart-generator.html>:
"systemd-xdg-autostart-generator is a generator that creates .service units
for XDG autostart files."

Existing repo artefact: `forex-bot.service` is the old Python systemd unit;
it is **replaced** by `~/.config/systemd/user/forex-ai.service` above and
will be deleted in the cleanup branch.

---

## §8 — Filesystem layout per platform

### 8.1 Windows

| Path | Purpose | Source |
|------|---------|--------|
| `C:\Program Files\forex-ai\forex-app.exe` | Main desktop binary (per-machine) | `FOLDERID_ProgramFilesX64` |
| `C:\Program Files\forex-ai\forex-cli.exe` | CLI binary | same |
| `C:\Program Files\forex-ai\bundle\` | Bundled data files (symbol_metadata, templates) | same |
| `%LOCALAPPDATA%\forex-ai\data\` | Local model checkpoints, price cache | `FOLDERID_LocalAppData` |
| `%LOCALAPPDATA%\forex-ai\logs\` | Logs | same |
| `%LOCALAPPDATA%\forex-ai\cache\` | Disposable cache | same |
| `%APPDATA%\forex-ai\config.yaml` | Roaming user config | `FOLDERID_RoamingAppData` |

`FOLDERID_LocalAppData`, `FOLDERID_RoamingAppData`, `FOLDERID_ProgramFilesX64`
are the canonical Known Folder IDs documented at
<https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid>.

Per-user install variant (no admin): everything under
`%LOCALAPPDATA%\Programs\forex-ai\` instead of `Program Files`.

### 8.2 macOS

| Path | Purpose | Source |
|------|---------|--------|
| `/Applications/Forex AI.app/` | App bundle | Apple HIG / File System Programming Guide |
| `/Applications/Forex AI.app/Contents/MacOS/forex-app` | Main binary | same |
| `/Applications/Forex AI.app/Contents/MacOS/forex-cli` | CLI binary (also symlinked to `/usr/local/bin/forex-cli` by the installer post-flight) | same |
| `/Applications/Forex AI.app/Contents/Resources/` | Bundled assets | same |
| `~/Library/Application Support/forex-ai/` | User data (model checkpoints, price cache) | Apple File System Programming Guide |
| `~/Library/Logs/forex-ai/` | Logs | same |
| `~/Library/Caches/forex-ai/` | Disposable cache | same |
| `~/Library/Preferences/com.forex-ai.app.plist` | NSUserDefaults | same |

The Apple File System Programming Guide
(<https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/MacOSXDirectories/MacOSXDirectories.html>)
spells out these locations.

### 8.3 Linux

Verified against the XDG Base Directory Specification
(<https://specifications.freedesktop.org/basedir/latest/>):

| Path | Purpose | XDG variable / default |
|------|---------|------------------------|
| `/usr/bin/forex-app` | Main binary (system-wide deb/rpm install) | FHS |
| `/usr/bin/forex-cli` | CLI binary | FHS |
| `/usr/share/forex-ai/` | Bundled assets, default config templates | FHS |
| `~/.local/share/forex-ai/` | User data | `$XDG_DATA_HOME`, default `$HOME/.local/share` |
| `~/.config/forex-ai/` | User config | `$XDG_CONFIG_HOME`, default `$HOME/.config` |
| `~/.cache/forex-ai/` | Disposable cache | `$XDG_CACHE_HOME`, default `$HOME/.cache` |
| `~/.local/state/forex-ai/logs/` | Logs (state, not cache) | `$XDG_STATE_HOME`, default `$HOME/.local/state` |
| `$XDG_RUNTIME_DIR/forex-ai/` | Runtime socket(s) for IPC between forex-cli and the watchdog | set by `pam_systemd`, typically `/run/user/$UID` |

Per the XDG spec summary:
- `XDG_DATA_HOME` — "$HOME/.local/share" — user-specific data.
- `XDG_CONFIG_HOME` — "$HOME/.config" — user-specific configuration.
- `XDG_CACHE_HOME` — "$HOME/.cache" — non-essential cache.
- `XDG_STATE_HOME` — "$HOME/.local/state" — persistent state (logs, history).
- `XDG_RUNTIME_DIR` — set by pam_systemd, sub-second lifetime sockets.

AppImage install path (per-user, when AppImage is the chosen channel):
- `~/.local/bin/forex-app` (symlink into the AppImage), with the AppImage
  itself living at `~/.local/share/forex-ai/forex-app.AppImage`.

The implementation library is `dirs` 5.x or `directories` 5.x in Rust; both
implement the XDG spec and the macOS / Windows equivalents correctly.

---

## §9 — CI/CD release pipeline

### 9.1 GitHub Actions matrix

cargo-dist generates the workflow but the resulting `release.yml` is
human-readable; the operator should treat the generated file as source of
truth.

```
on:
  push:
    tags:
      - 'v[0-9]+.[0-9]+.[0-9]+'
      - 'v[0-9]+.[0-9]+.[0-9]+-beta.[0-9]+'
  workflow_dispatch:
    inputs:
      dry_run: { type: boolean, default: true }

jobs:
  plan:                 # cargo-dist plan
    runs-on: ubuntu-latest
  build-windows:
    needs: plan
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo install cargo-packager --locked
      - run: cargo packager --formats nsis,msi --release
      - run: pwsh ./ci/sign-windows.ps1 target/release/forex-ai-*.{msi,exe}
  build-macos:
    needs: plan
    runs-on: macos-latest                # 14 / Sonoma, Apple Silicon
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { targets: 'x86_64-apple-darwin,aarch64-apple-darwin' }
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --release --target x86_64-apple-darwin
      - run: cargo build --release --target aarch64-apple-darwin
      - run: ./ci/lipo-and-bundle.sh
      - run: ./ci/codesign-and-notarize.sh
  build-linux:
    needs: plan
    runs-on: ubuntu-22.04                # glibc 2.35 floor
    strategy:
      matrix: { distro: [deb, rpm, appimage] }
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo install cargo-deb cargo-generate-rpm cargo-packager --locked
      - run: ./ci/build-${{ matrix.distro }}.sh
      - run: ./ci/gpg-sign.sh
  release:
    needs: [build-windows, build-macos, build-linux]
    runs-on: ubuntu-latest
    steps:
      - run: gh release create v$VERSION --notes-file CHANGELOG.md *.msi *.exe *.dmg *.pkg *.deb *.rpm *.AppImage appcast.xml
```

### 9.2 Cross-compile for arm64 macOS from x86_64 macOS runner

GitHub-hosted `macos-14` and later runners are Apple Silicon by default; we
cross-compile *to* x86_64 from the arm64 runner (not the other way around).
The toolchain flag is `--target x86_64-apple-darwin` and the Rust standard
library is provided by rustup. The CI script `lipo-and-bundle.sh` then
merges the two slice binaries with:

```
lipo -create -output forex-app.universal \
    target/x86_64-apple-darwin/release/forex-app \
    target/aarch64-apple-darwin/release/forex-app
```

before packaging.

### 9.3 Artifact upload

GitHub Releases is the single source of truth. cargo-dist creates the
release object, uploads artifacts (`.msi`, `.exe`, `.dmg`, `.pkg`, `.deb`,
`.rpm`, `.AppImage`), the appcast XML, and an SBOM (`cargo sbom` →
`forex-ai-v$VERSION.cdx.json`).

### 9.4 Triggers

- **Tag-based**: `git tag v0.5.0 && git push --tags` triggers the full
  release.
- **Manual `workflow_dispatch`** with `dry_run: true` allows the operator to
  run the full pipeline without publishing — useful for testing a release on
  a fork.

### 9.5 Caching

`Swatinem/rust-cache@v2` keys on Cargo.lock + Cargo.toml hashes, cuts a
cold build from ~30 min to ~6 min on macOS. Cache is invalidated on
toolchain bump.

### 9.6 Secrets

Required GitHub Actions secrets:

| Secret | Purpose | Used by |
|--------|---------|---------|
| `WINDOWS_SIGNING_CERT_BASE64` | OV/EV cert PFX, base64-encoded | sign-windows.ps1 |
| `WINDOWS_SIGNING_CERT_PASSWORD` | PFX password | same |
| `AZURE_TRUSTED_SIGNING_CLIENT_ID` | If using Azure Artifact Signing instead | same |
| `AZURE_TRUSTED_SIGNING_TENANT_ID` | same | same |
| `AZURE_TRUSTED_SIGNING_CLIENT_SECRET` | same | same |
| `APPLE_ID` | Developer Apple ID email | notarize |
| `APPLE_APP_SPECIFIC_PASSWORD` | App-specific password (Apple ID web setting) | notarize |
| `APPLE_TEAM_ID` | 10-char team ID | notarize |
| `APPLE_DEV_ID_APPLICATION_CERT_BASE64` | "Developer ID Application" cert .p12 | codesign |
| `APPLE_DEV_ID_INSTALLER_CERT_BASE64` | "Developer ID Installer" cert .p12 | productsign |
| `APPLE_CERT_PASSWORD` | .p12 passphrase | both |
| `GPG_PRIVATE_KEY` | Linux package signing key | gpg-sign.sh |
| `GPG_PASSPHRASE` | same | same |
| `SPARKLE_PRIVATE_KEY` | Sparkle Ed25519 private key | appcast generation |
| `TAURI_SIGNING_PRIVATE_KEY` | Only if Tauri migration | tauri build |
| `GITHUB_TOKEN` | provided by Actions | release |

---

## §10 — Migration / coexistence

### 10.1 Existing portable users

The pre-installer workflow assumed a user-extracted ZIP at
`~/forex-ai/` (Linux/macOS) or `C:\forex-ai\` (Windows). The installer wizard
on its first run scans for that legacy location and offers a **one-time
migration**:

> "We found an existing Forex AI installation at `<path>`. Move your config
> and data to the new locations?
> [x] Migrate config (`config.yaml`)
> [x] Migrate data (`data/`, `checkpoints/`)
> [x] Migrate logs (`logs/`)
> [ ] Delete the old folder after migration"

The migration is a file-copy (NOT a move) until the user confirms the last
checkbox. We preserve symlinks. We rewrite any *absolute* paths inside
`config.yaml` to the new locations.

Per platform scan paths:

- Windows: `C:\forex-ai\`, `%USERPROFILE%\forex-ai\`, `%USERPROFILE%\Downloads\forex-ai\`.
- macOS: `~/forex-ai/`, `~/Downloads/forex-ai/`.
- Linux: `~/forex-ai/`, `~/Downloads/forex-ai/`, `/opt/forex-ai/`.

### 10.2 "Uninstall the portable copy" guidance

After successful migration, the wizard prints platform-specific guidance:

- Windows: "Delete `C:\forex-ai\` from File Explorer."
- macOS: "Drag `~/forex-ai/` to the Trash."
- Linux: "Run `rm -rf ~/forex-ai/`."

We do **not** auto-delete unless the user checked the box — the operator's
data is too valuable to delete on a wizard's authority alone.

### 10.3 Side-by-side coexistence

The installer always writes to the canonical paths in §8. If the user
already has a portable copy running, the installer detects the pidfile at
the legacy path (`~/forex-ai/forex-app.pid`) and refuses to proceed until
it is stopped. No silent overlap.

### 10.4 Downgrade path

If a user installs v0.5.0 on top of v0.4.x, the installer:

1. Saves the old binary at `forex-app.v0.4.bak` for one release cycle.
2. Migrates the data dir in-place (data format is versioned in a `meta.json`).
3. Provides a `forex-ai --rollback` CLI flag that restores the .bak binary.

Downgrade from v0.5 → v0.4 is **not** automatic — data format migrations
are forward-only.

---

## §11 — Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Apple Developer ID enrollment lead time (1–6 weeks for organisation accounts) | High | Begin enrollment *before* engineering work; use the personal account in the interim to test the notarization flow. |
| Windows EV cert delivery delay (USB token shipping) | Medium | Use Azure Artifact Signing instead — no hardware, no shipping. |
| WiX Toolset Open Source Maintenance Fee uncertainty | Medium | Read the FireGiant fee schedule; budget annually. Switch to NSIS-only if the fee is unacceptable. |
| Notarization service occasionally takes hours instead of minutes | Low | CI step uses `notarytool ... --wait` with a 2-hour timeout; the GitHub Actions job is allowed to retry once. |
| AppImage `FUSE2` deprecation on newer distros | Medium | Ship the AppImage with the new FUSE3-compatible runtime (`appimagetool --runtime-file <new-runtime>`) and document the manual `chmod +x; ./forex-ai.AppImage --appimage-extract-and-run` fallback. |
| Sparkle Ed25519 key rotation | Medium | Embed both current and *next* public key from v0.5.0 onward; the next-key is unused until rotation, then the swap is silent. |
| User has CUDA installed at install time but uninstalls it later | Low | Detect at startup, fall back to CPU, log a warning. |
| EU CRA "vulnerability handling" obligations (2027) | High (compliance) | Establish a security disclosure channel before v1.0; require SBOM generation in CI; sign all artifacts. |
| Browser-saved OAuth refresh token outliving uninstall | Medium | Document the cTID revocation URL in the uninstall final dialog (§6.4). |

---

## §12 — Open questions

1. **Tauri migration timeline** — if Tauri replaces egui in 2026 Q3, we can
   collapse the entire build matrix to "tauri build" and use the Tauri
   updater. Decision needed by 2026 Q2.
2. **Microsoft Store distribution** — MSIX rules out our background daemon
   model (§2.1.5). Are we willing to split the desktop GUI (Store-friendly)
   from the watchdog (sideload-only)? Decision deferred.
3. **Snap Store** — Canonical's auto-update model conflicts with our
   "no silent unverified updates" rule. We can pin the Snap to the
   `--channel=stable --hold` model, but the UX is awkward. Decision: Snap
   is out for v1.0.
4. **AUR maintainer** — Arch users prefer AUR over generic binary packages.
   Do we maintain `forex-ai-bin` ourselves or hand it off to a community
   maintainer with our blessing? Decision needed before v1.0.
5. **Homebrew tap vs cask** — cargo-dist supports a `homebrew` tap that emits
   a `Formula`. For a GUI app the user really wants a `Cask`. Need to confirm
   cargo-dist supports cask emission, or hand-author the cask file.
6. **Reproducible builds** — to satisfy a paranoid operator who wants to
   audit our binaries, set `SOURCE_DATE_EPOCH`, pin all toolchains, and
   publish a `release.lock` hash file. Not required for v1.0 but cheap to
   add.

---

## §13 — Bill of materials (BOM)

Final list of installer artifacts emitted per release:

```
forex-ai-v0.5.0-x86_64-pc-windows-msvc-setup.exe       # NSIS
forex-ai-v0.5.0-x86_64-pc-windows-msvc.msi             # WiX MSI
forex-ai-v0.5.0-universal-apple-darwin.dmg             # universal DMG
forex-ai-v0.5.0-universal-apple-darwin.pkg             # universal PKG wizard
forex-ai-v0.5.0_amd64.deb                              # Debian / Ubuntu
forex-ai-v0.5.0-1.x86_64.rpm                           # Fedora
forex-ai-v0.5.0-x86_64.AppImage                        # universal Linux
forex-ai-v0.5.0.appcast.xml                            # Sparkle/WinSparkle feed
forex-ai-v0.5.0.cdx.json                               # CycloneDX SBOM
forex-ai-v0.5.0.SHA256SUMS                             # checksums
forex-ai-v0.5.0.SHA256SUMS.asc                         # GPG-signed checksums
forex-ai-v0.5.0.sig                                    # Sigstore cosign sig (optional)
```

Every file in this list except the SBOM and SHA256SUMS is *itself* signed
with the platform-native code-signing chain. The SHA256SUMS file's GPG
signature gives the operator a single point of verification for the entire
release.

---

## §14 — Reference quick-lookup

| Topic | Primary URL |
|-------|-------------|
| WiX Toolset releases | <https://github.com/wixtoolset/wix/releases> |
| WiX release notes (FireGiant) | <https://docs.firegiant.com/wix/whatsnew/releasenotes/> |
| cargo-wix | <https://github.com/volks73/cargo-wix> |
| Inno Setup SignTool | <https://jrsoftware.org/ishelp/topic_setup_signtool.htm> |
| NSIS 3.11 release | <https://sourceforge.net/p/nsis/news/2025/03/nsis-311-released/> |
| cargo-bundle | <https://github.com/burtonageo/cargo-bundle> |
| create-dmg | <https://github.com/sindresorhus/create-dmg> |
| cargo-deb | <https://github.com/kornelski/cargo-deb> |
| cargo-generate-rpm | <https://github.com/cat-in-136/cargo-generate-rpm> |
| AppImage updates | <https://docs.appimage.org/packaging-guide/optional/updates.html> |
| AppImage FUSE | <https://github.com/AppImage/AppImageKit/wiki/FUSE> |
| Flatpak sandbox | <https://docs.flatpak.org/en/latest/sandbox-permissions.html> |
| Snapcraft confinement | <https://documentation.ubuntu.com/snapcraft/stable/explanation/classic-confinement/> |
| Tauri distribute | <https://v2.tauri.app/distribute/> |
| Tauri Windows installer | <https://v2.tauri.app/distribute/windows-installer/> |
| Tauri updater | <https://v2.tauri.app/plugin/updater/> |
| Tauri Windows signing | <https://v2.tauri.app/distribute/sign/windows/> |
| cargo-dist book | <https://axodotdev.github.io/cargo-dist/book/install.html> |
| cargo-dist MSI | <https://github.com/axodotdev/cargo-dist/blob/main/book/src/installers/msi.md> |
| cargo-packager | <https://github.com/crabnebula-dev/cargo-packager> |
| cargo-packager docs | <https://docs.crabnebula.dev/packager/> |
| Sparkle | <https://sparkle-project.org/documentation/> |
| Sparkle EdDSA | <https://sparkle-project.org/documentation/eddsa-migration/> |
| WinSparkle | <https://github.com/vslavik/winsparkle> |
| Apple Developer Program | <https://developer.apple.com/programs/whats-included/> |
| Apple Developer ID | <https://developer.apple.com/developer-id/> |
| Apple notarization | <https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution> |
| notarytool migration | <https://developer.apple.com/news/upcoming-requirements/?id=11012023a> |
| launchd | <https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html> |
| launchd info | <https://launchd.info/> |
| XDG Base Directory | <https://specifications.freedesktop.org/basedir/latest/> |
| Arch XDG wiki | <https://wiki.archlinux.org/title/XDG_Base_Directory> |
| systemd XDG autostart | <https://www.freedesktop.org/software/systemd/man/latest/systemd-xdg-autostart-generator.html> |
| Windows Known Folders | <https://learn.microsoft.com/en-us/windows/win32/shell/knownfolderid> |
| Windows Run / RunOnce | <https://learn.microsoft.com/en-us/windows/win32/setupapi/run-and-runonce-registry-keys> |
| Windows code signing options | <https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options> |
| SignTool | <https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool> |
| MSIX docs | <https://learn.microsoft.com/en-us/windows/msix/> |
| MSIX prepare | <https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-prepare> |
| Azure Artifact Signing FAQ | <https://learn.microsoft.com/en-us/azure/artifact-signing/faq> |
| Sectigo code signing | <https://www.sectigo.com/ssl-certificates-tls/code-signing> |
| WinGet manifest | <https://learn.microsoft.com/en-us/windows/package-manager/package/manifest> |
| Sigstore cosign | <https://docs.sigstore.dev/quickstart/quickstart-cosign/> |
| cTrader Open API | <https://help.ctrader.com/open-api/> |
| cTrader proxies & endpoints | <https://help.ctrader.com/open-api/proxies-endpoints/> |
| cTrader connection | <https://help.ctrader.com/open-api/connection/> |

---

*End of spec. Word count: this file targets the 1000–1500 LOC range; sections
1–14 inclusive land near the upper bound of that envelope.*
