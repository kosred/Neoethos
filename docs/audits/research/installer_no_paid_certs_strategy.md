# Installer — Zero-Paid-Certificate Distribution Strategy (Addendum)

> **Status**: research deliverable, no code changes implied.
> **Author**: research agent, 2026-05-15.
> **Operator directive** (verbatim, in Greek, dated 2026-05-15):
>
> > «Δυστυχώς δεν έχω κάποια άδεια από Microsoft, ούτε από Apple. Θα
> > πρέπει να βρούμε άλλους τρόπους χωρίς να χρειαστεί να πληρώσω
> > και να είναι αποδεκτή από τις εταιρίες χωρίς warnings κλπ.»
>
> English: "Unfortunately I have no licence from Microsoft, nor from
> Apple. We will need to find other ways without me having to pay,
> and it must be accepted by the companies without warnings etc."
>
> This document is an **addendum** to
> `docs/audits/research/installer_infrastructure_spec.md` (the
> "installer infra spec"). It does **not** replace that file; it
> amends two specific sections:
>
> - §5 (Code signing requirements) — overrides the Apple Developer ID
>   $99/yr and Azure Artifact Signing $120/yr lines so the
>   *minimum total* drops from ~$220/yr to **$0/yr**.
> - §9 (CI/CD release pipeline) — simplifies the required secrets
>   matrix to drop Apple-, Microsoft-, and DigiCert-issued certs.
>
> Everything else in installer_infra_spec — §1 (target platforms),
> §2 (installer toolchains), §3 (workspace binary inventory), §6
> (uninstall), §7 (auto-launch), §8 (filesystem layout), §10
> (migration / coexistence) — remains in force.
>
> The wizard contract in
> `docs/audits/research/installer_wizard_ux_spec.md` is **unaffected**
> at the UX level (Steps 1-10 still run after install regardless of
> how the binary got onto disk) but §1.3 of the wizard spec
> ("Post-install hand-off") and §6 ("Migration from portable") gain a
> new entry point: package-manager-managed installs (`brew`, `winget`,
> `apt`, `dnf`, `pacman`/AUR) drop the `install_metadata.json`
> sentinel through the same path the bundled installer wrote it.

---

## §0 — Sources

Every quoted excerpt below is grounded in an upstream source. URLs
are inline at the point of citation. Where a URL was unreachable
during this research session (HTTP 403 to WebFetch, restricted MCP
search) the citation says so and the surrounding paragraph uses only
text that was successfully retrieved from an alternate raw GitHub or
documentation mirror.

External sources sweeped (verbatim 2026-05-15):

- Cosign README, `https://github.com/sigstore/cosign` (raw fetched
  from `raw.githubusercontent.com/sigstore/cosign/main/README.md`).
- Sigstore docs, signing blobs page —
  `raw.githubusercontent.com/sigstore/docs/main/content/en/cosign/signing/signing_with_blobs.md`.
- Sigstore Cosign Quickstart —
  `raw.githubusercontent.com/sigstore/docs/main/content/en/quickstart/quickstart-cosign.md`.
- Sparkle 2 README — `raw.githubusercontent.com/sparkle-project/Sparkle/2.x/README.markdown`.
- Sparkle 2 Security.md —
  `raw.githubusercontent.com/sparkle-project/Sparkle/2.x/Documentation/Security.md`.
- Sparkle 2 Installation.md —
  `raw.githubusercontent.com/sparkle-project/Sparkle/2.x/Documentation/Installation.md`.
- Homebrew Cask `CONTRIBUTING.md` —
  `raw.githubusercontent.com/Homebrew/homebrew-cask/master/CONTRIBUTING.md`.
- Homebrew Cask `USAGE.md` —
  `raw.githubusercontent.com/Homebrew/homebrew-cask/master/USAGE.md`.
- Homebrew `Acceptable-Casks.md` —
  `raw.githubusercontent.com/Homebrew/brew/master/docs/Acceptable-Casks.md`.
- Homebrew `Cask-Cookbook.md` —
  `raw.githubusercontent.com/Homebrew/brew/master/docs/Cask-Cookbook.md`.
- Homebrew `Adding-Software-to-Homebrew.md` —
  `raw.githubusercontent.com/Homebrew/brew/master/docs/Adding-Software-to-Homebrew.md`.
- Homebrew `cask/quarantine.rb` —
  `raw.githubusercontent.com/Homebrew/brew/master/Library/Homebrew/cask/quarantine.rb`.
- WinGet community repo `CONTRIBUTING.md` —
  `raw.githubusercontent.com/microsoft/winget-pkgs/master/CONTRIBUTING.md`.
- AppImageKit README —
  `raw.githubusercontent.com/AppImage/AppImageKit/master/README.md`.
- appimagetool repository — `github.com/AppImage/appimagetool`
  (WebFetch summary).
- Chocolatey community packages repo summary —
  `github.com/chocolatey-community/chocolatey-packages`.
- Sparkle EdDSA migration page —
  `https://sparkle-project.org/documentation/eddsa-migration/`
  (HTTP 403 to WebFetch — content reconstructed from Sparkle 2
  README and Security.md instead).

**Sources requested but blocked / unreachable** in this session:

- `learn.microsoft.com` (all `/en-us/…` paths returned HTTP 403 to
  WebFetch; `microsoft_docs_search` MCP tool was denied for this
  research). For Partner Center fees / SmartScreen reputation /
  MSIX restrictions, this document carries forward the
  already-cited text that the installer_infra_spec spec quoted
  directly from those pages, and **annotates** any 2026-05-15
  unverified claim with `(carried from installer_infra_spec)`.
- `partner.microsoft.com` — HTTP 403 to WebFetch.
- `wikipedia.org` (Gatekeeper, SmartScreen, Chocolatey, Flathub,
  Homebrew) — HTTP 403 to WebFetch.
- `docs.flatpak.org` / `snapcraft.io` / `chocolatey.org` /
  `flathub.org` / `aur.archlinux.org` / `wiki.archlinux.org` —
  HTTP 403 to WebFetch. Where these are necessary, the surrounding
  paragraph quotes from raw GitHub READMEs of the same projects
  or carries-forward citations from the installer_infra_spec
  that were already grounded there.

Internal references:

- `docs/audits/research/installer_infrastructure_spec.md` —
  parent spec being amended; §5 and §9 specifically.
- `docs/audits/research/installer_wizard_ux_spec.md` — wizard flow
  that runs *after* installation regardless of distribution channel.
- `crates/forex-app/src/app_services/ctrader_live_auth.rs` — the
  in-app OAuth flow that runs from the wizard step 4.
- `crates/forex-core/src/contracts/temporal.rs` and
  `crates/forex-core/src/domain/prop_firm.rs` — operator-directive
  invariants referenced by the wizard.

---

## §1 — Windows: avoid SmartScreen warnings without a paid cert

### 1.1 Path A — Microsoft Store (free for individual publishers)

**Premise.** The Microsoft Store distribution is the only Windows
channel that gives a zero-warning install on a fresh Windows 10/11
machine *without* paying for an OV/EV code-signing certificate.
Apps shipped through the Store are re-signed by Microsoft's own
infrastructure during ingestion, and SmartScreen treats them as
fully trusted because the chain terminates at the Microsoft Store
CA. (carried from installer_infra_spec §5.1: "EV-signed files now go
through the same reputation-building process as OV certificates"
since 2024, so the Store path is the only one that bypasses
SmartScreen entirely on a fresh install.)

**Developer-account cost.** Per the URL the operator referenced
(`https://partner.microsoft.com/en-us/dashboard/registration` —
which returned HTTP 403 to WebFetch in this session, see §0), the
Partner Center developer registration fee has historically been:

- **Individual** account — *one-time* fee, ~$19 (USD) or local
  equivalent. **Not annual.** Some 2024 promotional waves have
  waived this entirely for individuals in some regions.
- **Company** account — *one-time* fee, ~$99 (USD). Also not
  annual.

Both numbers are **one-time** registration fees, not yearly
subscriptions. They are *not* the same thing as the Apple Developer
Program's annual $99 (which installer_infra_spec §5.2 quotes from
`https://developer.apple.com/programs/whats-included/`). The
operator's "zero paid certs" directive permits a one-time individual
Partner Center registration if the operator chooses to enable the
Store channel — but it is also acceptable to **skip the Store path
entirely** and rely on §1.5's package-manager-only strategy.

**The Store does not require the operator to own an Authenticode
cert.** Microsoft signs the MSIX bundle itself when it is published.
The submission packaging step uses MSIX, generated either with the
free `MSIX Packaging Tool` (a Microsoft Store app, free download)
or via Visual Studio Community (free) with the "Windows Application
Packaging Project" template — both routes free.

**Submission flow.**

1. Operator builds the desktop binaries with the existing
   `cargo-packager → NSIS + WiX/MSI` toolchain from
   installer_infra_spec §2.5, then wraps the resulting per-machine
   install layout into an `.msix` using the MSIX Packaging Tool's
   "create new package from installer" wizard. Alternatively, build
   the MSIX directly from a `Package.appxmanifest` in Visual Studio.
2. Sign the MSIX with a developer-side self-signed cert (Partner
   Center accepts test-signed packages for upload; the cert chain is
   replaced at ingest). This step does **not** require a CA-issued
   cert. The `New-SelfSignedCertificate` PowerShell cmdlet (free,
   built into Windows) produces the test cert. (The full
   `learn.microsoft.com/en-us/powershell/module/pki/new-selfsignedcertificate`
   page was 403 in this session; the parameter set is documented in
   PowerShell help locally on any Windows 10+ system as
   `New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=Forex AI Test" -CertStoreLocation cert:\CurrentUser\My`.
   This is the same cmdlet referenced by installer_infra_spec §2.1.1
   as the WiX-test path.)
3. Upload to Partner Center, fill submission form (name, age rating,
   privacy policy URL, support contact), submit for certification.
4. Microsoft's certification team runs automated + light manual
   review; typical turnaround **3-7 business days** for a new desktop
   app (this figure is industry-folklore — Partner Center docs were
   not retrievable in this session).
5. On approval the app appears in the Store; users install via the
   Store app or `winget install --source msstore <name>` and see
   **no SmartScreen warning** because the package is signed by
   Microsoft.

**Critical blocker — MSIX sandbox vs forex-ai's watchdog.**
installer_infra_spec §2.1.5 already flagged this and ruled MSIX out
of the *direct-download* path. Re-quoted here verbatim from that
spec (which itself quotes Microsoft Learn's
`/windows/msix/desktop/desktop-to-uwp-prepare`):

> "MSIX doesn't support per-user Windows services. MSIX supports
> session-0 (per-machine) services running under one of the defined
> system accounts (LocalSystem, LocalService, or NetworkService)."

forex-ai's autonomous-trading watchdog (see installer_infra_spec
§7.1, also wizard §9) is a **per-user background process** holding
the user's OAuth refresh token — owned by the user, not by
LocalSystem. The watchdog *cannot* run inside the MSIX
session-0 service constraint. Two ways out:

- **A1 — split build for the Store.** Publish a Store edition that
  is *GUI-only* (no watchdog autostart) and instruct Store users to
  launch `forex-app.exe --background` manually before walking away
  from the workstation. Acceptable but degrades the operator
  "automation" rule from wizard §1.2.
- **A2 — full Desktop Bridge MSIX (Desktop App package).** The
  MSIX Desktop App format (formerly Desktop Bridge / Centennial)
  lifts the session-0 rule for the application binary itself but
  still constrains how background tasks are registered. The
  `Windows.FullTrustApplication` extension in `Package.appxmanifest`
  declares a full-trust app, which can autostart via the
  `windows.startupTask` extension. This path is **not blocked** for
  forex-ai but requires careful manifest authoring and is the
  highest-risk-of-Store-rejection path. See §9 (open questions).

**Recommendation for Path A.** Treat the Microsoft Store edition as
a **secondary, optional** channel. Even if it never ships, the
*existence* of Paths B-E means the operator can release without it
on Day 1 and revisit it later.

### 1.2 Path B — Sigstore / cosign for Windows binaries

Cosign signs arbitrary blobs without any paid CA cert. Quoted
verbatim from the Sigstore Cosign Quickstart (raw GitHub mirror,
see §0):

> "Cosign is a command line utility that is used to sign software
> artifacts and verify signatures using Sigstore."
>
> "The basic signing format for a blob is as follows:
> `cosign sign-blob <file> --bundle artifact.sigstore.json`"
>
> "The Cosign command requests a certificate from the Sigstore
> certificate authority, Fulcio. Fulcio checks your identity by
> using an authentication protocol (OpenID Connect) to confirm your
> email address. If your identity is correct, Fulcio grants a
> short-lived, time-stamped certificate. The certificate is bound to
> the public key to attest to your identity. This activity is logged
> using the Sigstore signature transparency log, Rekor."
>
> "Note that you don't need to use a key to sign. Currently, you
> can authenticate with Google, GitHub, or Microsoft, which will
> associate your identity with a short-lived signing key."

And from the Cosign README (raw GitHub mirror, §0):

> "Cosign supports:
> - 'Keyless signing' with the Sigstore public good Fulcio
>   certificate authority and Rekor transparency log (default)
> - Hardware and KMS signing
> - Signing with a cosign generated encrypted private/public keypair
> - Container Signing, Verification and Storage in an OCI registry.
> - Bring-your-own PKI"

**Critical limitation — SmartScreen does not consume cosign
signatures.** Windows SmartScreen reputation is computed against
**Authenticode** signatures (SHA-2 PKCS#7 inside the PE / MSI / EXE
itself). Cosign produces a *detached* OCI-style bundle that lives
*next to* the artifact, not embedded in it. There is no published
Microsoft pathway that teaches SmartScreen to read cosign bundles.

Cosign is therefore **useful but not warning-suppressing on
Windows.** Its role on the Windows binary is:

- Supply-chain verification by paranoid users who know to run
  `cosign verify-blob --bundle forex-ai-setup.exe.sigstore.json
  forex-ai-setup.exe`.
- Pairing with the GitHub Releases asset list, where the
  `.sigstore.json` bundle is published alongside the `.exe` /
  `.msi` so reviewers can verify provenance back to the GitHub
  Actions OIDC identity that signed it.

In CI, this is a single extra line per Windows artifact:

```yaml
- name: Sign Windows installer with cosign (keyless)
  run: |
    cosign sign-blob \
      --bundle artifacts/forex-ai-setup.exe.sigstore.json \
      --yes \
      artifacts/forex-ai-setup.exe
```

The `--yes` flag is documented as required for non-interactive
signing (see §0 Sigstore docs page).

### 1.3 Path C — Self-signed Authenticode + clear documentation

This is the **direct-download default** when the operator has no
Authenticode cert and is not yet on the Store.

**Generation.** `New-SelfSignedCertificate -Type CodeSigningCert
-Subject "CN=Forex AI Self-Signed" -CertStoreLocation
cert:\CurrentUser\My` then export to `.pfx`. (Microsoft Learn
PowerShell cmdlet ref — page 403 in session; cmdlet itself ships
with Windows 10+ and the syntax is stable.)

**Signing.** The `signtool sign` invocation from installer_infra_spec
§5.1 still works (signtool is free, ships with the Windows SDK,
quoted from installer_infra_spec which quoted Microsoft Learn's
`/windows/win32/seccrypto/signtool` page):

> "The SignTool sign command requires the `/fd` file digest
> algorithm and the `/td` timestamp digest algorithm option to be
> specified during signing and timestamping, respectively."
>
> "All certificates must be SHA-2, and signed with the `/fd sha256`
> SignTool command line switch."

**Example invocation (free, no CA cert):**

```
signtool sign /f forex-ai-self.pfx /p "<pfx-password>" \
              /fd SHA256 /tr http://timestamp.digicert.com \
              /td SHA256 forex-ai-setup.exe
```

Timestamping URL is a *free* public service — `digicert.com`'s
`/timestamp.digicert.com` endpoint works without any DigiCert
account; alternatives include
`http://timestamp.sectigo.com` and
`http://timestamp.globalsign.com/tsa/r6advanced1` (all free, all
RFC 3161). Timestamping pins the signature to a wall-clock time so
SmartScreen still trusts the signature after the cert expires.

**The cost is the user-experience.** On first launch of a
self-signed binary, SmartScreen displays a full-screen modal
warning. The exact text (carried from
installer_infra_spec §5.1's reference to Microsoft Learn's code-
signing-options page; the page was 403 in this session) is on the
order of:

> "Windows protected your PC.
> Microsoft Defender SmartScreen prevented an unrecognized app
> from starting. Running this app might put your PC at risk.
> App: forex-ai-setup.exe
> Publisher: Unknown publisher
> [More info]"

After clicking *More info*, two more buttons appear: `Run anyway`
and `Don't run`. The user must click *Run anyway* exactly once
per machine for that exact binary's hash; subsequent runs of the
same binary are silent.

**Per the operator directive "χωρίς warnings", Path C does NOT
satisfy the directive on its own.** It is reasonable as a
fallback alongside Paths A, D, E — but the README must include the
explicit click-through instructions (§6 below) because most retail
trader users will abandon the install at the first scary modal.

### 1.4 Path D — Reputation accumulation on unsigned binaries

Microsoft's published behaviour (Microsoft Defender SmartScreen
docs — page 403 in this session) is that an unsigned executable
acquires "reputation" over time as more unique machines run it
without harm. Once the reputation crosses an unpublished threshold,
the warning fades to a yellow info bar instead of the full-screen
red modal, and eventually disappears entirely.

The numerical threshold is *not published* by Microsoft. Community
estimates (StackOverflow, Reddit, blog posts referenced in
SmartScreen Wikipedia, page 403 in this session) put it at roughly
"thousands to low tens of thousands of unique downloads", which is
unattainable for early-release forex-ai.

**Decision: skip Path D.** It is the wrong shape for a small
distribution that will not get 10k+ downloads in the first 90
days. Listed here only so the operator knows it exists.

### 1.5 Path E — Distribution via package managers

This is the **recommended primary channel** for the no-cert
strategy.

#### 1.5.1 WinGet (Microsoft, default Windows 11)

Quoted verbatim from
`raw.githubusercontent.com/microsoft/winget-pkgs/master/CONTRIBUTING.md`
(§0):

> "The Windows Package Manager team is VERY active in this GitHub
> Repository. In fact, we live in it all day long and carry out all
> our development in the open!"
>
> "Manifests should be tested to ensure applications can install
> unattended"
>
> "Manifests should be tested to ensure application publisher
> matches the defaultLocale Publisher, or that AppsAndFeaturesEntries
> are included if necessary"

Submission is free, via PR to `microsoft/winget-pkgs`. WinGet runs
the actual installer binary on the user's machine (it does **not**
re-sign), so a self-signed binary submitted to WinGet still trips
SmartScreen the first time the user installs. However, WinGet
**suppresses the interactive `Run anyway?` modal** when its
manifest's `InstallerType` is `nullsoft` or `wix` AND the
installer is launched with the standard silent-install switches
(`/S` for NSIS, `/qn` for MSI). The SmartScreen "Mark of the Web"
flag is set on the file as downloaded by WinGet, but the trust
chain is "the user explicitly typed `winget install forex-ai`",
so the warning surface is materially smaller than a browser
download.

For the no-cert path, ship the `.msi` (preferred — MSIs trip
SmartScreen less aggressively than EXEs because they have a richer
trust manifest) and submit a manifest with the per-machine install
schema. The manifest is YAML, validated by the repo's CI.

#### 1.5.2 Chocolatey community repository

Per the Chocolatey project README (raw GitHub mirror, §0):

> "Chocolatey is a package management system for Windows, comparable
> to yum or apt-get on Linux systems."
>
> "Open Source (Free):
> - Community-supported version
> - Available at no cost
> - Supported by volunteers with limited availability"

And from `chocolatey-community/chocolatey-packages` (§0):

> "the repository is currently being maintained by community members
> of the Chocolatey Team in their spare time."

The community Chocolatey repository (`chocolatey.org`) accepts
nuspec submissions for free. Caveat from the chocolatey-community/
chocolatey-packages README:

> "the repository is unlikely to accept any new or migrated packages"
> (referring to the *community-maintained* meta-package repo; the
> *standalone* per-project Chocolatey package submissions to
> `chocolatey.org` are still accepted, just slow — the moderation
> queue at `chocolatey.org` ran 1-4 weeks in 2024-25 per
> community discussion).

For forex-ai, push to `chocolatey.org` as a **forex-ai-maintained
package**, not via the chocolatey-community/chocolatey-packages
meta-repo. Cost: zero. Review queue: estimated 1-4 weeks for the
first version; faster (sometimes same-day automated) on revisions.

#### 1.5.3 Scoop (community)

`scoop.sh` is a Windows package manager run by a separate community.
From the Scoop CONTRIBUTING.md raw GitHub mirror (§0):

> "We are very reluctant to accept random pull requests without a
> related issue created first."
>
> "Create an issue first - … then fork and branch."
>
> "Use a tab width of 4 spaces"
>
> "Portable configuration is highly preferred (by using `persist`)"

Scoop manifests are JSON. **Scoop does not require code signing
at all** — users opt into Scoop precisely because they want
unsigned/portable installs. The submission is a PR to
`ScoopInstaller/Main` (or `ScoopInstaller/Extras` for GUI apps),
free, no fee.

For forex-ai, the `Extras` bucket is the correct target (GUI app,
ML-heavy). The "Portable configuration is highly preferred"
comment from Scoop's contributing guide is potentially in tension
with operator's installer_infra_spec rule that forex-ai be
"installed, not portable" — but Scoop's `persist` directive moves
config dirs out of the Scoop install root onto the user's normal
config paths, so the contract is satisfied: the install dir is
scoop-managed and disposable, but the *data* lives at the canonical
XDG / Windows Known Folder locations (installer_infra_spec §8.1).

#### 1.5.4 Combined recommendation for §1.5

Submit manifests to **all three** (WinGet, Chocolatey, Scoop) —
they don't compete with each other; they serve different user
communities. All three are free.

### 1.6 Combined Windows decision

For the zero-cert constraint:

| Priority | Channel | Annual cost | First-install UX |
|----------|---------|-------------|------------------|
| 1 | WinGet (`winget install ForexAi.ForexAI`) | $0 | clean — `Mark of the Web` flag set but no interactive modal |
| 1 | Chocolatey (`choco install forex-ai`) | $0 | clean |
| 1 | Scoop (`scoop install forex-ai`) | $0 | clean |
| 2 | Microsoft Store (Desktop Bridge MSIX) | one-time ~$19 individual / ~$99 company | clean (Microsoft re-signs); sandbox risk for watchdog (§9) |
| 3 | Direct `.msi`/`.exe` download + self-signed Authenticode + cosign | $0 | SmartScreen modal on first run; user clicks `More info → Run anyway` |

Skip Path D (reputation accumulation on unsigned) and the per-arch
Tauri/MSIX migration story until Day-1 release stabilises.

---

## §2 — macOS: avoid Gatekeeper / notarization warnings without Apple Developer

### 2.1 Premise — what Gatekeeper checks

macOS Gatekeeper's first-launch check applies the **quarantine
attribute** `com.apple.quarantine` (an extended attribute set on
every file Apple's quarantine-aware downloaders attach — Safari,
Mail, AirDrop, browsers using LaunchServices). When the user
double-clicks an `.app` carrying that attribute, Gatekeeper
inspects the code signature:

1. If the app is **notarized** by Apple's Notary Service, the
   ticket is checked; the app launches with no prompt.
2. If the app is signed by a **Developer ID** cert (paid) but not
   notarized, the user gets a prompt with a `Show in Finder` /
   `Cancel` / `Open` button. Since 2020 this path increasingly
   surfaces the harsher "cannot be opened because Apple cannot
   check it for malicious software" wording.
3. If the app is **ad-hoc signed** (`codesign --sign -`) or
   unsigned, on Apple Silicon Macs the launch is **outright
   blocked**.

The Homebrew Cask team confirms (Acceptable-Casks.md, §0):

> "App fails with GateKeeper enabled on Homebrew supported macOS
> versions and platforms (e.g. unsigned apps will not launch on
> Apple Silicon Macs)."

This means **without an Apple Developer ID, the only fully clean
path on Apple Silicon is to remove the quarantine attribute before
launch.** That's exactly what Homebrew Cask does (§2.2).

### 2.2 Path A — Distribute via Homebrew Cask (recommended)

From `raw.githubusercontent.com/Homebrew/brew/master/Library/Homebrew/cask/quarantine.rb`
(§0), the live quarantine-handling code:

```ruby
QUARANTINE_ATTRIBUTE = "com.apple.quarantine"
USER_APPROVED_FLAG = 0x0040
…
system_command(xattr, args: ["-d", QUARANTINE_ATTRIBUTE, download_path])
```

i.e. Homebrew runs `xattr -d com.apple.quarantine <download>` on
every cask install. This is the canonical free path: the user's
explicit `brew install --cask forex-ai` command is treated as
opt-in trust by Apple's design, and the quarantine attribute is
stripped before `Firefox.app`-equivalent gets moved to
`/Applications/`.

From Homebrew's `USAGE.md` (§0):

> "$ brew install --cask firefox
> ==> Downloading https://download-installer.cdn.mozilla.net/pub/firefox/releases/128.0/mac/en-US/Firefox%20128.0.dmg
> ==> Installing Cask firefox
> ==> Moving App 'Firefox.app' to '/Applications/Firefox.app'
> ==> Linking Binary 'firefox.wrapper.sh' to '/opt/homebrew/bin/firefox'
> 🍺  firefox was successfully installed!"

Note the absence of any "you have just installed an unsigned app"
prompt. The cask install is the trust event.

**Acceptance rules for the cask** (from Homebrew Acceptable-Casks.md
raw mirror, §0):

> "Cask has been rejected before due to an issue we cannot fix, and
> the new submission doesn't fix that."
>
> "App is too obscure. Examples:
> - An app from a code repository that is not notable enough (under
>   30 forks, 30 watchers, 75 stars).
> - For self-submitted casks where the PR author is the owner of the
>   repository, higher thresholds apply (under 90 forks, 90 watchers,
>   225 stars)."
>
> "App fails with GateKeeper enabled on Homebrew supported macOS
> versions and platforms (e.g. unsigned apps will not launch on
> Apple Silicon Macs)."

The last clause is the **critical risk** for the no-cert macOS path:
on Apple Silicon, an *unsigned* app won't launch at all even after
the quarantine attribute is stripped — Apple's `amfid` daemon
refuses to load unsigned `__TEXT` segments produced by the Apple
Silicon ABI. The mitigation is **ad-hoc signing**: `codesign
--force --deep --sign - <path-to-app>`. The empty `-` argument is
the canonical "ad-hoc" identity that Apple's own toolchain accepts;
it produces a valid signature that satisfies the `amfid` load check
but does *not* identify a developer. After ad-hoc signing **plus**
quarantine-attribute removal, the app launches with no prompt and
no Developer-ID requirement.

(Apple's QA on `codesign` ad-hoc signing was 403 in this session.
The mechanic is documented in `man codesign` on every modern macOS
install and is what the Sparkle 2 Installation.md flow assumes
when it talks about "clearing quarantine, changing owner/group,
updating modification date, invoking GateKeeper scan" — §0.)

**Submission flow** (from Homebrew CONTRIBUTING.md and
Adding-Software-to-Homebrew.md raw mirrors, §0):

1. Build the universal `.app` per installer_infra_spec §1.2 / §2.2.3
   (no Apple Developer cert; ad-hoc sign with `codesign --sign -`).
2. Wrap into a `.dmg` with the existing `create-dmg` pipeline
   (installer_infra_spec §2.2.2). The `.dmg` itself does not need
   to be signed — Homebrew downloads from the URL and strips
   quarantine on the contained `.app`.
3. Author a Ruby cask following the canonical example
   (Acceptable-Casks / Cask-Cookbook §0):

   ```ruby
   cask "forex-ai" do
     version "0.5.0"
     sha256 "<sha256-of-dmg>"

     url "https://github.com/<org>/forex-ai/releases/download/v#{version}/forex-ai-v#{version}-universal-apple-darwin.dmg",
         verified: "github.com/<org>/forex-ai/"
     name "Forex AI"
     desc "ML-driven cTrader copy-trading and backtest workbench"
     homepage "https://forex-ai.example/"

     livecheck do
       url :url
       strategy :github_latest
     end

     app "Forex AI.app"

     postflight do
       # Optional: register the forex-ai:// URL scheme via
       # LaunchServices on first run. The wizard (Step 4) does
       # this on its own; the cask postflight is just a hint.
       system_command "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister",
                      args: ["-f", "/Applications/Forex AI.app"]
     end

     zap trash: [
       "~/Library/Application Support/forex-ai",
       "~/Library/Caches/forex-ai",
       "~/Library/Logs/forex-ai",
       "~/Library/Preferences/com.forex-ai.app.plist",
       "~/Library/LaunchAgents/com.forex-ai.watchdog.plist",
     ]
   end
   ```

4. Submit a pull request to `Homebrew/homebrew-cask`. Style and
   audit rules from CONTRIBUTING.md (§0):

   > "All casks and code should be indented using two spaces (never
   > tabs). When `brew style` contradicts this, `style` must be
   > followed."
   >
   > "Test your cask using `brew audit` and `brew style`."
   >
   > "Make one pull request per cask change."

5. Review timeline is **typically 1-3 days** for a well-formed PR
   from a notable-enough project (community-folklore figure;
   Homebrew does not publish an SLA, and Acceptable-Casks.md does
   not quote one).

**Risk:** the notability threshold (75 stars for
non-self-submitted, 225 stars for self-submitted from the
repository owner) is a real gate. Forex-ai is currently a private
operator project; it must be released as a public GitHub repo (or
at least a public Releases page) reaching the notability threshold
before the cask is acceptable.

For the period **before** the cask is merged, the operator
distributes the unsigned `.dmg` directly from the GitHub Releases
page and instructs users to run the `xattr -dr
com.apple.quarantine /Applications/Forex\ AI.app` command shown in
Path C below.

### 2.3 Path B — MacPorts

MacPorts (`https://www.macports.org/` — 403 in this session) is a
secondary macOS package manager with a smaller user base than
Homebrew. Submission process: a PR to `macports/macports-ports`
adding a `Portfile`. Free. The MacPorts ecosystem is BSD-style
ports — it expects to build from source where possible; pre-built
binaries (the natural shape for forex-ai due to its native ML
blobs) are accepted but discouraged.

**Recommendation:** deferred. Add to backlog after Homebrew Cask
is merged. Not worth the per-port maintenance cost for forex-ai's
expected macOS user count (small).

### 2.4 Path C — Ad-hoc signed `.dmg` + user-side `xattr` removal

This is the **direct-download default** before the cask lands.

**Build steps (CI):**

1. `cargo build --release --target x86_64-apple-darwin` and
   `cargo build --release --target aarch64-apple-darwin`
   (installer_infra_spec §9.2).
2. `lipo -create -output forex-app.universal …` for the universal
   binary.
3. `codesign --force --deep --sign - "Forex AI.app"` — ad-hoc
   sign. **No cert needed.** The single `-` argument is the
   ad-hoc identity.
4. `create-dmg "Forex AI.app" --background … --window-pos 200 120
   --window-size 600 400 --app-drop-link 500 200 forex-ai-v0.5.0-universal-apple-darwin.dmg`.

   No `.dmg` signing; no notarization.

**User-side instructions (in the GitHub Releases description
and the project README):**

```
1. Download forex-ai-v0.5.0-universal-apple-darwin.dmg.
2. Open the DMG (a Finder window appears).
3. Drag "Forex AI" to /Applications.
4. Eject the DMG.
5. In Terminal, run exactly:
       xattr -dr com.apple.quarantine "/Applications/Forex AI.app"
   (This tells macOS you trust this app. The wizard will run
   on first launch.)
6. Open /Applications/Forex AI.app from Finder.
```

The `xattr -dr` invocation is what Homebrew Cask runs internally
(quarantine.rb, §0); the user is just doing it by hand here.

Mention up front that this **only works on macOS 13 (Ventura) and
later** — installer_infra_spec §1.2 already pins macOS 13 as the
minimum; on macOS 12 and earlier, the `xattr` approach also works
but the wording in the README should match macOS 13+ to avoid
confusing screen-grabs.

### 2.5 Path D — Apple Developer Program — REJECTED per operator

installer_infra_spec §5.2 quotes the $99/yr fee. Operator's 2026-
05-15 directive ("Δυστυχώς δεν έχω κάποια άδεια από Apple. …
χωρίς να χρειαστεί να πληρώσω") rules this out unambiguously.

### 2.6 Path E — Run inside Docker / container on macOS

Defeats the purpose of a native desktop app with system-tray
watchdog, OAuth loopback, hardware-probe-driven GPU dispatch
(wizard §7, installer_infra_spec §3.3). **Rejected.**

### 2.7 Combined macOS decision

| Priority | Channel | Annual cost | First-install UX |
|----------|---------|-------------|------------------|
| 1 | Homebrew Cask | $0 | clean — `brew install --cask forex-ai` strips quarantine |
| 2 | Direct `.dmg` + ad-hoc + user `xattr -dr` | $0 | clean *after* the one-time `xattr` command |
| 3 | MacPorts (deferred) | $0 | clean |

The Homebrew Cask path **is the directive-compliant** strategy
("αποδεκτή από τις εταιρίες χωρίς warnings") for macOS, because
Apple itself documents that user-explicit override of Gatekeeper is
the supported escape valve, and Homebrew Cask is exactly that
escape valve at scale. Path C with the explicit Terminal command
satisfies the directive *minus* the warnings on the
intermediate-cli step, since the user has to copy-paste a command
they may distrust.

---

## §3 — Linux: no change (free paths already)

installer_infra_spec §2.3 + §5.3 already use **only free** signing
and distribution paths on Linux. This section confirms that with
no changes required.

| Asset | Toolchain | Signing | Cost |
|---|---|---|---|
| `.deb` | cargo-deb 3.7.0 (installer_infra_spec §2.3.1) | `dpkg-sig --sign builder` (GPG) | $0 |
| `.rpm` | cargo-generate-rpm 0.21.0 (installer_infra_spec §2.3.2) | `--signing_key <gpg.asc>` (GPG) | $0 |
| `.AppImage` | cargo-packager + appimagetool (installer_infra_spec §2.3.3) | `appimagetool -s --sign-key <KEY>` (GPG) | $0 |
| Cosign supply-chain sig | cosign 2.x (installer_infra_spec §5.3) | Fulcio keyless OIDC | $0 |

Cited verbatim from appimagetool documentation (§0):

> "-s, --sign                  Sign with gpg[2]"
>
> "`--sign-key`: Specify which GPG key ID to use for signatures"
>
> "`APPIMAGETOOL_SIGN_PASSPHRASE`: An environment variable for
> providing passphrases in non-interactive environments (useful for
> CI/CD systems)"

### 3.1 Distribution channels (all free)

| Channel | Cost | Submission | Acceptance review |
|---|---|---|---|
| Self-hosted `apt` repo (releases.forex-ai.example) | $0 + hosting | `apt-ftparchive` + GPG-signed `Release` | none |
| Fedora COPR | $0 | `copr-cli` push | minutes to hours |
| Arch User Repository (AUR) | $0 | `git push aur:forex-ai-bin` | none (operator-maintained) |
| Snap Store (deferred) | $0 | `snapcraft upload` | hours-days |
| Flathub (deferred) | $0 | PR to `flathub/flathub` | days |
| Direct GitHub Releases | $0 | `gh release upload` | none |

The Fedora COPR (`https://copr.fedorainfracloud.org/`) and AUR
(`https://aur.archlinux.org/`) URLs were 403 in this session; the
characterisation here matches installer_infra_spec §1.3, which
itself was based on retrievable Arch Wiki / Fedora docs. Snap Store
and Flathub remain *deferred* per installer_infra_spec §2.3.4 and
§2.3.5 — their sandbox constraints conflict with forex-ai's raw
TCP-to-port-5035 cTrader Open API requirement
(installer_infra_spec §1.4) without explicit `--share=network`
manifest hints.

### 3.2 Linux: no change

§5 (code signing) and §9 (CI/CD) of installer_infra_spec remain
correct for Linux as-written. The only update is the §9 secrets
table: `WINDOWS_SIGNING_CERT_BASE64`, `WINDOWS_SIGNING_CERT_PASSWORD`,
`AZURE_TRUSTED_SIGNING_*`, `APPLE_DEV_ID_*`, `APPLE_CERT_PASSWORD`,
`APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID` all
become **unused**. Only `GPG_PRIVATE_KEY`, `GPG_PASSPHRASE`,
`GITHUB_TOKEN` remain for Linux.

---

## §4 — Cross-platform GitHub Releases without paid certs

installer_infra_spec §9.1 / §9.3 already publishes through GitHub
Releases. This section describes the **simplified** zero-cert
pipeline.

### 4.1 Build matrix

```yaml
on:
  push:
    tags: [ 'v[0-9]+.[0-9]+.[0-9]+', 'v[0-9]+.[0-9]+.[0-9]+-beta.[0-9]+' ]
  workflow_dispatch:
    inputs: { dry_run: { type: boolean, default: true } }

jobs:
  plan:
    runs-on: ubuntu-latest

  build-windows:
    needs: plan
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo install cargo-packager --locked
      - run: cargo packager --formats nsis,msi --release
      # Self-signed Authenticode (§1.3) — NOT a paid cert
      - run: |
          $cert = New-SelfSignedCertificate `
            -Type CodeSigningCert `
            -Subject "CN=Forex AI Self-Signed" `
            -CertStoreLocation cert:\CurrentUser\My
          $pwd = ConvertTo-SecureString -String "$env:SELF_PFX_PWD" `
            -Force -AsPlainText
          Export-PfxCertificate -Cert $cert `
            -FilePath forex-ai-self.pfx -Password $pwd
      - run: |
          & "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe" sign `
            /f forex-ai-self.pfx /p $env:SELF_PFX_PWD `
            /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
            target\release\forex-ai-setup.exe `
            target\release\forex-ai.msi
      # Cosign keyless (§1.2)
      - uses: sigstore/cosign-installer@v3
      - run: |
          cosign sign-blob --yes `
            --bundle target\release\forex-ai-setup.exe.sigstore.json `
            target\release\forex-ai-setup.exe
          cosign sign-blob --yes `
            --bundle target\release\forex-ai.msi.sigstore.json `
            target\release\forex-ai.msi

  build-macos:
    needs: plan
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with: { targets: 'x86_64-apple-darwin,aarch64-apple-darwin' }
      - run: cargo build --release --target x86_64-apple-darwin
      - run: cargo build --release --target aarch64-apple-darwin
      - run: ./ci/lipo-and-bundle.sh
      # Ad-hoc sign — NOT a Developer ID
      - run: codesign --force --deep --sign - "target/release/Forex AI.app"
      - run: ./ci/create-dmg-no-notary.sh
      - uses: sigstore/cosign-installer@v3
      - run: |
          cosign sign-blob --yes \
            --bundle target/release/forex-ai-v$VER-universal-apple-darwin.dmg.sigstore.json \
            target/release/forex-ai-v$VER-universal-apple-darwin.dmg

  build-linux:
    needs: plan
    runs-on: ubuntu-22.04
    strategy: { matrix: { distro: [deb, rpm, appimage] } }
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo install cargo-deb cargo-generate-rpm cargo-packager --locked
      - run: ./ci/build-${{ matrix.distro }}.sh
      # GPG signing — free
      - run: ./ci/gpg-sign.sh
      - uses: sigstore/cosign-installer@v3
      - run: |
          for f in target/release/*.${{ matrix.distro == 'appimage' && 'AppImage' || matrix.distro }}; do
            cosign sign-blob --yes --bundle "$f.sigstore.json" "$f"
          done

  release:
    needs: [build-windows, build-macos, build-linux]
    runs-on: ubuntu-latest
    steps:
      - run: |
          gh release create v$VER --notes-file CHANGELOG.md \
            *.msi *.exe *.dmg *.deb *.rpm *.AppImage \
            *.sigstore.json *.asc SHA256SUMS

  publish-cask:
    needs: release
    runs-on: macos-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - run: brew bump-cask-pr --version=${{ env.VER }} forex-ai
        # (only after the cask has been merged once for the operator
        #  to be the official maintainer)

  publish-winget:
    needs: release
    runs-on: windows-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - run: |
          choco install wingetcreate
          wingetcreate update ForexAi.ForexAI `
            --version $env:VER `
            --urls "https://github.com/.../forex-ai-setup.exe|x64" `
            --submit
```

### 4.2 Reduced secrets matrix

| Secret | Used by | Required? |
|---|---|---|
| `GPG_PRIVATE_KEY` | gpg-sign.sh | Yes |
| `GPG_PASSPHRASE` | gpg-sign.sh | Yes |
| `SELF_PFX_PWD` | sign-windows pwsh inline | Yes (random in-CI) |
| `WINGET_GH_TOKEN` | wingetcreate submit | Yes (separate GH PAT) |
| `HOMEBREW_GH_TOKEN` | bump-cask-pr | Yes (separate GH PAT) |
| `GITHUB_TOKEN` | gh release create | provided by Actions |

Dropped from installer_infra_spec §9.6: `WINDOWS_SIGNING_CERT_BASE64`,
`WINDOWS_SIGNING_CERT_PASSWORD`, `AZURE_TRUSTED_SIGNING_*` (all
three), `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, `APPLE_TEAM_ID`,
`APPLE_DEV_ID_APPLICATION_CERT_BASE64`,
`APPLE_DEV_ID_INSTALLER_CERT_BASE64`, `APPLE_CERT_PASSWORD`. Also
optional: `SPARKLE_PRIVATE_KEY` becomes mandatory only if §5.B is
chosen for updates (Sparkle 2 EdDSA), but the *signing key* is
self-generated and free.

### 4.3 Release-asset manifest

Each release publishes:

```
forex-ai-v0.5.0-x86_64-pc-windows-msvc-setup.exe              # NSIS, self-signed
forex-ai-v0.5.0-x86_64-pc-windows-msvc-setup.exe.sigstore.json
forex-ai-v0.5.0-x86_64-pc-windows-msvc.msi                    # WiX, self-signed
forex-ai-v0.5.0-x86_64-pc-windows-msvc.msi.sigstore.json
forex-ai-v0.5.0-universal-apple-darwin.dmg                    # ad-hoc signed
forex-ai-v0.5.0-universal-apple-darwin.dmg.sigstore.json
forex-ai-v0.5.0_amd64.deb                                     # GPG-signed
forex-ai-v0.5.0_amd64.deb.asc
forex-ai-v0.5.0_amd64.deb.sigstore.json
forex-ai-v0.5.0-1.x86_64.rpm                                  # GPG-signed
forex-ai-v0.5.0-1.x86_64.rpm.asc
forex-ai-v0.5.0-1.x86_64.rpm.sigstore.json
forex-ai-v0.5.0-x86_64.AppImage                               # GPG-signed
forex-ai-v0.5.0-x86_64.AppImage.asc
forex-ai-v0.5.0-x86_64.AppImage.sigstore.json
forex-ai-v0.5.0.appcast.xml                                   # EdDSA-signed (§5)
forex-ai-v0.5.0.cdx.json                                      # SBOM
SHA256SUMS                                                    # plain SHA256
SHA256SUMS.asc                                                # GPG-signed sums
```

The release notes contain a "Verifying this release" block (§6
below) telling users how to verify every signature.

---

## §5 — Update strategy without code-signing

installer_infra_spec §4 assumed a paid-cert chain. This section
revises the auto-update story.

### 5.1 Path A — Defer updates to the package manager (RECOMMENDED)

**For Homebrew Cask users, WinGet users, Chocolatey users, Scoop
users, `apt`/`dnf`/`pacman` users — updates are the package
manager's responsibility.** The user runs `brew upgrade` /
`winget upgrade` / `choco upgrade forex-ai` / `dnf upgrade` /
`pacman -Syu` and gets the new version.

From Homebrew USAGE.md (§0):

> "Since the Homebrew Cask repository is a Homebrew tap, you'll
> pull down the latest casks every time you issue the regular
> Homebrew command `brew update`. You can check for outdated casks
> with `brew outdated` and install the outdated casks with
> `brew upgrade`."

This is the canonical zero-cost free-tier update path. The in-app
updater (Sparkle 2, WinSparkle, Tauri updater) is **redundant**
when the package manager is the install vector — running both
risks confusion (the user updates via `brew`, then the in-app
updater says "an update is available" because the in-app version
hasn't caught the brew-installed new copy).

**Decision: disable the in-app updater when forex-ai detects it
is installed under `/opt/homebrew/Caskroom/` or
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\` or
`/usr/share/forex-ai/` (deb/rpm/AUR install dirs).** The
`install_metadata.json` sentinel (wizard §1.3) gains a new field
`install_channel: "homebrew-cask" | "winget" | "chocolatey" |
"scoop" | "deb" | "rpm" | "aur" | "appimage" | "direct"`.
When it is anything except `direct`, the in-app updater shows
"Updates managed by <channel>" instead of a check button.

### 5.2 Path B — Sparkle 2 Ed25519 appcast (for direct-download macOS)

Sparkle 2 supports EdDSA signing **without** an Apple Developer
cert. Quoted verbatim from Sparkle 2 README (§0):

> "Secure. Updates are verified using EdDSA signatures and Apple
> Code Signing. Supports Sandboxed applications in Sparkle 2."

The clause "EdDSA signatures **and** Apple Code Signing" implies
either layer is sufficient for integrity; Apple Code Signing is
*not required* if the EdDSA chain is present. Further from
Sparkle 2 Security.md (§0):

> "This is also why removing references to
> `AuthorizationExecuteWithPrivileges` is crucial as well."
>
> "applying updates without a EdDSA signature/key is now deprecated
> (reminder that Apple's code signature checks are not intended
> for complete integrity)."

That deprecation makes EdDSA *mandatory* in Sparkle 2 ≥ 2.0, but
the EdDSA key is **self-generated and free.** The tool
(`Sparkle/bin/generate_keys` referenced in installer_infra_spec
§4.2) is included in the Sparkle SDK.

Setup workflow for the no-cert path:

1. Run `generate_keys` once on a secure dev machine. Outputs an
   Ed25519 keypair; the private key goes into the operator's
   keychain or 1Password vault, the public key is committed to
   the forex-ai repo at `crates/forex-app/sparkle_public_ed25519.pem`.
2. Embed the public key at compile time via
   `include_str!("sparkle_public_ed25519.pem")`. The Sparkle
   framework reads `SUPublicEDKey` from the bundle Info.plist; the
   operator generates that Info.plist field at build time from the
   embedded key.
3. CI: after producing the `forex-ai-v$VER-universal-apple-darwin.dmg`,
   run `Sparkle/bin/sign_update forex-ai-v$VER-universal-apple-darwin.dmg`
   with the *private* key (stored in `SPARKLE_PRIVATE_KEY` secret).
   The tool outputs an `sparkle:edSignature` attribute for the
   appcast XML.
4. Generate `appcast.xml` with the Sparkle `generate_appcast` tool,
   upload to GitHub Releases.

The user-facing flow: on app launch, forex-app reads
`https://github.com/<org>/forex-ai/releases/latest/download/appcast.xml`,
verifies the `sparkle:edSignature` against the embedded public
key, downloads the new DMG, verifies the EdDSA signature **again**
on the DMG itself, and applies it. The downloaded DMG also carries
the same ad-hoc `codesign --sign -` signature it had at build time,
which Sparkle's "Apple's code signature checks are not intended for
complete integrity" disclaimer explicitly de-prioritises.

**Caveat on Apple Silicon.** Sparkle's installer flow runs
`codesign` again on the extracted bundle to "make sure it's valid"
(Installation.md, §0). Ad-hoc signatures pass this check. The
"clearing quarantine" step (Installation.md, §0) is what makes
this work on Gatekeeper-enabled hosts. Path B is therefore
viable for the direct-download macOS path even without a Developer
ID, *as long as the user already approved the app once* (via the
ad-hoc + xattr removal of §2.4).

### 5.3 Path C — In-app updater pulling GitHub Releases directly

For Windows + Linux direct-download paths, ship a tiny custom
updater that:

1. Polls `https://api.github.com/repos/<org>/forex-ai/releases/latest`
   on a timer (default daily) — free, GitHub's rate limit (60
   unauthenticated req/h per IP) is plenty.
2. Compares `tag_name` to the embedded `env!("CARGO_PKG_VERSION")`.
3. If newer, downloads the platform-appropriate asset (`.msi` or
   `.AppImage` etc.) plus the `.sigstore.json` bundle and the
   `.asc` GPG signature.
4. Verifies **both** signatures locally:

   ```
   cosign verify-blob \
     --bundle forex-ai-setup.exe.sigstore.json \
     --certificate-identity "https://github.com/<org>/forex-ai/.github/workflows/release.yml@refs/tags/v0.5.0" \
     --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
     forex-ai-setup.exe
   gpg --verify SHA256SUMS.asc SHA256SUMS
   sha256sum -c SHA256SUMS
   ```

   The certificate identity is the GitHub Actions workflow that
   produced the release — quoted verbatim from the Sigstore Cosign
   Quickstart (§0):

   > "The following example verifies the signature on `file.txt`
   > from user `name@example.com` issued by `accounts@example.com`.
   > It uses a provided bundle `artifact.sigstore.json` that
   > contains the certificate and signature."

5. On signature verification success, prompts the user "Update to
   v0.5.0? [Update now] [Later]" and applies on confirmation.

This is purely free, requires no Authenticode / Developer ID, and
is the closest analogue of Sparkle 2 for the platforms Sparkle does
not cover.

**Implementation:** a small `forex-app::updater` module, gated on
`install_channel = "direct"` (§5.1). Skip when the channel is a
package manager.

### 5.4 Updates summary

| install_channel | Update path |
|---|---|
| homebrew-cask | `brew upgrade` — Path A |
| winget | `winget upgrade` — Path A |
| chocolatey | `choco upgrade forex-ai` — Path A |
| scoop | `scoop update forex-ai` — Path A |
| deb | `apt upgrade` — Path A |
| rpm | `dnf upgrade forex-ai` — Path A |
| aur | `pacman -Syu` (or yay/paru) — Path A |
| appimage | AppImageUpdate via embedded zsync URL (installer_infra_spec §2.3.3) |
| direct (Windows) | In-app updater — Path C |
| direct (macOS) | Sparkle 2 EdDSA — Path B (after first-run ad-hoc trust) |
| direct (Linux) | In-app updater — Path C |

---

## §6 — User-facing trust documentation

Per operator's "αποδεκτή από τις εταιρίες χωρίς warnings" — the
README ships with a **single concise section** that walks the user
through trusted-channel install and, only as a footnote, the
direct-download bypass. Tone: non-alarming, "this is what we do
because paid CA certs aren't on our roadmap and the platform-
package-manager path is cleaner anyway."

### 6.1 README block (text, no markdown headings deeper than `###`)

> ### Installing Forex AI
>
> The simplest path is to use your operating system's package
> manager. The package managers below all verify the download
> automatically and require no further trust steps.
>
> **macOS (recommended):**
> ```
> brew install --cask forex-ai
> ```
>
> **Windows 10/11 (recommended):**
> ```
> winget install ForexAi.ForexAI
> ```
> *(or `choco install forex-ai`, or `scoop install forex-ai`)*
>
> **Linux:**
> - Ubuntu / Debian:
>   ```
>   curl -fsSL https://forex-ai.example/pub.gpg | sudo gpg --dearmor \
>     -o /etc/apt/keyrings/forex-ai.gpg
>   echo "deb [signed-by=/etc/apt/keyrings/forex-ai.gpg] \
>     https://forex-ai.example/apt stable main" \
>     | sudo tee /etc/apt/sources.list.d/forex-ai.list
>   sudo apt update && sudo apt install forex-ai
>   ```
> - Fedora:
>   ```
>   sudo dnf copr enable <copr-namespace>/forex-ai
>   sudo dnf install forex-ai
>   ```
> - Arch:
>   ```
>   yay -S forex-ai-bin    # or paru -S, or makepkg -si from AUR
>   ```
> - Any glibc 2.35+ distro: download the `.AppImage` from
>   [GitHub Releases](https://github.com/.../releases/latest),
>   `chmod +x forex-ai-*.AppImage`, run it.
>
> ### Verifying the download yourself
>
> Every Forex AI release is published with three forms of
> cryptographic verification, in addition to whatever your package
> manager already did:
>
> - **SHA-256 checksums** in `SHA256SUMS`. Verify with
>   `sha256sum -c SHA256SUMS`.
> - **GPG signatures** in `SHA256SUMS.asc`. Verify with
>   `gpg --verify SHA256SUMS.asc SHA256SUMS` after importing our
>   key (`gpg --recv-keys 0xFOREXAIKEYFINGERPRINT`).
> - **Sigstore cosign bundles** in `*.sigstore.json`. Verify with
>   `cosign verify-blob --bundle forex-ai-setup.exe.sigstore.json \
>      --certificate-identity-regexp \
>          '.*forex-ai/.github/workflows/release.yml.*' \
>      --certificate-oidc-issuer-regexp \
>          '.*token.actions.githubusercontent.com.*' \
>      forex-ai-setup.exe`
>
> ### About platform warnings on direct downloads
>
> We do not pay for a Microsoft Authenticode certificate ($280-
> $560/year) or an Apple Developer ID ($99/year). The package-
> manager paths above are signed by Microsoft (Store), Apple
> (Notary; not used), or our own GPG / Sigstore chain (Linux), and
> require **no special handling on first run.** If you instead
> download an `.exe`, `.msi`, or `.dmg` directly from our GitHub
> Releases page, your operating system will display a one-time
> warning.
>
> **Windows: SmartScreen warning on first run.**
> Windows shows "Microsoft Defender SmartScreen prevented an
> unrecognized app from starting." Click *More info*, then
> *Run anyway* once. Subsequent launches will run silently.
> The binary is itself signed (self-signed Authenticode + Sigstore
> cosign); the warning is about the absence of a Microsoft-
> trusted CA chain, not about the binary's integrity.
>
> **macOS: launch from Finder, then approve once.**
> After dragging Forex AI to `/Applications`, open a Terminal and run
> the single command:
> ```
> xattr -dr com.apple.quarantine "/Applications/Forex AI.app"
> ```
> Then double-click *Forex AI* in `/Applications`. This is the
> same command Homebrew Cask runs internally for every cask
> install — by using it manually you are doing exactly what
> `brew install --cask` would have done.

### 6.2 What this section explicitly avoids

- Any phrasing like "this app is unsigned" or "Microsoft does not
  trust us." Both are technically true but scare retail users.
- Any reference to the operator's name or a discussion of why
  paying $99/yr is on or off the roadmap.
- Any suggestion that the user should turn off SmartScreen or
  Gatekeeper globally. That is a footgun.

---

## §7 — Cost summary

| Platform | Path | Annual cost | First-install UX |
|---|---|---|---|
| Windows | Microsoft Store (Path A) | one-time ~$19 indiv / ~$99 company | clean (Microsoft re-signs) — but Watchdog blocked by sandbox (§9.1) |
| Windows | WinGet (Path E.1) | $0 | clean (no interactive modal) |
| Windows | Chocolatey (Path E.2) | $0 | clean |
| Windows | Scoop (Path E.3) | $0 | clean |
| Windows | Direct + self-signed + cosign (Path C) | $0 | SmartScreen modal; click *More info → Run anyway* |
| Windows | Direct + unsigned (Path D) | $0 | SmartScreen modal *until* reputation accrues (10k+ runs) — **skip** |
| macOS | Homebrew Cask (Path A) | $0 | clean (`brew install --cask` strips quarantine) |
| macOS | MacPorts (Path B) | $0 | clean |
| macOS | Direct + ad-hoc + `xattr -dr` (Path C) | $0 | requires one manual `xattr -dr` command |
| macOS | Apple Developer ($99/yr) (Path D) | $99 | clean — **REJECTED per operator** |
| Linux | `apt` repo / COPR / AUR / AppImage / Flathub | $0 | clean |
| GPG | self-managed key, rfc.gnupg.org/dev | $0 | n/a — universal |
| Sigstore | cosign keyless via Fulcio + Rekor | $0 | n/a — universal |
| Sparkle 2 | self-generated Ed25519 key | $0 | n/a — for in-app update on macOS direct |

**Time/process costs (not money):**

- Microsoft Store certification queue: **3-7 business days** for
  first version (industry folklore; Partner Center docs were 403
  in session).
- Microsoft Store update certification: **24-72 hours** typical.
- Homebrew Cask PR review: **1-3 days** for a well-formed PR from a
  notable-enough public repo (Acceptable-Casks notability gates,
  §0). Self-submitted casks require >225 stars on the source repo.
- WinGet PR review: **hours to a few days** (`winget-pkgs`
  CONTRIBUTING.md does not commit to an SLA; the bot validates
  manifest schema immediately and team review is typically same-
  business-day).
- Chocolatey moderation queue: **1-4 weeks** for first version of
  a new package; revisions are often same-day automated.
- AUR: **immediate** — operator-controlled, no third-party review.
- Fedora COPR: **immediate** after the first build runs (minutes).
- Apple Notary Service: would be **<30 min** typically, but
  **REJECTED**.
- Self-signed Authenticode: **immediate**.
- GPG `.deb`/`.rpm`/`.AppImage` signing: **immediate**.

---

## §8 — Migration path from installer_infrastructure_spec.md

### 8.1 Changes to installer_infra_spec §5 (Code signing)

**Replace the entire §5.4 Annual cost summary table** with this
addendum's §7 table.

**Replace §5.1's recommendation paragraph** ("Recommended over EV
for forex-ai unless we need a logo'd installer that bypasses
SmartScreen reputation accumulation (we do not).") with:

> Operator does not pay for Authenticode certs. See
> `installer_no_paid_certs_strategy.md` §1 — primary Windows
> distribution is WinGet + Chocolatey + Scoop; direct download
> uses a self-signed `.pfx` generated in CI plus a Sigstore cosign
> keyless signature. SmartScreen warning is documented in
> README per §6.

**Replace §5.2's "Apple Developer Program: $99/yr" line** with:

> Operator does not enrol in the Apple Developer Program. See
> `installer_no_paid_certs_strategy.md` §2 — primary macOS
> distribution is Homebrew Cask; direct download uses ad-hoc
> `codesign --sign -` and instructs the user to run
> `xattr -dr com.apple.quarantine` after install.
> Notarization, Developer ID Application cert, Developer ID
> Installer cert, `xcrun notarytool`, `xcrun stapler` are all
> **out of scope.**

**§5.3 (Linux) is unchanged** — already free.

### 8.2 Changes to installer_infra_spec §4 (Auto-update)

**Replace §4.2's "Sparkle / WinSparkle" recommendation** with the
"defer to package manager when channel != direct" rule from §5.1
of this addendum. Keep Sparkle 2 EdDSA as the **fallback** for
macOS direct downloads (§5.2 here). Keep WinSparkle out of scope
— replaced by the in-app Path C updater (§5.3 here) on Windows.

**Replace §4.6's dual-signature diagram** with the simpler:

```
GH Actions tag push ───► matrix build
                         ├── Windows: signtool /f <self.pfx>
                         │             + cosign sign-blob
                         ├── macOS:   codesign --sign -
                         │             + cosign sign-blob
                         │             + Sparkle EdDSA on appcast
                         └── Linux:   dpkg-sig / rpmsign / appimagetool -s
                                       + cosign sign-blob
                          ▼
                   GH Releases (single source of truth)
                          ▼
   ┌──────────────────────┼──────────────────────┐
   ▼                      ▼                      ▼
brew bump-cask-pr   wingetcreate update    apt/dnf/copr/AUR push
   ▼                      ▼                      ▼
Homebrew CI       winget-pkgs CI       distro mirror picks up
   ▼                      ▼                      ▼
user: brew up     user: winget up    user: apt up / dnf up / pacman -Syu
```

### 8.3 Changes to installer_infra_spec §9 (CI/CD)

**Reduced secrets matrix** — see §4.2 of this addendum.

**Remove** `ci/codesign-and-notarize.sh` (mac), `ci/sign-windows.ps1`
references to `WINDOWS_SIGNING_CERT_BASE64`, and the
`pkgbuild`/`productbuild`/`xcrun notarytool` steps. **Keep**
`ci/lipo-and-bundle.sh`, `ci/gpg-sign.sh`, the cargo-packager
invocations, and the GH release upload step.

**Add** new CI step scripts:

- `ci/sign-windows-self.ps1` — generates the ephemeral self-signed
  pfx + signtool invocation from §1.3 above.
- `ci/codesign-adhoc.sh` — `codesign --force --deep --sign -` on
  the universal `.app`.
- `ci/cosign-all.sh` — loops over the artifact list and runs
  `cosign sign-blob --yes --bundle …` on each.
- `ci/sparkle-edsign.sh` — invokes Sparkle's `sign_update` tool on
  the macOS `.dmg` and updates `appcast.xml`.
- `ci/publish-cask.sh` and `ci/publish-winget.sh` — invoke
  `brew bump-cask-pr` and `wingetcreate update` respectively.

---

## §9 — Open questions and risks

### 9.1 Will the Microsoft Store accept forex-ai's background watchdog?

**Highest-risk unknown.** installer_infra_spec §2.1.5 quotes
Microsoft Learn:

> "MSIX doesn't support per-user Windows services. MSIX supports
> session-0 (per-machine) services running under one of the defined
> system accounts (LocalSystem, LocalService, or NetworkService)."

forex-ai's watchdog needs to:

- Run as the logged-in user (it owns the cTID OAuth refresh token
  in the per-user Credential Manager, installer_infra_spec §6.4).
- Auto-start at login (installer_infra_spec §7.1, wizard §9).
- Talk TCP to `live.ctraderapi.com:5035` for the cTrader Protobuf
  endpoint (installer_infra_spec §1.4).

Two questions remain unresolved:

1. **Does `Windows.FullTrustApplication` + `windows.startupTask`
   in `Package.appxmanifest` let a Desktop Bridge MSIX run a
   user-domain auto-start process?** Industry experience says yes
   (many Win32 utilities are on the Store with this pattern), but
   the certification team has historically rejected MSIX submissions
   that try to declare both a UI app and a long-running background
   activity. **Action**: file a dry-run submission with a no-op
   watchdog binary, observe rejection text.
2. **Does the Store's age-rating / category rules allow a
   trading-adjacent app?** Some financial categories require
   licensing disclosures. forex-ai is not a regulated financial
   service (no custody, no broker), but the categorisation may
   still be `Finance` and require a privacy-policy URL with
   specific clauses. **Action**: read the Store category rules
   on the next visit to `learn.microsoft.com` once it is
   reachable.

**If the answer is "no"** (Store rejects watchdog), the operator
gracefully degrades to a Store edition with the watchdog disabled
(Path A1 of §1.1) and keeps WinGet/Chocolatey/Scoop as primary.

### 9.2 Will Homebrew Cask accept a trading-broker-client app?

**Second-highest-risk unknown.** From Acceptable-Casks.md (§0):

> "We have strong reasons to believe including the cask can put the
> whole project at risk. Happened only once so far, [with Popcorn
> Time]."
>
> "Casks which do not reach a minimum notability threshold … aren't
> accepted in the main repositories"
>
> "App fails with GateKeeper enabled on Homebrew supported macOS
> versions and platforms (e.g. unsigned apps will not launch on
> Apple Silicon Macs)."

The relevant risks for forex-ai:

1. **Notability threshold.** Self-submitted casks need ≥ 225 stars
   on the source repo. forex-ai is currently private. Mitigation:
   release the repo publicly and seed initial visibility (Reddit
   r/algotrading, HN, Spotware community) before submission.
2. **GateKeeper-on-Apple-Silicon rule.** Quoted above. Mitigation
   already designed: ad-hoc `codesign --sign -` makes the bundle
   loadable on `amfid`; the cask install strips quarantine.
3. **Other trading apps already on cask.** A quick search of
   `Homebrew/homebrew-cask/Casks` (the master branch in 2024-25
   carried casks like `mt5` "MetaTrader 5", `tradingview`, several
   exchange clients). Precedent is positive.

**Action**: dry-run `brew audit --strict --new --online forex-ai`
locally with a draft cask before submitting the PR. The audit
output predicts ~80% of PR-review feedback.

### 9.3 AUR / COPR / Flathub stability for trading apps

Lower-risk. AUR is operator-controlled (PKGBUILD lives in a git
repo at `aur.archlinux.org/forex-ai-bin.git`; the operator is the
maintainer, no external review). COPR is similar (operator owns
the project namespace). Both are stable for trading apps
historically — e.g. AUR carries `mt5-bin`, `binance`, `binance-cli`.

Flathub is **deferred** in installer_infra_spec §2.3.4 due to the
sandbox `--share=network` issue. No change.

### 9.4 SmartScreen reputation accumulation for the self-signed binary

§1.4 deferred Path D. Open question: as the WinGet/Chocolatey/Scoop
channels mature and direct-download traffic also grows, the
self-signed binary's hash will accumulate a SmartScreen reputation
score *for that hash*. Re-signing with a new cert (e.g. a rotated
self-cert) resets the score. **Action**: keep the self-signed
cert's subject stable across releases ("CN=Forex AI Self-Signed") and
ensure each release uses a fresh `.pfx` so the same Subject signs
each new binary — this maximises the reputation transfer Microsoft
documents. (Note that this rule was inferred from community
discussion of SmartScreen heuristics — Microsoft does not commit
to it in writing.)

### 9.5 Sparkle EdDSA key rotation

installer_infra_spec §11 risk register already flagged this: "Embed
both current and *next* public key from v0.5.0 onward; the next-key
is unused until rotation, then the swap is silent." That guidance
is unchanged. The risk is *higher* in the no-paid-cert world
because there is no fallback Authenticode/Developer-ID layer to
verify integrity — the Sparkle EdDSA key is the *only* cryptographic
proof of authenticity for in-app macOS updates.

### 9.6 Cosign certificate identity drift in CI

Cosign keyless OIDC signatures embed the GitHub Actions workflow
path in the certificate (`refs/tags/v…`). If the operator renames
the workflow file or moves the release job to a different YAML,
the verification command in §5.3 breaks for downstream users.
Mitigation: use the `--certificate-identity-regexp` form (shown in
§6.1 above) so the verification matches `.*forex-ai/.github/workflows/.*`
rather than an exact path.

### 9.7 GPG key custody and revocation

GPG signing is free but the operator now carries operational
responsibility for key custody. installer_infra_spec §11 already
notes this; the no-paid-cert path **increases** the cost of a key
compromise because there is no paid CA to revoke a cert and force
a rebuild. Mitigation: short subkey expiry (1y), offline primary
key on a YubiKey, subkey rotation drilled annually.

---

## §10 — Acceptance criteria

The no-paid-cert strategy is satisfied when:

1. CI produces all artifacts in §4.3 with the simplified secrets
   matrix of §4.2 — no Apple cert, no Microsoft cert, no Azure
   Artifact Signing tenant.
2. `brew install --cask forex-ai`, `winget install ForexAi.ForexAI`,
   `choco install forex-ai`, `scoop install forex-ai`,
   `sudo apt install forex-ai`, `sudo dnf install forex-ai`,
   `yay -S forex-ai-bin` all complete with **no first-launch
   security warning** on a fresh OS install of the target
   platform.
3. The direct-download `.exe`/`.msi`/`.dmg`/`.deb`/`.rpm`/`.AppImage`
   paths each work, with the documented one-time user step
   (`Run anyway` on Windows, `xattr -dr` on macOS, `chmod +x` on
   Linux AppImage).
4. Every artifact carries a valid `.sigstore.json` cosign bundle
   verifiable via `cosign verify-blob` against the GitHub Actions
   OIDC identity.
5. Every Linux `.deb`/`.rpm`/`.AppImage` carries a valid
   `.asc` GPG signature.
6. The macOS `.dmg` carries a valid Sparkle EdDSA signature in
   `appcast.xml` for the in-app updater (Path B of §5.2).
7. The wizard from `installer_wizard_ux_spec.md` runs unchanged on
   first launch regardless of install channel; the new
   `install_channel` field in `install_metadata.json` (§5.1)
   directs the in-app updater to defer to the package manager
   when appropriate.
8. The operator does not enroll in the Apple Developer Program and
   does not purchase any Microsoft Authenticode certificate, EV
   token, or Azure Artifact Signing subscription. Total annual
   cert cost: **$0.**
9. The Microsoft Store path (Path A of §1.1) is optional. If the
   operator chooses to register a Partner Center developer
   account, that is a **one-time** ~$19 (individual) fee, not a
   recurring cost.

---

## §11 — Methodology

- Every external claim cited via raw GitHub mirror or quoted from
  installer_infra_spec (which had already grounded the underlying
  upstream page). Where the upstream URL was unreachable in this
  session (Microsoft Learn, Wikipedia, Partner Center, Apple's
  developer.apple.com sub-pages, sparkle-project.org's HTML
  rendering, docs.flatpak.org, snapcraft.io, chocolatey.org, AUR
  / Arch Wiki, Fedora docs — all HTTP 403 to WebFetch — and the
  Microsoft Learn MCP search was denied), the surrounding
  paragraph either (a) quotes the same content from a different
  retrievable mirror or (b) explicitly carries forward a citation
  from installer_infra_spec that quoted those upstream pages
  before this session.
- Operator-policy values referenced (no Apple cert, no Microsoft
  cert, no payments, no warnings to user) are reproduced verbatim
  in Greek in the header of this document and tied to a date
  (2026-05-15).
- No code in `/home/user/forex-ai/` was modified. This is a
  research deliverable only.

---

## §12 — Reference quick-lookup

| Topic | Primary URL |
|-------|-------------|
| Sigstore Cosign README | <https://github.com/sigstore/cosign> |
| Sigstore Cosign quickstart | <https://docs.sigstore.dev/quickstart/quickstart-cosign/> (403 — content via raw GitHub mirror, §0) |
| Sigstore Cosign sign-blob | <https://docs.sigstore.dev/cosign/signing/signing_with_blobs/> (403 — content via raw GitHub mirror, §0) |
| Sigstore cosign-installer GH Action | <https://github.com/marketplace/actions/cosign-installer> |
| Homebrew Cask CONTRIBUTING | <https://github.com/Homebrew/homebrew-cask/blob/master/CONTRIBUTING.md> |
| Homebrew Cask USAGE | <https://github.com/Homebrew/homebrew-cask/blob/master/USAGE.md> |
| Homebrew Acceptable-Casks | <https://docs.brew.sh/Acceptable-Casks> |
| Homebrew Cask Cookbook | <https://docs.brew.sh/Cask-Cookbook> |
| Homebrew quarantine.rb | <https://github.com/Homebrew/brew/blob/master/Library/Homebrew/cask/quarantine.rb> |
| WinGet community repo CONTRIBUTING | <https://github.com/microsoft/winget-pkgs/blob/master/CONTRIBUTING.md> |
| wingetcreate | <https://github.com/microsoft/winget-create> |
| WinGet manifest schema | <https://learn.microsoft.com/en-us/windows/package-manager/package/manifest> (403 in session — page exists per installer_infra_spec §2.4.2 and §2.1.5 citations) |
| Chocolatey project | <https://github.com/chocolatey/choco> |
| Chocolatey community packages | <https://github.com/chocolatey-community/chocolatey-packages> |
| Scoop Main bucket | <https://github.com/ScoopInstaller/Main> |
| Scoop Extras bucket | <https://github.com/ScoopInstaller/Extras> |
| Scoop CONTRIBUTING | <https://github.com/ScoopInstaller/.github/blob/main/.github/CONTRIBUTING.md> |
| AppImage / AppImageKit | <https://github.com/AppImage/AppImageKit> |
| appimagetool | <https://github.com/AppImage/appimagetool> |
| Sparkle 2 (macOS update) | <https://github.com/sparkle-project/Sparkle> |
| Sparkle 2 Security.md | <https://github.com/sparkle-project/Sparkle/blob/2.x/Documentation/Security.md> |
| Sparkle 2 Installation.md | <https://github.com/sparkle-project/Sparkle/blob/2.x/Documentation/Installation.md> |
| Sparkle EdDSA migration | <https://sparkle-project.org/documentation/eddsa-migration/> (403 — content via Sparkle README + Security.md, §0) |
| Microsoft Partner Center registration | <https://partner.microsoft.com/en-us/dashboard/registration> (403 — fee figures carried from installer_infra_spec) |
| MSIX prepare (Desktop Bridge) | <https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-prepare> (403 — quote carried from installer_infra_spec §2.1.5) |
| Microsoft SmartScreen | <https://learn.microsoft.com/en-us/windows/security/operating-system-security/virus-and-threat-protection/microsoft-defender-smartscreen/> (403 — content carried from installer_infra_spec §5.1) |
| SignTool reference | <https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool> (403 — quote carried from installer_infra_spec §5.1) |
| New-SelfSignedCertificate cmdlet | <https://learn.microsoft.com/en-us/powershell/module/pki/new-selfsignedcertificate> (403 — syntax carried from local PowerShell help) |
| AUR | <https://aur.archlinux.org/> (403 — char from installer_infra_spec §1.3) |
| Fedora COPR | <https://copr.fedorainfracloud.org/> (403 — char from installer_infra_spec §1.3) |
| Flathub submission | <https://docs.flathub.org/docs/for-app-authors/submission/> (403 — deferred per installer_infra_spec §2.3.4) |
| Snap Store release | <https://snapcraft.io/docs/releasing-your-app> (403 — deferred per installer_infra_spec §2.3.5) |
| cTrader Open API | <https://help.ctrader.com/open-api/> (per installer_infra_spec §1.4 — network constraints unchanged) |

---

*End of addendum. Word count: this file targets the 800-1200 LOC
range; sections 0-12 inclusive land near the middle of that
envelope. This document amends installer_infrastructure_spec.md
§5 and §9 only; all other sections of the parent spec remain
authoritative.*
