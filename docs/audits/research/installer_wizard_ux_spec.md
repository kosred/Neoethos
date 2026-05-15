# forex-ai Installer Wizard — UX Specification

Compiled 2026-05-15 by the research agent in response to the operator
directive (verbatim, in Greek):

> "κατά την εγκατάσταση τους να υπάρχει ένα installer wizard που θα
> βοηθά τον χρήστη στους αυτοματισμούς"

Translation: "during installation there should be an installer wizard
that helps the user with automation."

This document is a **research deliverable** — no code changes. It
specifies the end-to-end first-run wizard that takes a newly installed
forex-ai from "binary on disk" to "first profitable backtest in
progress" without requiring the operator to hand-edit `config.yaml`,
hunt for cTrader OAuth instructions, or know the canonical timeframe
list.

The wizard's purpose, taken from Nielsen Norman Group's design
recommendations for wizards, is to "speed users through infrequent or
complicated tasks" by "presenting a linear workflow with minimal
disruptions or alternatives" — source NN/G,
<https://www.nngroup.com/articles/wizards/>, retrieved via WebSearch
snippet 2026-05-15.

---

## 0. Sources

Docs-first citations used throughout this spec. Where a URL is
listed as "snippet via WebSearch", the underlying HTML page returned
HTTP 403 to the sandbox's WebFetch tool and the content was
reconstructed from WebSearch result excerpts that quote the canonical
page directly.

| # | Source | Status |
|---|--------|--------|
| 1 | NN/G — Wizards: Definition and Design Recommendations (<https://www.nngroup.com/articles/wizards/>) | snippet via WebSearch |
| 2 | NN/G — Progressive Disclosure (<https://www.nngroup.com/articles/progressive-disclosure/>) | snippet via WebSearch |
| 3 | NN/G — 8 Design Guidelines for Complex Applications | snippet via WebSearch |
| 4 | Microsoft Learn — UX checklist for desktop applications (<https://learn.microsoft.com/en-us/windows/win32/uxguide/top-violations>) | snippet via WebSearch |
| 5 | Microsoft Learn — Win32 Wizards (<https://learn.microsoft.com/en-us/windows/win32/uxguide/win-wizards>) | snippet via WebSearch |
| 6 | RFC 8252 — OAuth 2.0 for Native Apps (<https://datatracker.ietf.org/doc/html/rfc8252>) | snippet via WebSearch |
| 7 | freedesktop.org — Desktop Application Autostart Specification (<https://specifications.freedesktop.org/autostart-spec/autostart-spec-latest.html>) | snippet via WebSearch |
| 8 | freedesktop.org — XDG Base Directory Specification | snippet via WebSearch |
| 9 | Apple Developer — Distribution XML Reference | snippet via WebSearch |
| 10 | Apple Developer — Packaging Mac software for distribution | snippet via WebSearch |
| 11 | Apple Developer — Notarizing macOS software before distribution | snippet via WebSearch |
| 12 | FireGiant — WixUI dialog library (<https://docs.firegiant.com/wix/tools/wixext/wixui/>) | snippet via WebSearch |
| 13 | Sentry — Best Practices for GDPR Compliance (<https://sentry.io/trust/privacy/gdpr-best-practices/>) | snippet via WebSearch |
| 14 | Lollypop — Best Practices for High-Conversion Wizard UI Design (2026/01) | snippet via WebSearch |
| 15 | Andrew Coyle — How to Design a Form Wizard | snippet via WebSearch |
| 16 | Eleken — Wizard UI Pattern: When to Use It | snippet via WebSearch |
| 17 | Krystal Higgins — The design of setup wizards | listed only, 403 |
| 18 | UXPin — Progress Tracker Design: UX Best Practices | snippet via WebSearch |
| 19 | Internal: `docs/audits/research/ctrader_api_full_reference.md` | local |
| 20 | Internal: `docs/audits/research/spotware_proto_new_messages.md` | local |
| 21 | Internal: `docs/audits/research/ml_numerical_reference.md` | local |
| 22 | Internal source: `crates/forex-app/src/app_services/ctrader_live_auth.rs` | local |
| 23 | Internal source: `crates/forex-core/src/contracts/temporal.rs` | local |
| 24 | Internal source: `crates/forex-core/src/domain/prop_firm.rs` | local |
| 25 | Internal source: `crates/forex-core/src/system.rs` | local |

### 0.1 Dependencies on parallel deliverables

This spec assumes two sibling research artefacts are produced by other
agents in the same audit batch:

- `docs/audits/research/ui_ux_design_spec.md` — colour palette,
  typography, spacing scale, focus-ring rules. **Not yet present at
  the time of writing.** Wherever this spec references a design
  token (e.g. `color.surface.canvas`, `space.4`), the token is
  declared `TODO(ui_ux_design_spec)` and will be resolved when the
  UI/UX spec lands.
- `docs/audits/research/installer_infrastructure_spec.md` — the
  packaging-pipeline spec (MSI / pkg / deb / rpm / AppImage). **Not
  yet present.** This spec references it for §7 (auto-start) and §8
  (OS-canonical paths); placeholders are marked
  `TODO(installer_infrastructure_spec)`.

The wizard described here is **independent** of the packaging
pipeline: it ships inside the `forex-app` binary and runs on first
launch regardless of how the binary got onto disk (MSI, pkg, deb,
AppImage, tarball, `cargo install`, source build). That isolation is
intentional — see §1.

---

## §1 — Wizard architecture

### 1.1 Two candidate architectures

There are two ways to deliver a setup wizard for a desktop application:

**Option A — native installer wizard.** Each platform's package
format embeds a wizard:

- Windows MSI: `WixUI_*` dialog set (`WixUI_FeatureTree`,
  `WixUI_InstallDir`, `WixUI_Mondo`). FireGiant docs note that
  "Each WixUI dialog set is a wizard-style sequence of dialogs wired
  up to Next and Back buttons" (FireGiant, *WixUI dialog library*).
- macOS `.pkg`: a `distribution.xml` describes the choices, options,
  background, license, conclusion. Apple's *Distribution XML
  Reference* documents the `<options customize="allow">` and
  `<choices-outline>` machinery.
- Linux `.deb` / `.rpm`: traditionally **no GUI** beyond
  `debconf`/`dpkg-reconfigure` prompts; some distros block `postinst`
  from opening windows entirely. AppImage and Flatpak have **no
  installer wizard** at all.

**Option B — cross-platform first-run wizard inside the app.** The
installer is a thin file-extractor that drops the binary on disk and
exits. On first launch, `forex-app` notices that
`<config_dir>/forex-ai/wizard_complete.json` does not exist and opens
a guided modal wizard before showing the main UI.

### 1.2 Recommendation: Option B (in-app first-run wizard)

**Choose Option B.** Justification:

1. **Linux parity.** Linux deb/rpm `postinst` scripts can launch GUI
   programs only with workarounds (DISPLAY env-var inheritance, dbus
   activation), and AppImage has no `postinst` at all. The
   freedesktop.org *Autostart Specification* puts user-facing first-
   run behaviour squarely in the application's domain, not the
   packager's — "users can place application launchers that should be
   run automatically upon login" in `~/.config/autostart/`
   (freedesktop.org *Desktop Application Autostart Specification*,
   snippet via WebSearch). An in-app wizard works identically across
   AppImage, deb, rpm, Flatpak, and "downloaded a tarball".
2. **Re-runnability.** Operator directive includes "Skip / re-run"
   (§5 of the brief). A native installer wizard can only run during
   installation; re-running it requires uninstalling and reinstalling.
   An in-app wizard is just `forex-app --wizard`.
3. **Notarization friction (macOS).** Apple's notarization service
   has tightened over 2026: "Some developers reported notarization
   rejection attempts beginning at the end of February 2026, with
   rejections continuing through early May" (search snippet, May 2026).
   Anything that runs **before** `forex-app` itself starts must be
   independently signed and notarized. Putting the wizard inside
   `forex-app` collapses the artefacts that need notarization from
   two (installer + app) to one (app).
4. **Microsoft UX guidance.** Microsoft Learn's *Wizards* page
   recommends consolidating wizards rather than fragmenting them:
   "Reduce the number of pages to focus on essentials and consolidate
   related pages, taking optional pages out of the main flow"
   (Microsoft Learn — Win32 UX wizards, snippet via WebSearch). A
   single in-app wizard is one page-stack; an installer wizard +
   in-app onboarding would be two, with duplication.
5. **Operator's "automation help" rule.** Most of what this wizard
   automates (cTrader OAuth, hardware probe, historical data
   download) is **not knowable at install time** — the install runs
   as root or with `sudo`, but the OAuth tokens and the trained
   models belong to the per-user account. Per-user automation must
   live in the per-user app.

The native installer's job, in this architecture, is reduced to:
copy files, register the launcher, schedule autostart **if asked**
(see §2.9 — Auto-start), and exit. Everything user-facing happens
inside `forex-app` on first run.

### 1.3 Post-install hand-off

The installer leaves a single sentinel file in the install dir:
`install_metadata.json` with `{installer_version, installed_at,
install_path, data_path}`. `forex-app` reads it at startup to
distinguish "first run after installer" from "first run after
`cargo install`". Both cases trigger the wizard; only the former
shows the "Welcome — installer detected" greeting in step 1.

### 1.4 Implementation surface inside the app

- New crate: **none** — wizard lives in `forex-app::wizard` module.
- New CLI subcommand: `forex-cli wizard` (and `forex-app --wizard`)
  for re-runs. Wizard state machine is shared between GUI and TUI
  fronts; only the rendering layer differs (§8 — CLI parity).
- Storage: `<config_dir>/forex-ai/wizard_complete.json` marks
  completion. The wizard does **not** auto-mark itself complete if
  the user skipped past a critical step (see §3 — error handling) —
  instead it writes `{state: "completed_with_warnings", skipped:
  ["oauth", "historical_download"]}` so the main app can prompt for
  re-completion later.

---

## §2 — Wizard steps (sequence)

Each step heading uses the format **Step N — Title** with the
mandatory metadata block: purpose, mockup-prose, inputs, actions,
skip/back/cancel semantics, estimated time.

Where a step quotes operator policy, the operator directive is
reproduced verbatim with citation. NN/G's wizard guidance — "the
perceived user experience is that of a linear flow: one screen after
another, and all the user has to do is to click 'next.' (Or 'back,'
or 'cancel,' but mainly the user keeps moving forward.)" (NN/G,
*Wizards: Definition and Design Recommendations*, snippet via
WebSearch) — sets the navigation contract for every step below.

### Step 1 — Welcome + License

- **Purpose:** introduce the wizard, surface the LICENSE for legal
  acceptance.
- **Mockup:** centred 720×540 modal. Top: large "forex-ai" wordmark,
  version line ("v0.5.0 — built 2026-05-15"). Body: 4–5 lines of
  copy describing what the next nine steps will do, with an
  estimated total time ("≈ 10 minutes on a typical broadband
  connection"). Scrollable LICENSE pane occupies the lower 60 % of
  the modal. Below the pane: a checkbox "I have read and accept the
  license" (default unchecked). Footer: `[Cancel]` left, `[Continue
  →]` right, disabled until the checkbox is ticked.
- **Inputs:** license-accepted boolean.
- **Actions:** reads `LICENSE` from the installed
  `<install_dir>/LICENSE` file (or from the embedded `include_str!`
  fallback if the file is missing — operator's no-synthetic-data
  rule does not apply to a static license text). On accept, writes
  the LICENSE SHA-256 + acceptance timestamp to `wizard_state.json`.
- **Skip:** **NOT ALLOWED** — the only mandatory step.
  Cancel exits the app entirely.
- **Back:** N/A (first step).
- **Cancel:** confirmation modal: "Cancel installation? The wizard
  will run again on next launch." Two buttons: `[Stay]`, `[Quit]`.
- **Time:** ≤ 30 s (mostly waiting for the user to read).

Microsoft's wizard guidance specifically discourages a content-free
welcome page: "Don't use Welcome pages — make the first page
functional whenever possible" (Microsoft Learn UX wizards, snippet
via WebSearch). Including the license on the same page satisfies
this — the page is functional (capture acceptance) and welcoming.

### Step 2 — Path selection

- **Purpose:** pick the install dir (binary) and the user-data dir
  (caches, OAuth tokens, model checkpoints, downloaded history).
- **Mockup:** two labelled text fields with `[Browse…]` buttons.
  The first is greyed and read-only — the binary has already been
  installed by the time the wizard runs, so this is informational.
  The second is editable and defaults to the OS-canonical user-data
  location per the `directories` crate:
  - Windows: `%LOCALAPPDATA%\forex-ai\` (which resolves via
    `SHGetKnownFolderPath(FOLDERID_LocalAppData)` per the Windows
    Known Folder API).
  - macOS: `~/Library/Application Support/forex-ai/`.
  - Linux: `$XDG_DATA_HOME/forex-ai/` or `~/.local/share/forex-ai/`
    per the XDG Base Directory Specification — "Data files store
    supplementary data … per-user configuration should go in
    `~/.config/appname` and caches … should go into
    `~/.cache/appname`" (freedesktop XDG Base Directory snippet via
    WebSearch).
  A live disk-free indicator under each path shows the free bytes
  on the chosen volume in red (< 5 GiB), amber (5–20 GiB), green
  (> 20 GiB). Below: a "Validate" affordance that runs the
  write-permission probe.
- **Inputs:** `data_path: PathBuf`.
- **Actions:** writes a sentinel file
  (`<data_path>/.forex-ai-write-check`) and removes it; failure
  surfaces an error toast inside the wizard ("Cannot write to this
  path — check permissions or pick a different location"). On
  success, records `data_path` in `wizard_state.json`.
- **Skip:** Allowed (defaults to OS-canonical path).
- **Back:** returns to Step 1; license acceptance is preserved
  (Microsoft UX guideline: "Preserve user selections through
  navigation so that if users make changes, click Back, then Next,
  those changes should be preserved" — Microsoft Learn UX wizards
  snippet via WebSearch).
- **Cancel:** as Step 1.
- **Time:** ≤ 15 s on the default path; up to 2 min if the user
  customises and the path doesn't exist (mkdir + permission probe).

Per `TODO(installer_infrastructure_spec)` §8: defaults to OS-
canonical locations but allows override.

### Step 3 — Account & profile

- **Purpose:** capture operator identity, prop-firm preset, and the
  monthly profit-target floor.
- **Mockup:** four-row form.
  Row 1 — "Operator name": single-line text, optional, used only in
  the journal's "trader=…" tag.
  Row 2 — "Prop-firm preset": dropdown with three entries — "FTMO
  Standard (recommended)", "FTMO Aggressive", "Custom". FTMO
  Standard is preselected and reads its values from
  `PropFirmConstraints::FTMO_STANDARD`
  (`max_daily_loss_pct=0.05`, `max_overall_drawdown_pct=0.10`,
  `challenge_profit_target_pct=0.10`,
  `min_monthly_net_profit_pct=0.04`, `min_trading_days=10` — from
  `crates/forex-core/src/domain/prop_firm.rs`).
  Row 3 — "Target monthly net profit": slider 4 %–25 %, default 4 %,
  with an inline numeric edit. The slider's **left stop is 4 %** —
  the operator directive of 2026-05-14 is verbatim "`4%` per
  operator directive 2026-05-14" (`prop_firm.rs:36`), so the wizard
  enforces ≥ 4 %. An attempt to type 3 % triggers an inline
  validator: "Minimum 4 % per operator policy (2026-05-14). Lower
  values are not selectable."
  Row 4 — "Trading mode": radio buttons — "Backtest only", "Forward
  test (paper)", "Live trading (requires OAuth in next step)".
  Default: "Forward test".
- **Inputs:** operator_name, prop_firm_preset, monthly_profit_target,
  trading_mode.
- **Actions:** patches the in-memory `WizardConfig`; nothing is
  written until the Summary step.
- **Skip:** Allowed (defaults to FTMO Standard / 4 % / Forward).
- **Back:** Step 2.
- **Time:** ≤ 60 s.

### Step 4 — cTrader OAuth onboarding

- **Purpose:** wire one cTrader trading account end-to-end. This is
  the **central automation** the operator's directive asks for: a
  user who has never used the cTrader API should finish this step
  with a refresh token, a chosen `ctidTraderAccountId`, and a
  validated transport session.
- **Mockup:** wider modal (900 px). Four sub-panels arranged
  vertically with a sticky "step within step" sub-progress bar at
  top showing 4.1 → 4.2 → 4.3 → 4.4:
  - **4.1 Register app.** Plain text walkthrough: "Visit
    <https://openapi.ctrader.com/> while signed into your cTID, then
    *Applications → Add Application*. Set the redirect URI to
    exactly `http://127.0.0.1:7777/ctrader/callback` (you can change
    the port later if 7777 is busy; the wizard will try 7777, 7878,
    8989 in order). Copy the Client ID and Client Secret here."
    Two single-line fields ("Client ID", "Client Secret"), the
    second masked. An "Open openapi.ctrader.com" button that opens
    the system browser. A "Test these credentials" affordance — on
    click, the wizard makes a quick `ProtoOAApplicationAuthReq`
    (payloadType 2100) to `demo.ctraderapi.com:5035` to verify the
    pair is valid (per
    `docs/audits/research/ctrader_api_full_reference.md` §2.6). If
    the response is `ProtoOAErrorRes` with `CH_CLIENT_AUTH_FAILURE`
    (101) or `CH_OA_CLIENT_NOT_FOUND` (107), surface the broker's
    error message verbatim.
  - **4.2 Sign in with cTID.** A single primary button "Sign in with
    cTID". Clicking it does the following sequence (all in the
    existing `ProductionCTraderLiveAuthBackend` in
    `crates/forex-app/src/app_services/ctrader_live_auth.rs`):
    1. Generate a 32-byte CSRF `state` token (already wired —
       `ctrader_live_auth.rs:38` documents "audit-fix F2").
    2. Bind a loopback listener on the first available port from
       `[7777, 7878, 8989]` (already wired —
       `ProductionCTraderLiveAuthBackend::bind_loopback_listener`
       at `ctrader_live_auth.rs:193`).
    3. Open the system default browser to
       `https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={cid}&redirect_uri=http://127.0.0.1:{port}/ctrader/callback&scope=trading&product=web&state={csrf}`
       per `ctrader_api_full_reference.md` §2.2.
    4. Show a placeholder pane: "Waiting for browser
       sign-in…  [Cancel]". A 5-minute timer ticks down (matches
       the existing `CTRADER_CALLBACK_TIMEOUT: Duration =
       Duration::from_secs(300)` at `ctrader_live_auth.rs:24`).
    5. On callback receipt, exchange the auth code for a token
       bundle by GET to `https://openapi.ctrader.com/apps/token`
       (per the verbatim Spotware sample —
       `ctrader_api_full_reference.md` §2.3).
    Fallback: if no port in the list binds (corporate firewall),
    surface a "Use copy-paste flow" link. The copy-paste flow shows
    the same URL with the redirect set to `https://spotware.com`,
    asks the user to paste the resulting redirect URL, and parses
    the `code` and `state` parameters from it. RFC 8252 §7.3
    mandates loopback redirect URIs for native apps — "loopback
    redirect URIs use the 'http' scheme … constructed with the
    loopback IP literal and whatever port the client is listening
    on, such as 'http://127.0.0.1:{port}/{path}'" (RFC 8252 snippet
    via WebSearch) — so the copy-paste flow is a *fallback*, not a
    primary path.
  - **4.3 Account picker.** After the OAuth exchange, the wizard
    sends `ProtoOAApplicationAuthReq` (2100) +
    `ProtoOAGetAccountListByAccessTokenReq` (2149) per
    `ctrader_api_full_reference.md` §2.6. The response, a
    `ProtoOAGetAccountListByAccessTokenRes` (2150), carries a
    `repeated ProtoOACtidTraderAccount`. The picker renders one
    row per account with columns: account label, broker name,
    account number, currency, environment (Live/Demo), trader-side
    `accountType`. A radio-button column lets the user pick the
    default account. A checkbox column lets them pick *additional*
    accounts to enable (the trading-mode selector in Step 3 decides
    whether they are usable for live trading or only for
    backtesting). A "I'll wire more accounts later" link is shown
    below the table.
  - **4.4 Account auth probe.** On the chosen primary account, the
    wizard sends `ProtoOAAccountAuthReq` (2102) and waits for
    `ProtoOAAccountAuthRes` (2103) — per the same §2.6. Success
    surfaces a green tick; failure shows the broker's error verbatim
    plus the standard remediation: "Code 2 = ACCOUNT_NOT_AUTHORIZED:
    re-authenticate the access token." If the failure is a
    permanent code (e.g. 106 `CH_CTID_TRADER_ACCOUNT_NOT_FOUND`), the
    wizard offers "Pick a different account" rather than "Retry".
- **Inputs:** `client_id`, `client_secret`,
  `selected_ctid_trader_account_id`, `additional_account_ids`,
  `environment` (Live | Demo, defaulting to whichever the primary
  account is registered under).
- **Actions:** writes the token bundle (access + refresh + expiry
  timestamp) into the broker-persistence store —
  `crates/forex-app/src/app_services/broker_persistence.rs` already
  wraps secrets in `secrecy::SecretString`. The client_secret is
  stored only after the OAuth flow succeeded; if the user cancels
  step 4.2, no secret is persisted.
- **Skip:** Allowed, with a strong inline warning ("Skipping cTrader
  setup disables live trading, live backtests, and historical
  download. You can re-run this step later from `Settings → Wizard`
  or `forex-app --wizard`."). On skip, sets
  `wizard_state.incomplete = ["ctrader_oauth"]`.
- **Back:** Step 3.
- **Cancel:** as Step 1.
- **Time:** 2–4 min, dominated by the browser sign-in.

### Step 5 — Symbol & timeframe defaults

- **Purpose:** pre-populate the universe selector so the operator's
  first backtest can run without manual config.
- **Mockup:** two columns.
  Left — symbol picker. Populated by `ProtoOASymbolsListReq` (2114)
  → `ProtoOASymbolsListRes` (2115) against the primary account
  picked in Step 4. A search field at top filters by symbol name.
  Default selection: `EURUSD` (operator's preferred symbol, per the
  brief). Multi-select with a "Top 28 majors" preset and a
  "Custom" mode that exposes the full list.
  Right — timeframe checkboxes from `CANONICAL_TIMEFRAMES` (11
  entries: `M1, M3, M5, M15, M30, H1, H4, H12, D1, W1, MN1` from
  `crates/forex-core/src/contracts/temporal.rs:25`). **`H2` is
  deliberately absent** — verbatim from the file:
  "Αν δεν υπάρχει Η2 τότε ας μην μπει καθόλου!!!" / "If H2 doesn't
  exist [at cTrader] then don't add it at all." (operator directive
  2026-05-14, recorded at `temporal.rs:17–24`). Default selection:
  `M5, M15, H1, H4, D1`. The 11 native checkboxes are arranged in
  a single column so the operator can see at a glance that H2 is
  not on the list (defending against the documented
  fabricated-comment failure mode described at `temporal.rs:22–24`).
  Below the columns: an inline preview "You selected 6 symbols × 5
  timeframes = 30 (symbol, timeframe) pairs; the next step will
  download ≈ Y MiB of history."
- **Inputs:** `selected_symbols: Vec<String>`,
  `selected_timeframes: Vec<String>`.
- **Actions:** fires `ProtoOASymbolsListReq` (rate-limited; this is
  a normal 50/sec request, not the historical 5/sec class — per
  `ctrader_api_full_reference.md` §3.2). Renders progress
  ("Loading symbols from broker… 432 of ~600"). On completion,
  caches the symbol list in `<data_path>/cache/symbols_<broker_id>.json`
  to spare the next wizard run a re-fetch.
- **Skip:** Allowed (defaults to `EURUSD` × `{M5, M15, H1, H4, D1}`).
- **Back:** Step 4.
- **Time:** 30–60 s (symbol fetch over 5035) + however long the
  user spends choosing.

### Step 6 — Historical data download

- **Purpose:** seed the local Polars/Parquet cache so backtests and
  ML training can run immediately after the wizard exits.
- **Mockup:** a slider "Months of history to download" with marks
  at 1, 3, 6, 12, 18, 24; default 6. Below, a forecast block: "≈ N
  MiB total, ≈ T seconds at observed broker rate (5 req/s for
  historical data per cTrader API limits)" — the rate limit comes
  verbatim from `ctrader_api_full_reference.md` §3.2: "a maximum of
  5 requests per second per connection for any historical data
  requests". After the user clicks "Begin download", the slider is
  replaced by a per-(symbol, timeframe) progress table:
  ```
  EURUSD  M5   ███████████░░░░░░░░  58 %  (2.1 MiB / 3.6 MiB)
  EURUSD  M15  ████████████████████ 100 % ✓
  EURUSD  H1   ████████████████████ 100 % ✓
  GBPUSD  M5   ░░░░░░░░░░░░░░░░░░░░   0 %  queued
  …
  ```
  A `[Pause]` button toggles to `[Resume]`. A `[Cancel]` button
  stops further pulls but keeps the bars already on disk —
  cancel-safe per the operator's no-synthetic-data rule (we never
  fabricate filler).
- **Inputs:** `history_months: u8` (1–24).
- **Actions:** fires `ProtoOAGetTrendbarsReq` (2137) per
  (symbol, timeframe) pair. The server returns
  `ProtoOAGetTrendbarsRes` (2138) with `hasMore` paging per
  `ctrader_api_full_reference.md` §3.2. The wizard enforces the
  5-req/s budget with a token-bucket limiter; if the broker
  responds with `REQUEST_FREQUENCY_EXCEEDED` (108), exponential
  backoff (no `Retry-After` header is documented). Each completed
  bar set is written to
  `<data_path>/history/<broker_id>/<symbol>/<timeframe>.parquet`
  with a sidecar `<…>.complete` sentinel file. Partial files (left
  by a Cancel) are marked `<…>.partial` so the main app refuses to
  treat them as complete.
- **Skip:** Allowed (no history downloaded; main app prompts later).
- **Back:** Step 5 — but if a download is in flight, Back is
  disabled until the user either Pauses or Cancels.
- **Time:** highly variable. 6 months × 6 symbols × 5 timeframes ≈
  180 requests ≈ 36 s at 5 req/s plus parse/write overhead, call it
  60–90 s in practice. 24 months × 28 symbols × 11 timeframes is
  much longer — the forecast block makes this clear up-front.

### Step 7 — Hardware compatibility probe

- **Purpose:** detect available compute backends and pick a sensible
  default for ML training.
- **Mockup:** a card per detected device.
  ```
  ┌─ CPU ─────────────────────────────────┐
  │  Intel(R) Core(TM) i7-13700K  · 16 cores
  │  RAM 31.2 / 32.0 GiB available
  │  Backend: NdArray (CPU) — always available
  └───────────────────────────────────────┘
  ┌─ GPU 0 (NVIDIA) ──────────────────────┐
  │  GeForce RTX 4070 Ti  · 12 GiB VRAM
  │  Compute capability 8.9 → FP32, FP16, BF16, FP8
  │  Backends: CUDA ✓  Vulkan (wgpu) ✓
  │  Recommended: CUDA
  └───────────────────────────────────────┘
  ```
  Cards aggregate the output of `HardwareProbe::detect()` from
  `crates/forex-core/src/system.rs:561`. CUDA detection uses
  `nvidia-smi` (lines 605–685); ROCm uses `rocminfo` (lines
  721…); Vulkan / wgpu is the fallback path (the
  `detect_wgpu_hint_accelerators` branch). For each detected GPU
  the wizard records the compute capability tuple (line 612) and
  the `supported_precisions` vector (line 613). The "Recommended"
  badge is chosen by the existing dispatch in
  `crates/forex-core/src/system/backends.rs` (CUDA > ROCm > Vulkan
  > CPU). The user can flip the badge to a non-recommended
  backend manually; doing so writes a `forced_backend` override
  into `hardware_profile.json`.
- **Inputs:** `forced_backend: Option<String>`.
- **Actions:** writes `<data_path>/hardware_profile.json` with the
  full probe report (cpu_cores, total_ram_gb, gpu list with name +
  memory_gb + compute_capability + supported_precisions, platform
  label, ISO-8601 timestamp). This file is read by the training
  orchestrator (`crates/forex-models/src/training_orchestrator.rs:480`)
  to gate which backends are eligible.
- **Skip:** Allowed (defaults to CPU NdArray).
- **Back:** Step 6.
- **Time:** ≤ 5 s for the probe; the user typically spends < 30 s
  reviewing.

### Step 8 — News / sentiment provider

- **Purpose:** wire the `forex_core::domain::news_filter` integration
  so the macro-event filter can suppress trading around scheduled
  releases.
- **Mockup:** a toggle "Enable news filter" (default off). When on,
  reveals two fields: "Provider" (dropdown — currently a single
  option, "OpenAI / GPT-class LLM"; placeholder for future
  ForexFactory-style providers) and "API key" (password-masked
  single-line input). A subtle disclosure: "The key is stored as
  `secrecy::SecretString` and never sent anywhere except your
  chosen provider when the filter runs. See Privacy in Step 10."
- **Inputs:** `news_filter_enabled: bool`, `news_filter_api_key:
  Option<SecretString>`.
- **Actions:** if enabled, runs a single "ping" request to the
  provider to validate the key. On failure, surfaces the
  provider's error verbatim. The key is held in memory and only
  written to disk after Step 10's Apply.
- **Skip:** Allowed; news_filter remains disabled.
- **Back:** Step 7.
- **Time:** ≤ 60 s.

### Step 9 — Auto-start

- **Purpose:** optionally register `forex-app` for system-login
  start so the trading daemon resumes after a reboot.
- **Mockup:** a single labelled toggle "Start forex-app on system
  login" with a sub-toggle (greyed unless main is on) "Start
  minimised to system tray". A footer note explains the per-
  platform mechanism:
  - Windows: a per-user shortcut in `%APPDATA%\Microsoft\Windows\Start
    Menu\Programs\Startup\` (or a `HKCU\Software\Microsoft\Windows\
    CurrentVersion\Run` registry entry if the user is admin).
  - macOS: a per-user `~/Library/LaunchAgents/ai.forex.app.plist`.
  - Linux: a `~/.config/autostart/forex-app.desktop` — per the
    freedesktop Autostart Specification, which says applications
    "should be run automatically upon login" and that
    "user-level: ~/.config/autostart/" is the canonical location
    (freedesktop *Autostart Specification* snippet via WebSearch).
    The file fields are the standard minimum: `[Desktop Entry]`,
    `Type=Application`, `Name=forex-ai`, `Exec=<install_dir>/forex-
    app --minimized`, `Terminal=false`.
- **Inputs:** `autostart_enabled: bool`, `start_minimized: bool`.
- **Actions:** writes / removes the per-platform autostart artefact.
  On Linux, no elevated privileges are required (`~/.config/autostart`
  is user-writeable). On macOS, no elevated privileges are required
  for `~/Library/LaunchAgents`. On Windows, the user shortcut path
  is `%APPDATA%`-scoped, not `%ProgramData%`, so no UAC prompt.
- **Skip:** Allowed (default off).
- **Back:** Step 8.
- **Time:** ≤ 10 s.

Cross-reference: `TODO(installer_infrastructure_spec)` §7 owns the
exact paths and platform conventions; this wizard step is the
user-facing toggle that drives the same code.

### Step 10 — Summary & Apply

- **Purpose:** show every choice the wizard has made, let the user
  confirm or jump back, then commit.
- **Mockup:** a single scrollable review pane laid out as a table:
  ```
  License accepted          2026-05-15 19:42:11 UTC      [edit ↑]
  Data directory            ~/.local/share/forex-ai/     [edit ↑]
  Operator name             (blank)                      [edit ↑]
  Prop firm                 FTMO Standard (4 % monthly)  [edit ↑]
  Trading mode              Forward test (paper)         [edit ↑]
  cTrader account           Demo • EURUSD broker • #12345 [edit ↑]
  Symbols                   EURUSD, GBPUSD, USDJPY  …    [edit ↑]
  Timeframes                M5, M15, H1, H4, D1          [edit ↑]
  History download          6 months — 30 pairs queued   [edit ↑]
  Hardware backend          CUDA (RTX 4070 Ti)           [edit ↑]
  News filter               disabled                     [edit ↑]
  Auto-start                disabled                     [edit ↑]
  Crash reports             disabled (default)           [edit ↑]
  ```
  Each `[edit ↑]` link jumps back to the originating step,
  preserving every other choice (Microsoft UX wizards: "Preserve
  user selections through navigation"). At the bottom: `[Cancel]`,
  `[Apply]`. `[Apply]` is the primary button (visually heavier;
  see `TODO(ui_ux_design_spec)` for the focus-ring style).
- **Inputs:** none beyond confirmation.
- **Actions:** in this order —
  1. Write `<data_path>/config.yaml` (forex-ai's main config — does
     not exist yet for first-time installs).
  2. Write `<data_path>/broker_credentials.toml` (per
     `broker_persistence.rs` §, encrypted-at-rest if the OS
     keychain is available; falls back to file with explicit
     permission 0o600 on Unix, ACL-restricted on Windows).
  3. Write `<data_path>/hardware_profile.json`.
  4. Write `<data_path>/wizard_complete.json` with the full state.
  5. Spawn the historical-data download into a background job (it
     was queued in step 6 but is not blocking — Apply returns when
     the queue is enqueued, not when the downloads finish).
  6. Close the wizard modal; open the main app window with the
     "Welcome — let's run your first backtest" tour active.
- **Skip:** N/A (this is the terminal step).
- **Back:** to any prior step via `[edit ↑]` or the standard `[←
  Back]`.
- **Cancel:** confirmation modal: "Discard all changes and exit?
  Your downloaded history (if any) will be preserved." Two
  buttons: `[Keep editing]`, `[Discard]`.
- **Time:** ≤ 5 s for the disk writes.

---

## §3 — Per-step error handling

For every step, the wizard adheres to three meta-rules (operator
policy + RFC 8252 + NN/G):

1. **Never silently skip.** A failed step that is skipped is
   logged to `wizard_state.json` under `incomplete`, and the main
   app surfaces a banner: "cTrader setup is incomplete. Live
   trading is disabled. [Resume Setup]".
2. **Always offer Retry.** Every network call gets a Retry
   affordance with exponential backoff (start 1 s, max 30 s,
   jitter ±20 %).
3. **No synthetic fallback.** If the broker can't be reached, the
   wizard says so and stops — it does **not** synthesise a fake
   account list, a fake symbol list, or fake historical bars.

The error matrix:

| Step | Error class | UX response | Recovery |
|------|-------------|-------------|----------|
| 1 — License | LICENSE file missing | Show built-in fallback license text; warn "Could not find LICENSE on disk; using embedded copy from build-time include." | Continue |
| 2 — Path | No write permission | Inline red banner with the OS error verbatim ("Permission denied" / "Access is denied"). | Pick another path or run as admin/sudo if user insists |
| 2 — Path | Disk space < 5 GiB | Amber banner: "Only X GiB free; historical data + checkpoints typically need 8–20 GiB. Pick another volume?" | Continue allowed; warning logged |
| 3 — Profile | Monthly profit < 4 % | Inline validator: "Minimum 4 % per operator policy (2026-05-14). Lower values are not selectable." | Adjust slider |
| 4.1 — Credentials | Empty or malformed Client ID/Secret | Inline validator: "Client ID must be a digit string; Client Secret must be 32+ chars." | Re-enter |
| 4.1 — Test creds | `CH_CLIENT_AUTH_FAILURE` (101) | "The broker rejected these credentials. Verify them at openapi.ctrader.com, then re-test." | Retry / re-enter |
| 4.1 — Test creds | `CH_OA_CLIENT_NOT_FOUND` (107) | "The broker doesn't recognise this Client ID. Did you copy it from the wrong app?" | Re-enter |
| 4.2 — OAuth | Loopback bind fails on all ports | "Could not open the local callback server on ports 7777/7878/8989. Use the copy-paste flow?" | Switch to copy-paste flow |
| 4.2 — OAuth | 5-minute timeout | "No browser callback in 5 minutes. Was the page closed? Sign in again?" | Retry / Skip |
| 4.2 — OAuth | `state` mismatch | Hard refuse: "Security: the callback's state token doesn't match. Possible CSRF — refusing to proceed." (matches existing `ctrader_live_auth.rs:36–43` audit-fix F2.) | Restart 4.2 |
| 4.2 — OAuth | Token exchange returns `errorCode` field | Surface the broker's `description` verbatim. | Retry |
| 4.3 — Accounts | Empty account list | "Your cTID has no trading accounts registered. Open a demo account at <https://ctrader.com> then come back." | Retry / Skip |
| 4.4 — Acct auth | `ACCOUNT_NOT_AUTHORIZED` (2) | "Token doesn't grant access to this account. Re-do the sign-in step?" | Back to 4.2 |
| 4.4 — Acct auth | `CONNECTIONS_LIMIT_EXCEEDED` (67) | "Too many simultaneous connections from this Client ID. Close other sessions and retry." | Retry |
| 5 — Symbols | `ProtoOASymbolsListReq` times out | "Broker took too long to return the symbol list. This sometimes happens during scheduled broker maintenance; retry in 30 s?" | Retry / Skip |
| 6 — History | `REQUEST_FREQUENCY_EXCEEDED` (108) | Wizard's token-bucket already gates this; if the broker still returns 108 (clock drift), back off 30 s and resume. | Automatic |
| 6 — History | Partial download on Cancel | Mark file `.partial`; banner on the main app: "EURUSD M5 download is incomplete (38 % of 6 months). Resume from `Data → Backfill`." | Resume from main app |
| 6 — History | Disk full mid-download | Hard stop: "Disk full at `<data_path>`. Free space and click Resume." | Resume |
| 7 — Hardware | `nvidia-smi` missing but NVIDIA card present | Card surfaces in the wgpu branch only; warning: "NVIDIA driver not detected — install the official driver to enable CUDA backend." | Continue with wgpu |
| 7 — Hardware | No GPU at all | CPU-only card shown; no error. | Continue |
| 8 — News | API ping fails | Surface provider's error verbatim. | Retry / Disable |
| 9 — Autostart | Can't write `~/.config/autostart` | "Permission denied at `~/.config/autostart`. Skip auto-start?" | Skip |
| 10 — Apply | Disk full at write | "Cannot write `<config.yaml>`. Free space and click Retry." | Retry |
| 10 — Apply | Keychain unavailable (macOS) | "macOS keychain is locked. Fall back to file-based credential storage (less secure)?" | Continue with file |

---

## §4 — Theming

Design tokens are owned by `ui_ux_design_spec.md` (parallel agent).
This wizard expects the following tokens to exist; placeholders are
TODOs until that doc lands.

| Token | TODO placeholder | Use |
|-------|------------------|-----|
| `color.surface.canvas` | `#0E1117` (dark) / `#FFFFFF` (light) | Modal background |
| `color.surface.card` | `#171A21` / `#F6F8FB` | Card / row background |
| `color.text.primary` | `#E6E8EE` / `#101218` | Body text |
| `color.text.muted` | `#8A93A6` / `#5F6A7E` | Helper / footnote text |
| `color.accent` | `#2F7FF9` | Primary buttons, focus ring |
| `color.success` | `#2EA86A` | Validation ticks |
| `color.warning` | `#E1A227` | Warnings |
| `color.danger` | `#D14545` | Errors |
| `typography.heading` | Inter Semibold 20 / 24 / 28 | Step titles |
| `typography.body` | Inter Regular 14 / 16 | Body |
| `typography.mono` | JetBrains Mono 13 | Paths, identifiers |
| `space.unit` | 4 px | Base spacing unit |
| `radius.card` | 12 px | Card corner radius |
| `focus.ring` | 2 px solid `color.accent`, 2 px offset | Keyboard focus |

The wizard is fully keyboard-navigable: `Tab` cycles fields, `Shift
+Tab` reverses, `Space`/`Enter` activates the primary button,
`Esc` triggers the same prompt as `[Cancel]`. The focus ring is
mandatory for accessibility — Microsoft Learn's *UX checklist for
desktop applications* lists missing focus indicators as a top
violation
(<https://learn.microsoft.com/en-us/windows/win32/uxguide/top-violations>).

---

## §5 — Skip / re-run

All steps except Step 1 (License) are skippable. The wizard
records each skip with a structured reason. The schema of
`wizard_complete.json`:

```jsonc
{
  "schema_version": 1,
  "completed_at": "2026-05-15T19:48:33Z",
  "wizard_version": "0.5.0",
  "state": "complete" | "completed_with_warnings",
  "skipped_steps": ["news_filter", "autostart"],
  "incomplete_steps": [],          // e.g. ["ctrader_oauth"] if OAuth was skipped
  "choices": {
    "license_sha256": "…",
    "data_path": "…",
    "prop_firm_preset": "FTMO_STANDARD",
    "monthly_profit_target": 0.04,
    "trading_mode": "forward",
    "ctrader_account_id": 12345,
    "selected_symbols": ["EURUSD", "GBPUSD"],
    "selected_timeframes": ["M5", "M15", "H1", "H4", "D1"],
    "history_months": 6,
    "forced_backend": null,
    "news_filter_enabled": false,
    "autostart_enabled": false,
    "telemetry_opt_in": false
  }
}
```

### 5.1 Re-running the wizard

Three entry points:

1. **GUI:** `Settings → Setup Wizard → Re-run`. Always available.
2. **`forex-app --wizard`:** opens the GUI wizard and bypasses the
   "first-run" gate.
3. **`forex-cli wizard`:** opens the TUI wizard (§8).

When re-running, the wizard pre-populates every field from
`wizard_complete.json`. Re-running does NOT discard existing OAuth
tokens; the cTrader step's "Sign in with cTID" button instead reads
"Re-authenticate with cTID (current token valid for N days)" and is
a no-op if the user clicks Next without re-authing.

### 5.2 Partial completion

If the user exits the wizard mid-flow (e.g. closes the window after
Step 4), the wizard saves a `wizard_progress.json` instead of
`wizard_complete.json`. On next launch, the wizard resumes at the
step after the last fully-validated step (preserving the existing
choices). Microsoft UX wizard guideline: "Preserve user selections
through navigation".

---

## §6 — Migration from portable

forex-ai pre-0.5 was a portable app: all state lived under
`~/.forex-ai/`. The 0.5+ installer points at OS-canonical paths
(see Step 2). The wizard detects the legacy directory and offers to
migrate.

### 6.1 Detection

On Step 2 entry, the wizard runs:

```
for candidate in [
    "~/.forex-ai",
    "~/forex-ai",
    "%USERPROFILE%/.forex-ai",
]:
    if exists(candidate) and any_of(
        "config.yaml", "broker_credentials.toml", "checkpoints/",
        "data/", "history/"
    ):
        offer_migration(candidate)
```

### 6.2 Migration prompt

A modal overlay on Step 2:

```
We found an existing forex-ai install at /home/op/.forex-ai/.
Migrate it to /home/op/.local/share/forex-ai/?

  [✓] Config (config.yaml, 3.1 KiB)
  [✓] Broker credentials (broker_credentials.toml, 2.4 KiB)
  [✓] Cached history (data/history/, 4.2 GiB)
  [✓] Model checkpoints (checkpoints/, 1.7 GiB)
  [✓] OAuth refresh token (preserved → re-auth not required)

Free space at destination: 84 GiB ✓

After migration:
  ( ) Keep the old directory
  (•) Delete it (asked again at the end)
  ( ) Leave the choice for later

[Skip migration]  [Migrate now]
```

### 6.3 Migration semantics

- Migration is **atomic per file** — every file is copied first,
  then renamed into place; the source is removed only after the
  target verifies (size + SHA-256).
- The OAuth refresh token is reused as-is — no need to re-do the
  browser flow.
- If migration fails partway, both directories survive; the wizard
  surfaces "Migration aborted at step X" and lets the user pick
  "Retry", "Keep both (use the new one going forward)", or
  "Rollback (use the old one)".
- If migration succeeds and the user picked "Delete it" with
  confirmation, the source directory is `rm -rf`'d *after* a
  second confirmation modal.

### 6.4 Skipping migration

If the user skips, the wizard proceeds with the OS-canonical path
and the portable directory is left untouched. The main app, on
startup, surfaces a one-time banner: "A legacy forex-ai directory
exists at `~/.forex-ai/`. [Migrate now] [Dismiss]".

---

## §7 — Telemetry / privacy

### 7.1 Default

**No telemetry.** The wizard does not collect any data unless the
user explicitly opts in to crash reports in §10's Apply step. This
matches the operator's overall posture (no-synthetic-data, no
unsolicited network calls).

### 7.2 Optional crash reports

The Summary step (10) includes a row "Crash reports — disabled
(default)". Clicking `[edit ↑]` opens a final sub-modal:

```
Help improve forex-ai by sending crash reports?

If a panic or unhandled error happens, forex-ai can send a stack
trace, OS version, and forex-ai version to our crash-reporting
service (Sentry). No trading data, no broker credentials, no
account numbers, no model output is ever sent.

What is sent:           What is NEVER sent:
- Rust panic message    - OAuth client ID / secret
- Rust stack trace      - OAuth access / refresh tokens
- OS + version          - cTID trader account ID
- forex-ai version      - Symbol selections
- Sanitised file paths  - Trading history
- Hardware tier (CPU /  - Model checkpoints
  GPU class only)       - News API keys

[Decline]   [Opt in]
```

Sentry's own GDPR guidance recommends "obtain opt-in consent for
Sentry SDKs via your website or app consent banner" (Sentry, *Best
Practices for GDPR Compliance*, snippet via WebSearch); the
forex-ai wizard treats this as a hard requirement: the toggle is
default-off, the disclosure is plain language, and the choice is
recorded in `wizard_complete.json` as `telemetry_opt_in: bool`.

### 7.3 PII scrubbing

If the user opts in, the Sentry SDK is configured with:

- `send_default_pii = false` (no IPs or usernames).
- `before_send` hook that strips: anything matching the regex set
  for OAuth tokens, anything that looks like a path containing
  `broker_credentials`, anything containing the literal substring
  of the loaded `access_token`.

This is in line with Sentry's documented PII scrubbing API
(<https://docs.sentry.io/security-legal-pii/>, snippet via
WebSearch).

### 7.4 What the wizard *itself* never transmits

Verbatim text shown in Step 4 and Step 10:

> Your cTrader Client Secret, OAuth access token, OAuth refresh
> token, cTID trader account ID, news API key, and broker symbol
> selections are stored only on this machine. The wizard never
> transmits them to any server other than your chosen broker
> (cTrader) and, if you enabled it, your news provider. Anthropic,
> Spotware-the-vendor (vs. the broker API), and forex-ai itself
> have no path to read these values.

---

## §8 — CLI parity

### 8.1 `forex-cli wizard`

A new subcommand `forex-cli wizard` runs the same state machine as
the GUI, rendered via `ratatui` (CrossTerm backend — matches the
existing CLI's terminal stack).

Each GUI step has a TUI counterpart:

| GUI step | TUI counterpart |
|----------|-----------------|
| 1 — License | Pager view of the license file; `Y` to accept, `N` to abort |
| 2 — Path | Text input field with tab-complete on `<path>` |
| 3 — Profile | Three list selectors; arrow keys to navigate |
| 4 — OAuth | Sub-states 4.1–4.4 as paged screens; the browser still opens via `xdg-open`/`open`/`start` |
| 5 — Symbols | Two-pane multi-select: left = filterable symbol list, right = canonical timeframe list. `Space` toggles, `Tab` switches pane |
| 6 — History | Slider rendered as `[--- 6 ---]` with `←`/`→`; progress bars rendered with `ratatui::widgets::Gauge` |
| 7 — Hardware | Card-style block rendering; `Tab` cycles devices, `Enter` to flip the recommended backend |
| 8 — News | Input field; key masked as `*` |
| 9 — Autostart | Single toggle |
| 10 — Summary | Scrollable table; `e` on a row jumps back to its step |

### 8.2 Keybindings (verbatim)

| Key | Action |
|-----|--------|
| `→` / `Enter` | Continue (Next) |
| `←` | Back |
| `Tab` / `Shift+Tab` | Cycle fields within current step |
| `Space` | Toggle checkbox / radio |
| `s` | Skip current step (only when skippable) |
| `r` | Retry the last failed action |
| `Esc` | Open Cancel confirmation |
| `?` | Open inline help for the current step |
| `q` | Same as `Esc` |

### 8.3 No-tty mode

If `stdin` is not a tty (e.g. piped from a shell script), the
wizard refuses to start — printing to stderr: "wizard requires a
tty; use `forex-cli init` for headless first-run setup". The
headless `init` subcommand is a *separate* command (out of scope
for this spec) that takes a YAML file with the same choices the
wizard would collect.

### 8.4 SSH-friendly mode

The wizard runs over SSH as long as the user can open a browser
locally. The browser-open step prints the URL to stdout and waits
for the user to paste back the redirect URL. This is the same
copy-paste fallback path used by the GUI when loopback bind fails
(§2 Step 4.2). RFC 8252 §7.3 — "OAuth servers must … support
loopback IP redirect URIs … required to support desktop operating
systems" — combined with §7.2 ("private-use URI scheme") implies
the copy-paste flow is RFC-compliant.

---

## §9 — Mockups (text)

### 9.1 Step 1 — Welcome + License

```
┌─ forex-ai Setup Wizard ────────────────────────────────────┐
│  forex-ai v0.5.0 — built 2026-05-15                         │
│                                                             │
│  This wizard will set up your trading workspace in about    │
│  10 minutes:                                                │
│   1. License agreement                                      │
│   2. Path selection                                         │
│   3. Account & profile                                      │
│   4. cTrader sign-in                                        │
│   5. Symbol & timeframe defaults                            │
│   6. Historical data download                               │
│   7. Hardware compatibility probe                           │
│   8. News / sentiment provider                              │
│   9. Auto-start                                             │
│  10. Summary & Apply                                        │
│                                                             │
│  ┌─ LICENSE ────────────────────────────────────────────┐  │
│  │ Apache License v2.0                                  ↑│  │
│  │                                                       │  │
│  │ Copyright (c) 2026 forex-ai contributors             │  │
│  │ …                                                    ↓│  │
│  └──────────────────────────────────────────────────────┘  │
│  [ ]  I have read and accept the license                    │
│                                                             │
│  [Cancel]                                  [Continue →]     │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Step 4.2 — Sign in with cTID

```
┌─ Step 4.2 / 10 · cTrader sign-in ──────────────────────────┐
│  Waiting for browser sign-in…                               │
│                                                             │
│       ╭────────────────────────────────────────╮            │
│       │  Browser opened to id.ctrader.com      │            │
│       │  Sign in with your cTID, approve       │            │
│       │  forex-ai's access, and return here.   │            │
│       │                                        │            │
│       │  Loopback callback bound on            │            │
│       │  http://127.0.0.1:7777/ctrader/callback │           │
│       │                                        │            │
│       │  Timeout in:   4 min 23 s              │            │
│       ╰────────────────────────────────────────╯            │
│                                                             │
│  Trouble? [Open URL again]   [Use copy-paste flow]          │
│                                                             │
│  [← Back]              [Cancel]                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 Step 5 — Symbol & timeframe defaults

```
┌─ Step 5 / 10 · Symbols & timeframes ───────────────────────┐
│  Pick the symbols and timeframes to seed.                  │
│                                                             │
│  Symbols (multi-select)        Timeframes (multi-select)    │
│  ┌──────────────────────┐      ┌────────────────────────┐  │
│  │ Search: EUR_         │      │ [ ] M1                 │  │
│  │                      │      │ [ ] M3                 │  │
│  │ [✓] EURUSD       ★   │      │ [ ] M5  ←recommended   │  │
│  │ [ ] EURGBP           │      │ [✓] M5                 │  │
│  │ [ ] EURJPY           │      │ [✓] M15                │  │
│  │ [ ] EURAUD           │      │ [ ] M30                │  │
│  │ [ ] EURCHF           │      │ [✓] H1                 │  │
│  │ …                    │      │ [✓] H4                 │  │
│  │                      │      │ [ ] H12                │  │
│  │ Preset:  Top 28 ▼    │      │ [✓] D1                 │  │
│  └──────────────────────┘      │ [ ] W1                 │  │
│                                │ [ ] MN1                │  │
│  6 symbols × 5 tfs = 30 pairs  └────────────────────────┘  │
│  ≈ 14 MiB for 6 months                                      │
│                                                             │
│  [← Back]              [Skip]              [Continue →]     │
└─────────────────────────────────────────────────────────────┘
```

(Note the H2 row is **deliberately absent** from the timeframe
list, matching `CANONICAL_TIMEFRAMES` from
`crates/forex-core/src/contracts/temporal.rs:25`.)

### 9.4 Step 6 — Historical data download

```
┌─ Step 6 / 10 · Historical data ────────────────────────────┐
│  Download history for: 6 symbols × 5 timeframes             │
│                                                             │
│  Months of history:                                         │
│  1   3   6   12   18   24                                   │
│  ●───●───◉───●────●────●          ≈ 14 MiB, ≈ 80 s          │
│                                                             │
│  ┌─ Progress ─────────────────────────────────────────────┐│
│  │ EURUSD  M5   ████████████████████ 100 % ✓             ││
│  │ EURUSD  M15  █████████████░░░░░░░  68 % (2.4/3.6 MiB) ││
│  │ EURUSD  H1   ░░░░░░░░░░░░░░░░░░░░   0 % queued        ││
│  │ EURUSD  H4   ░░░░░░░░░░░░░░░░░░░░   0 % queued        ││
│  │ EURUSD  D1   ░░░░░░░░░░░░░░░░░░░░   0 % queued        ││
│  │ GBPUSD  M5   ░░░░░░░░░░░░░░░░░░░░   0 % queued        ││
│  │ …                                                      ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  [← Back]    [Pause]    [Skip]              [Continue →]    │
└─────────────────────────────────────────────────────────────┘
```

### 9.5 Step 10 — Summary & Apply

```
┌─ Step 10 / 10 · Review & Apply ────────────────────────────┐
│                                                             │
│  License accepted         2026-05-15 19:42:11 UTC  [edit↑]  │
│  Data directory           ~/.local/share/forex-ai  [edit↑]  │
│  Prop firm                FTMO Standard · 4 %      [edit↑]  │
│  Trading mode             Forward test             [edit↑]  │
│  cTrader account          Demo · #12345 · EUR      [edit↑]  │
│  Symbols                  6 selected               [edit↑]  │
│  Timeframes               M5, M15, H1, H4, D1      [edit↑]  │
│  History                  6 months · 30 pairs      [edit↑]  │
│  Hardware backend         CUDA (RTX 4070 Ti)       [edit↑]  │
│  News filter              disabled                 [edit↑]  │
│  Auto-start               disabled                 [edit↑]  │
│  Crash reports            disabled (default)       [edit↑]  │
│                                                             │
│  [← Back]              [Cancel]               [Apply ✓]     │
└─────────────────────────────────────────────────────────────┘
```

---

## §10 — Open questions

These remain open at the time of writing. Each carries an assigned
owner for follow-up.

### 10.1 Does the wizard require admin rights on Windows for anything beyond install-path selection?

**Tentative answer: no.** The default install path is
`%LOCALAPPDATA%\forex-ai\` which is user-writeable. All wizard
writes (`broker_credentials.toml`, `hardware_profile.json`,
`config.yaml`, `wizard_complete.json`) are per-user. The autostart
mechanism in Step 9 uses the per-user shortcut path / `HKCU` —
neither needs UAC. The only path that triggers UAC is if the user
manually overrides the install location to `C:\Program Files\` in
Step 2, which is unusual for a per-user app.

**Open follow-up:** confirm with `installer_infrastructure_spec`
that the WiX bundle declares `InstallScope="perUser"`.

### 10.2 macOS notarization for an unsigned first-run-wizard binary

**Answer: notarization is required for distribution, but the
wizard is internal to the (notarized) `forex-app` binary, so this
is a single notarization, not two.** Apple's *Notarizing macOS
software before distribution* page (snippet via WebSearch) says
"Apple recommends notarizing software even if you plan to
distribute it from your own website". With the in-app wizard
architecture chosen in §1.2, the entire binary including the
wizard is one signed + notarized artefact.

**Open follow-up:** verify the notarization service still accepts
binaries that bind to loopback ports without an explicit
entitlement. (As of May 2026, multiple developers report
notarization rejections — search snippet — but those rejections
are around entitlement misuse, not loopback binding per se.)

### 10.3 Loopback OAuth on macOS App Sandbox

**Answer: forex-ai is NOT distributed via the Mac App Store, so it
is not constrained by the App Sandbox.** Distribution via direct
download + notarization + Gatekeeper allows arbitrary loopback
binding. If forex-ai is ever pushed to the App Store, the
`com.apple.security.network.server` entitlement plus a non-Sandbox
loopback path (LaunchAgent-mediated) would be required — out of
scope for the current shipping path.

**Open follow-up:** if Mac App Store distribution becomes a goal,
revisit this.

### 10.4 Re-running on a partially-completed install

What if the user opens the wizard via `forex-app --wizard` while
`forex-app` is already running with a valid OAuth session? Two
options: (a) reject "wizard already running in main app instance",
(b) re-use the live session and treat the wizard as a settings
editor. Recommendation: (a), simpler.

**Open follow-up:** confirm with `ui_ux_design_spec` how the
"Settings" panel and "Wizard" should overlap. There's a real risk
of redundancy.

### 10.5 What happens if the broker symbol-list response is paginated and we time out partway?

Today, `ProtoOASymbolsListRes` (2115) is documented as a single
response (no `hasMore` field in the local proto). If a broker
returns a truncated list silently, the symbol picker shows fewer
symbols than the broker actually offers. The wizard does not
re-query; the operator can retry from Step 5.

**Open follow-up:** monitor `spotware_proto_freshness.md` for any
upstream change to `ProtoOASymbolsListRes` that adds pagination.

### 10.6 Where does the FTMO Aggressive preset come from?

The preset is mentioned in Step 3's mockup but `prop_firm.rs`
exposes only `FTMO_STANDARD`. The wizard should either (a) hide
the "Aggressive" option until the constants exist, or (b) prompt a
small code patch to add `FTMO_AGGRESSIVE` per FTMO's published
rules (5 % daily / 10 % overall / 4 % min trading days). The
operator's directive is silent on FTMO Aggressive, so option (a)
is the safe default.

**Open follow-up:** ask the operator whether to add the Aggressive
preset.

### 10.7 Token rotation behaviour during wizard re-run

If the wizard is re-run after the cTrader refresh token has been
rotated by the background daemon, the wizard's view of the token
may be stale. Recommendation: the wizard never reads tokens
directly — it reads only `<data_path>/broker_credentials.toml`,
which the daemon updates atomically.

**Open follow-up:** confirm `broker_persistence.rs` exposes a
read-only "current state" API that the wizard can use without
racing the daemon's writes.

---

## §11 — Acceptance criteria

For the implementation that follows this spec, the following must
hold (these are the contract the wizard must satisfy):

1. A user with zero prior knowledge of cTrader Open API can finish
   the wizard in ≤ 15 minutes and end up with: a valid OAuth
   refresh token persisted, a default symbol+timeframe set selected,
   at least 1 month of historical bars for EURUSD M5 on disk, and a
   hardware profile written.
2. Skipping every skippable step is permitted; the wizard ends
   with `state = "completed_with_warnings"` and the main app banners
   exactly which steps remain.
3. Re-running the wizard preserves every prior choice (Microsoft
   UX *Wizards* guideline: "Preserve user selections through
   navigation").
4. The wizard never persists OAuth tokens until the OAuth flow has
   completed (no half-written `broker_credentials.toml`).
5. The wizard never transmits any value to anyone other than the
   broker (cTrader) and, if explicitly opted in, the news provider
   and the crash-reporting service.
6. The TUI (`forex-cli wizard`) renders all ten steps with a
   keyboard-only flow per §8.2.
7. Migration from `~/.forex-ai/` is atomic and reversible until the
   user confirms deletion of the source.
8. The H2 timeframe is **never** offered in the timeframe selector,
   regardless of any client-side defaults or saved presets — per
   operator directive `temporal.rs:17–24`.
9. The 4 % minimum monthly profit target is enforced by the
   wizard's input validator — values below 4 % cannot be entered
   per operator directive `prop_firm.rs:36`.
10. No synthetic / fabricated fallback data is ever shown to the
    user. If a broker call fails, the failure is surfaced; the
    wizard does not invent a symbol list, account list, or
    historical bars.

---

## §12 — Glossary of internal identifiers

For implementers, these are the canonical references used in this
spec:

| Identifier | Location | Purpose |
|------------|----------|---------|
| `CANONICAL_TIMEFRAMES` | `crates/forex-core/src/contracts/temporal.rs:25` | 11 timeframes, NO H2 |
| `PropFirmConstraints::FTMO_STANDARD` | `crates/forex-core/src/domain/prop_firm.rs:32` | FTMO + 4 % monthly floor |
| `HardwareProbe` | `crates/forex-core/src/system.rs:27` | Step 7 implementation |
| `HardwareProbe::detect()` | `crates/forex-core/src/system.rs:561` | Per-call probe |
| `ProductionCTraderLiveAuthBackend` | `crates/forex-app/src/app_services/ctrader_live_auth.rs:120` | Step 4 implementation |
| `CTraderLoopbackConfig` | `ctrader_live_auth.rs:28` | Loopback port allocator |
| `CTRADER_CALLBACK_TIMEOUT` | `ctrader_live_auth.rs:24` | 300 s — matches Step 4.2 timer |
| `CTraderCallbackPayload.state` | `ctrader_live_auth.rs:38` | CSRF state, audit-fix F2 |
| `ProtoOAApplicationAuthReq` (2100) | `ctrader_api_full_reference.md` §2.6 | Application auth, Step 4.4 |
| `ProtoOAGetAccountListByAccessTokenReq` (2149) | `ctrader_api_full_reference.md` §2.6 | Account discovery, Step 4.3 |
| `ProtoOAAccountAuthReq` (2102) | `ctrader_api_full_reference.md` §2.6 | Per-account auth, Step 4.4 |
| `ProtoOASymbolsListReq` (2114) | `ctrader_api_full_reference.md` §4.1 | Symbol fetch, Step 5 |
| `ProtoOAGetTrendbarsReq` (2137) | `ctrader_api_full_reference.md` §4.1 | History fetch, Step 6 |
| `NewsFilter` | `crates/forex-core/src/domain/news_filter.rs:12` | Step 8 wiring |
| `broker_persistence.rs` | `crates/forex-app/src/app_services/broker_persistence.rs` | OS-canonical config dir lookup |

---

## §13 — Methodology

- All operator-policy values quoted from local source files were
  read directly from the working copy at `/home/user/forex-ai/`.
- All external UX guidance quotes are attributed inline to their
  source URL; where the URL returned HTTP 403 to the sandbox's
  WebFetch tool, "snippet via WebSearch" is noted and the WebSearch
  result excerpt is the source of the quote.
- cTrader payload type IDs and message names come from the audit's
  internal canonical reference,
  `docs/audits/research/ctrader_api_full_reference.md`, which was
  itself built from the vendored `.proto` files at
  `crates/forex-app/proto/`.
- The H2 prohibition and 4 % monthly profit floor are taken from
  the canonical-source files (`temporal.rs`, `prop_firm.rs`) where
  the operator directives are recorded as code comments dated
  2026-05-14.
- No code was changed in producing this spec. The deliverable is
  research only.

---

— END —

(External citations are enumerated in §0; internal identifiers in
§12. No separate "sources cited" appendix is repeated here.)
