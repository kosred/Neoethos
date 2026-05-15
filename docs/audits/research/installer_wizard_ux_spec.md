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

Docs-first citations used throughout this spec. "snippet via
WebSearch" means the underlying page returned HTTP 403 to the
sandbox's WebFetch tool and content was reconstructed from
WebSearch result excerpts quoting the canonical page directly.

External:

- NN/G *Wizards* — <https://www.nngroup.com/articles/wizards/>
- NN/G *Progressive Disclosure* — <https://www.nngroup.com/articles/progressive-disclosure/>
- NN/G *8 Design Guidelines for Complex Applications*
- Microsoft Learn *UX checklist for desktop applications* — <https://learn.microsoft.com/en-us/windows/win32/uxguide/top-violations>
- Microsoft Learn *Win32 Wizards* — `…/uxguide/win-wizards`
- RFC 8252 *OAuth 2.0 for Native Apps* — <https://datatracker.ietf.org/doc/html/rfc8252>
- freedesktop.org *Desktop Application Autostart Specification*
- freedesktop.org *XDG Base Directory Specification*
- Apple Developer *Distribution XML Reference*
- Apple Developer *Packaging Mac software for distribution*
- Apple Developer *Notarizing macOS software before distribution*
- FireGiant *WixUI dialog library* — <https://docs.firegiant.com/wix/tools/wixext/wixui/>
- Sentry *GDPR Best Practices* — <https://sentry.io/trust/privacy/gdpr-best-practices/>
- Lollypop *Wizard UI Design 2026* / Andrew Coyle *Form Wizard* /
  Eleken *Wizard UI Pattern* / Krystal Higgins *Design of Setup
  Wizards* (403 — listed only) / UXPin *Progress Trackers 2026*

Internal:

- `docs/audits/research/ctrader_api_full_reference.md`
- `docs/audits/research/spotware_proto_new_messages.md`
- `docs/audits/research/ml_numerical_reference.md`
- `crates/forex-app/src/app_services/ctrader_live_auth.rs`
- `crates/forex-core/src/contracts/temporal.rs`
- `crates/forex-core/src/domain/prop_firm.rs`
- `crates/forex-core/src/system.rs`

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

**Choose Option B.** Five reasons:

1. **Linux parity.** deb/rpm `postinst` can't reliably open GUIs
   (DISPLAY inheritance, dbus activation); AppImage has no
   `postinst`. The freedesktop Autostart Specification puts
   first-run behaviour in the app's domain (snippet via WebSearch).
   An in-app wizard works identically across deb/rpm/AppImage/
   Flatpak/tarball.
2. **Re-runnability.** Native installer wizards can only run at
   install time; an in-app wizard is just `forex-app --wizard`.
3. **macOS notarization.** Apple's service has tightened in 2026
   (multiple Feb–May rejections — search snippet). One artefact
   (the app) vs two (installer + app) halves the notarization
   surface.
4. **Microsoft UX guidance.** Microsoft Learn *Win32 Wizards*:
   "Reduce the number of pages to focus on essentials and consolidate
   related pages" (snippet via WebSearch). One stack, not two.
5. **Operator's automation rule.** OAuth tokens / models / cached
   history are per-user — not knowable at install time when the
   installer often runs as root.

Installer's job collapses to: copy files, register launcher,
schedule autostart **if asked** (Step 9), exit. Everything
user-facing runs inside `forex-app` on first run.

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

- **Purpose:** introduce the wizard and capture LICENSE acceptance.
- **Mockup:** centred 720×540 modal. Wordmark + version at top,
  4–5 lines describing the upcoming nine steps with estimated total
  time ("≈ 10 minutes"). Scrollable LICENSE pane (lower 60 %),
  then "I have read and accept the license" checkbox, then footer
  `[Cancel]` / `[Continue →]` (Continue disabled until checked).
- **Inputs:** license-accepted boolean.
- **Actions:** reads `LICENSE` from `<install_dir>/LICENSE` (with
  an `include_str!` build-time fallback). On accept, writes the
  LICENSE SHA-256 + timestamp to `wizard_state.json`.
- **Skip:** NOT ALLOWED — only mandatory step.
- **Back:** N/A.
- **Cancel:** confirm modal: "Cancel installation? The wizard
  will run again on next launch." `[Stay]` / `[Quit]`.
- **Time:** ≤ 30 s.

Microsoft Learn UX wizards: "Don't use Welcome pages — make the
first page functional whenever possible" (snippet via WebSearch).
Including the license on this page satisfies that — it's
functional.

### Step 2 — Path selection

- **Purpose:** pick the install dir (binary, informational) and the
  user-data dir (caches, OAuth tokens, checkpoints, history).
- **Mockup:** two `[Browse…]` text fields. Binary path is read-only
  (set by installer). Data dir defaults to OS-canonical via the
  `directories` crate:
  - Windows: `%LOCALAPPDATA%\forex-ai\` (Windows Known Folder API
    `FOLDERID_LocalAppData`).
  - macOS: `~/Library/Application Support/forex-ai/`.
  - Linux: `$XDG_DATA_HOME/forex-ai/` (or `~/.local/share/forex-ai/`)
    per the freedesktop XDG Base Directory Specification — "per-user
    configuration should go in `~/.config/appname` and caches …
    into `~/.cache/appname`" (XDG snippet via WebSearch).
  Live disk-free indicator: red (< 5 GiB), amber (5–20 GiB), green
  (> 20 GiB). A "Validate" button runs the write-permission probe.
- **Inputs:** `data_path: PathBuf`.
- **Actions:** sentinel-file write/delete probe; toasted errors
  surface the OS error verbatim. On success, records `data_path`.
- **Skip:** Allowed (defaults to OS-canonical path).
- **Back:** Step 1; license acceptance preserved (Microsoft UX
  wizards: "Preserve user selections through navigation").
- **Time:** ≤ 15 s default; up to 2 min if customised.

Per `TODO(installer_infrastructure_spec)` §8: defaults to OS-
canonical with override allowed.

### Step 3 — Account & profile

- **Purpose:** operator identity, prop-firm preset, monthly target.
- **Mockup:** four-row form.
  Row 1 "Operator name" — optional text, journal tag.
  Row 2 "Prop-firm preset" — dropdown: "FTMO Standard
  (recommended)" / "FTMO Aggressive" / "Custom". Default loads
  `PropFirmConstraints::FTMO_STANDARD` (`max_daily_loss_pct=0.05`,
  `max_overall_drawdown_pct=0.10`,
  `challenge_profit_target_pct=0.10`,
  `min_monthly_net_profit_pct=0.04`, `min_trading_days=10`) from
  `crates/forex-core/src/domain/prop_firm.rs:32`.
  Row 3 "Monthly net profit target" — slider 4 %–25 %, default 4 %.
  **Left stop = 4 %** per operator directive 2026-05-14 verbatim at
  `prop_firm.rs:36` ("operator directive"). Typing 3 % surfaces a
  validator: "Minimum 4 % per operator policy."
  Row 4 "Trading mode" — radio: Backtest / Forward test (default) /
  Live.
- **Inputs:** operator_name, prop_firm_preset,
  monthly_profit_target, trading_mode.
- **Actions:** in-memory `WizardConfig` patch; written at Summary.
- **Skip:** Allowed (FTMO Standard / 4 % / Forward defaults).
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
  - **4.1 Register app.** Walkthrough text: "Visit
    <https://openapi.ctrader.com/> signed into your cTID, then
    *Applications → Add Application*. Set redirect URI to exactly
    `http://127.0.0.1:7777/ctrader/callback`. Paste Client ID and
    Client Secret here." Two fields (the Secret masked), an
    "Open openapi.ctrader.com" launcher, and a "Test these
    credentials" probe that sends `ProtoOAApplicationAuthReq` (2100)
    to `demo.ctraderapi.com:5035` per
    `ctrader_api_full_reference.md` §2.6. Errors
    `CH_CLIENT_AUTH_FAILURE` (101) and `CH_OA_CLIENT_NOT_FOUND`
    (107) are surfaced verbatim.
  - **4.2 Sign in with cTID.** Primary button "Sign in with cTID"
    drives the existing `ProductionCTraderLiveAuthBackend`
    (`ctrader_live_auth.rs:120`): (1) generate CSRF state — F2
    fix at `ctrader_live_auth.rs:38`; (2) bind loopback on first
    free port of `[7777, 7878, 8989]` —
    `bind_loopback_listener` at `ctrader_live_auth.rs:193`; (3)
    open the system browser to
    `https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={cid}&redirect_uri=http://127.0.0.1:{port}/ctrader/callback&scope=trading&product=web&state={csrf}`
    per `ctrader_api_full_reference.md` §2.2; (4) wait up to 300 s
    (`CTRADER_CALLBACK_TIMEOUT` at `ctrader_live_auth.rs:24`); (5)
    exchange the auth code via GET on
    `https://openapi.ctrader.com/apps/token` per
    `ctrader_api_full_reference.md` §2.3. If no port binds, fall
    back to a copy-paste flow (redirect to `https://spotware.com`,
    user pastes the URL back). RFC 8252 §7.3 — "loopback redirect
    URIs use the 'http' scheme … 'http://127.0.0.1:{port}/{path}'"
    (RFC 8252 snippet via WebSearch) — keeps copy-paste a fallback.
  - **4.3 Account picker.** Sends `ProtoOAApplicationAuthReq` (2100)
    + `ProtoOAGetAccountListByAccessTokenReq` (2149) (per §2.6).
    The response (`ProtoOAGetAccountListByAccessTokenRes` 2150)
    carries `repeated ProtoOACtidTraderAccount`. The picker renders
    one row per account (label, broker, account number, currency,
    Live/Demo, type). A radio column selects the default; a
    checkbox column enables additional accounts. "I'll wire more
    accounts later" link below.
  - **4.4 Account auth probe.** Sends `ProtoOAAccountAuthReq` (2102)
    on the primary account; success on `ProtoOAAccountAuthRes`
    (2103). Failure surfaces the broker's error verbatim plus
    remediation hint. Permanent codes (e.g. 106
    `CH_CTID_TRADER_ACCOUNT_NOT_FOUND`) offer "Pick a different
    account" instead of "Retry".
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

- **Purpose:** seed the universe selector so the first backtest is
  one click away.
- **Mockup:** two columns.
  Left = filterable symbol multi-select, populated by
  `ProtoOASymbolsListReq` (2114) → `ProtoOASymbolsListRes` (2115)
  against the primary account from Step 4. Default selection
  `EURUSD`; preset "Top 28 majors" available.
  Right = timeframe checkboxes from `CANONICAL_TIMEFRAMES`
  (`crates/forex-core/src/contracts/temporal.rs:25`): `M1, M3, M5,
  M15, M30, H1, H4, H12, D1, W1, MN1` — **11 entries, H2 deliberately
  absent** per operator directive 2026-05-14 (verbatim "Αν δεν
  υπάρχει Η2 τότε ας μην μπει καθόλου!!!" recorded at
  `temporal.rs:17–24`). Default selection `M5, M15, H1, H4, D1`.
  Inline preview: "N symbols × M timeframes = N×M pairs; ≈ Y MiB."
- **Inputs:** `selected_symbols`, `selected_timeframes`.
- **Actions:** `ProtoOASymbolsListReq` (50 req/s class per
  `ctrader_api_full_reference.md` §3.2). Caches result in
  `<data_path>/cache/symbols_<broker_id>.json`.
- **Skip:** Allowed (defaults to `EURUSD` × `{M5,M15,H1,H4,D1}`).
- **Back:** Step 4.
- **Time:** 30–60 s.

### Step 6 — Historical data download

- **Purpose:** seed the local Polars/Parquet cache so the first
  backtest and ML run are zero-extra-clicks.
- **Mockup:** slider "Months of history" 1/3/6/12/18/24, default 6.
  Forecast block: "≈ N MiB, ≈ T s at 5 req/s" — limit verbatim from
  `ctrader_api_full_reference.md` §3.2 ("a maximum of 5 requests
  per second per connection for any historical data requests").
  On `[Begin download]`, slider becomes a per-(symbol, timeframe)
  progress table with `[Pause]` / `[Resume]` / `[Cancel]`. Cancel
  keeps already-downloaded bars on disk; no fabricated fill.
- **Inputs:** `history_months: u8` (1–24).
- **Actions:** `ProtoOAGetTrendbarsReq` (2137) →
  `ProtoOAGetTrendbarsRes` (2138) with `hasMore` paging. Token-
  bucket gate at 5 req/s; on `REQUEST_FREQUENCY_EXCEEDED` (108),
  exponential backoff. Output:
  `<data_path>/history/<broker_id>/<symbol>/<timeframe>.parquet`
  + sidecar `.complete` (or `.partial` on Cancel).
- **Skip:** Allowed (main app re-prompts later).
- **Back:** Step 5; disabled while download is in flight (Pause /
  Cancel first).
- **Time:** ≈ 60–90 s for 6 months × 6 symbols × 5 timeframes
  (≈ 180 requests at 5 req/s); much longer at 24 months × 28 ×11.

### Step 7 — Hardware compatibility probe

- **Purpose:** detect compute backends and pick a sane ML default.
- **Mockup:** one card per detected device with: model name, RAM
  (CPU) or VRAM (GPU), compute capability, supported precisions
  (FP32/FP16/BF16/FP8), eligible backends with ticks, and a
  "Recommended" badge. Output aggregates
  `HardwareProbe::detect()` (`forex-core/src/system.rs:561`): CUDA
  via `nvidia-smi` (lines 605–685), ROCm via `rocminfo` (line 721+),
  Vulkan/wgpu fallback (`detect_wgpu_hint_accelerators`). For
  NVIDIA, compute capability gates the precision list (`system.rs:
  612–626`). "Recommended" follows existing dispatch in
  `forex-core/src/system/backends.rs` (CUDA > ROCm > Vulkan > CPU);
  user override writes `forced_backend` into
  `hardware_profile.json`.
- **Inputs:** `forced_backend: Option<String>`.
- **Actions:** writes `<data_path>/hardware_profile.json` (full
  probe). Read at training time by
  `forex-models/src/training_orchestrator.rs:480`.
- **Skip:** Allowed (defaults to CPU NdArray).
- **Back:** Step 6.
- **Time:** ≤ 5 s probe; < 30 s review.

### Step 8 — News / sentiment provider

- **Purpose:** wire `forex_core::domain::news_filter` so macro-event
  releases can suppress trading.
- **Mockup:** toggle "Enable news filter" (default off). When on,
  reveals "Provider" dropdown and masked "API key" field, with the
  disclosure "Stored as `secrecy::SecretString`; only sent to your
  chosen provider when the filter runs. See Privacy in Step 10."
- **Inputs:** `news_filter_enabled: bool`, `news_filter_api_key:
  Option<SecretString>`.
- **Actions:** if enabled, ping the provider once to validate. Key
  is held in-memory until Step 10's Apply.
- **Skip:** Allowed; news_filter remains disabled.
- **Back:** Step 7.
- **Time:** ≤ 60 s.

### Step 9 — Auto-start

- **Purpose:** optionally register `forex-app` to run at login.
- **Mockup:** toggle "Start forex-app on system login" + sub-toggle
  (greyed unless main is on) "Start minimised to system tray".
  Per-platform mechanism:
  - Windows: per-user shortcut in `%APPDATA%\Microsoft\Windows\Start
    Menu\Programs\Startup\` (or `HKCU\…\Run`).
  - macOS: `~/Library/LaunchAgents/ai.forex.app.plist`.
  - Linux: `~/.config/autostart/forex-app.desktop` per freedesktop
    Autostart Specification — "user-level: `~/.config/autostart/`"
    is the canonical location (snippet via WebSearch). Minimum
    keys: `[Desktop Entry]`, `Type=Application`, `Name=forex-ai`,
    `Exec=<install_dir>/forex-app --minimized`, `Terminal=false`.
- **Inputs:** `autostart_enabled: bool`, `start_minimized: bool`.
- **Actions:** writes / removes the per-platform artefact. All
  three paths are user-scoped — no UAC / sudo prompt.
- **Skip:** Allowed (default off).
- **Back:** Step 8.
- **Time:** ≤ 10 s.

Cross-reference: `TODO(installer_infrastructure_spec)` §7 owns
exact paths; this step is the user-facing toggle.

### Step 10 — Summary & Apply

- **Purpose:** review every choice, edit-in-place, commit.
- **Mockup:** scrollable review table (see §9 mockup). Each row
  has an `[edit ↑]` link that jumps back to the source step
  preserving every other choice (Microsoft UX wizards: "Preserve
  user selections through navigation"). Footer `[Cancel]` /
  `[Apply ✓]` (Apply is primary, heavier weight per
  `TODO(ui_ux_design_spec)`).
- **Actions (in order):**
  1. Write `<data_path>/config.yaml`.
  2. Write `<data_path>/broker_credentials.toml` (via
     `broker_persistence.rs`; OS keychain when available, else
     file mode 0o600 / ACL-restricted).
  3. Write `<data_path>/hardware_profile.json`.
  4. Write `<data_path>/wizard_complete.json`.
  5. Spawn the historical-data download as a background job
     (non-blocking — Apply returns on enqueue).
  6. Close the modal; open the main app with the "Run your first
     backtest" tour active.
- **Skip:** N/A (terminal).
- **Cancel:** confirm modal "Discard changes? Downloaded history
  (if any) preserved." `[Keep editing]` / `[Discard]`.
- **Time:** ≤ 5 s.

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

| Step | Error class | UX response & recovery |
|------|-------------|------------------------|
| 1 | LICENSE file missing | Fallback to embedded `include_str!` copy; warn-and-continue. |
| 2 | No write permission | Red banner with OS error verbatim; pick another path. |
| 2 | Disk < 5 GiB | Amber banner; continue allowed, warning logged. |
| 3 | Monthly profit < 4 % | Inline validator: "Minimum 4 % per operator policy (2026-05-14)." |
| 4.1 | Empty / malformed creds | Inline validator; re-enter. |
| 4.1 | `CH_CLIENT_AUTH_FAILURE` (101) / `CH_OA_CLIENT_NOT_FOUND` (107) | Broker rejection surfaced verbatim; retry / re-enter. |
| 4.2 | Loopback bind fails on all 3 ports | Offer copy-paste flow (RFC 8252 §7.3 fallback). |
| 4.2 | 5-min callback timeout | Retry / Skip. |
| 4.2 | `state` mismatch | Hard refuse — CSRF block per `ctrader_live_auth.rs:36–43` audit-fix F2. |
| 4.2 | Token exchange `errorCode` field set | Surface broker's `description` verbatim; retry. |
| 4.3 | Empty account list | "Your cTID has no trading accounts — open a demo at ctrader.com." Retry / Skip. |
| 4.4 | `ACCOUNT_NOT_AUTHORIZED` (2) | Back to 4.2 to refresh token. |
| 4.4 | `CONNECTIONS_LIMIT_EXCEEDED` (67) | Retry after user closes other sessions. |
| 5 | `ProtoOASymbolsListReq` timeout | Retry / Skip (broker maintenance window?). |
| 6 | `REQUEST_FREQUENCY_EXCEEDED` (108) | Token-bucket gates this; on bypass, 30 s backoff + resume. |
| 6 | Cancel partway / disk full | Mark `.partial`; main app surfaces Resume affordance. |
| 7 | `nvidia-smi` missing but NVIDIA present | Show wgpu card with "Install NVIDIA driver to enable CUDA". |
| 7 | No GPU at all | CPU-only card shown; no error. |
| 8 | News API ping fails | Surface provider error verbatim; Retry / Disable. |
| 9 | Can't write autostart artefact | "Skip auto-start?" — Skip. |
| 10 | Disk full at write | "Free space and Retry." |
| 10 | macOS keychain locked | Offer file-based fallback with explicit warning. |

---

## §4 — Theming

Design tokens are owned by `ui_ux_design_spec.md` (parallel agent;
not yet present). Until then, the wizard uses these placeholders
flagged `TODO(ui_ux_design_spec)`:

- `color.surface.canvas` `#0E1117` dark / `#FFFFFF` light (modal bg)
- `color.surface.card` `#171A21` / `#F6F8FB` (rows / cards)
- `color.text.primary` `#E6E8EE` / `#101218`
- `color.text.muted` `#8A93A6` / `#5F6A7E`
- `color.accent` `#2F7FF9` (primary buttons, focus ring)
- `color.success` `#2EA86A` / `color.warning` `#E1A227` /
  `color.danger` `#D14545`
- `typography.heading` Inter Semibold 20/24/28
- `typography.body` Inter Regular 14/16; `typography.mono`
  JetBrains Mono 13
- `space.unit` 4 px; `radius.card` 12 px; `focus.ring` 2 px solid
  accent + 2 px offset

The wizard is fully keyboard-navigable: `Tab` / `Shift+Tab`,
`Space` / `Enter`, `Esc` opens Cancel. Microsoft Learn's *UX
checklist for desktop applications* lists missing focus indicators
as a top violation
(<https://learn.microsoft.com/en-us/windows/win32/uxguide/top-violations>) —
the focus ring is mandatory.

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

forex-ai pre-0.5 was portable (all state under `~/.forex-ai/`).
The installed-app wizard detects and migrates.

**Detection (on Step 2 entry):** scan `~/.forex-ai`, `~/forex-ai`,
and `%USERPROFILE%/.forex-ai` for any of `config.yaml`,
`broker_credentials.toml`, `checkpoints/`, `data/`, `history/`.

**Prompt:** modal overlay on Step 2 listing detected payloads with
sizes and checkboxes (Config, Broker credentials, Cached history,
Model checkpoints, OAuth refresh token). Disk-free check at
destination; post-migration radio "Keep / Delete / Leave decision
for later"; `[Skip migration]` / `[Migrate now]`.

**Semantics:**

- Atomic per file — copy + verify (size + SHA-256), then remove
  source. Failure leaves **both** dirs intact; user picks Retry /
  Keep-both / Rollback.
- The OAuth refresh token is reused as-is — no browser re-auth.
- "Delete source" requires a second confirmation modal before
  `rm -rf`.

**Skip:** the portable directory is left alone; the main app
surfaces a one-time banner on startup ("Legacy `~/.forex-ai/`
detected — [Migrate now] [Dismiss]").

---

## §7 — Telemetry / privacy

### 7.1 Default

**No telemetry.** The wizard does not collect any data unless the
user explicitly opts in to crash reports in §10's Apply step. This
matches the operator's overall posture (no-synthetic-data, no
unsolicited network calls).

### 7.2 Optional crash reports

The Summary row "Crash reports — disabled (default)" opens a
disclosure modal listing the two columns "What is sent" (panic
message, stack trace, OS+version, forex-ai version, sanitised
paths, hardware tier) and "What is NEVER sent" (OAuth secrets,
tokens, account IDs, symbols, trading history, model checkpoints,
API keys). `[Decline]` / `[Opt in]`.

Sentry's own GDPR guidance — "obtain opt-in consent for Sentry SDKs
via your website or app consent banner" (Sentry *Best Practices for
GDPR Compliance*, snippet via WebSearch) — is enforced as a hard
default-off. The choice is recorded in `wizard_complete.json` as
`telemetry_opt_in: bool`.

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

### 10.1 Windows admin rights?

**No.** Default install path `%LOCALAPPDATA%\forex-ai\` is user-
writeable; all wizard writes are per-user; autostart uses per-user
shortcut / `HKCU` — no UAC prompt. Only triggers UAC if the user
manually overrides Step 2 to `C:\Program Files\`. **Follow-up:**
confirm `installer_infrastructure_spec` declares the WiX bundle
`InstallScope="perUser"`.

### 10.2 macOS notarization on the wizard binary

Notarization is required for distribution; with the in-app wizard
architecture from §1.2 the wizard is part of the (notarized)
`forex-app` binary — one artefact, not two. Apple's *Notarizing
macOS software* (snippet via WebSearch) recommends notarizing
"even if you plan to distribute it from your own website".
**Follow-up:** verify Apple still accepts notarization for
binaries that bind loopback ports without an explicit entitlement
(notarization rejections rose Feb–May 2026 — search snippet — but
on entitlement misuse, not loopback binding per se).

### 10.3 Loopback OAuth on macOS App Sandbox

forex-ai is **not** App-Store-distributed, so the App Sandbox does
not apply. Direct download + notarization + Gatekeeper allows
arbitrary loopback binding. If MAS distribution becomes a goal,
the `com.apple.security.network.server` entitlement + a non-Sandbox
LaunchAgent path would be required — out of scope.

### 10.4 Wizard re-run while forex-app is running

If the user invokes `forex-app --wizard` with a live session,
recommendation: reject with "wizard already running in main app
instance" (option a in the trade-off). Option (b) — re-use live
session as a settings editor — risks confusion with the eventual
Settings panel. **Follow-up:** confirm with `ui_ux_design_spec`.

### 10.5 Symbol-list pagination

`ProtoOASymbolsListRes` (2115) is documented as a single response;
no `hasMore`. If the broker silently truncates, Step 5 shows fewer
symbols than reality. Operator can re-enter Step 5 to retry.
**Follow-up:** watch `spotware_proto_freshness.md`.

### 10.6 FTMO Aggressive preset

The mockup mentions an "Aggressive" preset but `prop_firm.rs` only
exposes `FTMO_STANDARD`. Safe default: hide "Aggressive" until
the constants exist. **Follow-up:** ask operator whether to add.

### 10.7 Token rotation race on re-run

Recommendation: the wizard reads only
`<data_path>/broker_credentials.toml`, which the daemon updates
atomically. **Follow-up:** confirm `broker_persistence.rs` has a
read-only snapshot API.

---

## §11 — Acceptance criteria

The implementation must satisfy:

1. A zero-prior-knowledge user finishes the wizard in ≤ 15 min and
   ends with: OAuth refresh token persisted, default
   symbol+timeframe set selected, ≥ 1 month EURUSD M5 history on
   disk, hardware profile written.
2. All non-License steps are skippable; on skip,
   `state="completed_with_warnings"` and the main app banners
   remaining work.
3. Re-running preserves every prior choice (Microsoft UX wizards
   guideline).
4. OAuth tokens are persisted only after the flow completes — no
   half-written `broker_credentials.toml`.
5. The wizard never transmits any value outside the broker and (if
   opt-in) the news provider and crash-reporting service.
6. The TUI (`forex-cli wizard`) renders all ten steps with
   keyboard-only navigation (§8.2).
7. Migration from `~/.forex-ai/` is atomic and reversible until
   source deletion is confirmed.
8. H2 is **never** offered in the timeframe selector
   (`temporal.rs:17–24`).
9. The 4 % monthly profit floor cannot be reduced
   (`prop_firm.rs:36`).
10. No synthetic / fabricated fallback data is ever shown. Failed
    broker calls surface as failures.

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

- Operator-policy values read directly from working copy at
  `/home/user/forex-ai/`.
- External UX guidance attributed inline; "snippet via WebSearch"
  indicates WebFetch 403 with WebSearch excerpt used.
- cTrader payload IDs / names taken from
  `docs/audits/research/ctrader_api_full_reference.md`, itself
  built from vendored `.proto` at `crates/forex-app/proto/`.
- H2 prohibition + 4 % floor are operator directives recorded at
  `temporal.rs:17–24` and `prop_firm.rs:36` (both 2026-05-14).
- No code changed — research only.

---

— END —

(External citations are enumerated in §0; internal identifiers in
§12. No separate "sources cited" appendix is repeated here.)
