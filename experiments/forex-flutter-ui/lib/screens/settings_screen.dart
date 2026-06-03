// Settings — consolidated single-tab home for ALL backend + frontend
// configuration the user can touch.
//
// **F-327 (2026-05-29 rebuild)**: per the Codex mockup, the 4 sidebar
// entries that used to live under the "System" group (Broker Setup,
// Risk, Hardware, Data Bootstrap) plus the standalone Advanced
// Settings knob editor (#238) all collapse into a single Settings tab
// with an internal sub-tab bar:
//
//   🔐 Account       — cTrader OAuth + saved credentials
//   ⚙  App           — data dir, news, LLM, news-trading mode + raw YAML
//   ⚠  Risk          — prop-firm preset + drawdown caps
//   🛠 Advanced      — full 42+ knob editor (inline now, not a modal)
//   🖥 Hardware      — CPU/GPU probe (read-only)
//   📂 Data          — historical bootstrap + CSV import
//
// The old `SettingsScreen` (credentials form + 5 app settings + raw
// YAML editor) lives on as `AppSettingsScreen` and is rendered inside
// the "App" tab — its state/providers/tests are unchanged.
//
// The credentials form posts to /broker/credentials which writes
// broker_credentials.toml under %APPDATA%/neoethos/. After save, the
// operator goes to Settings → Account → Re-authenticate to do the
// actual OAuth flow against the freshly-saved client_id/secret.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../state/locale_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
import '_placeholder.dart';
import 'advanced_settings_screen.dart';
import 'broker_setup_screen.dart';
import 'data_bootstrap_screen.dart';
import 'hardware_screen.dart';
import 'help_screen.dart';
import 'risk_screen.dart';

/// The consolidated Settings tab.
///
/// **F-327 (2026-05-29)**: was a single rich screen exposed via the
/// sidebar's `Settings` entry. Now an outer TabBar wrapper whose body
/// hosts the 6 sub-tabs of the Codex mockup's consolidated Settings
/// group. The pre-F-327 content lives at the "App" sub-tab as
/// `AppSettingsScreen`.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _controller;

  // Section list — kept in this exact order so the "primary path" tab
  // (Account / App) is on the left and the rarely-touched ones
  // (Hardware / Data) are on the right. The Help link sits in the
  // top-right corner of the tab strip instead of being a tab body.
  // Labels + tooltips are localized at build time (see `_localizedTabs`);
  // only the count is fixed here so the TabController can be sized in
  // initState without a BuildContext.
  static const _tabCount = 6;

  /// (label, icon, tooltip) tuples sourced from [AppLocalizations]. The
  /// emoji icons stay verbatim; only the chrome text is translated.
  List<(String, String, String)> _localizedTabs(AppLocalizations l10n) => [
        (l10n.settingsTabAccount, '🔐', l10n.settingsTabAccountTooltip),
        (l10n.settingsTabApp, '⚙', l10n.settingsTabAppTooltip),
        (l10n.settingsTabRisk, '⚠', l10n.settingsTabRiskTooltip),
        (l10n.settingsTabAdvanced, '🛠', l10n.settingsTabAdvancedTooltip),
        (l10n.settingsTabHardware, '🖥', l10n.settingsTabHardwareTooltip),
        (l10n.settingsTabData, '📂', l10n.settingsTabDataTooltip),
      ];

  @override
  void initState() {
    super.initState();
    _controller = TabController(length: _tabCount, vsync: this);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _SettingsTabStrip(controller: _controller, tabs: _localizedTabs(l10n)),
        const SizedBox(height: NeoethosTokens.spSm),
        Expanded(
          child: TabBarView(
            controller: _controller,
            physics: const NeverScrollableScrollPhysics(),
            children: const [
              // Account = OAuth + saved credentials. BrokerSetupScreen
              // wraps the full flow.
              SingleChildScrollView(child: BrokerSetupScreen()),
              // App = original SettingsScreen body. Stays scrollable so
              // the long form (credentials → settings → account picker
              // → raw YAML) still fits on smaller windows.
              SingleChildScrollView(child: AppSettingsScreen()),
              // Risk = prop-firm preset + drawdown caps.
              SingleChildScrollView(child: RiskScreen()),
              // Advanced = the 2-pane knob editor (#238). It manages
              // its own scroll surfaces so don't wrap in another
              // SingleChildScrollView (that'd nest vertical scrolls).
              AdvancedSettingsScreen(),
              // Hardware = read-only probe.
              SingleChildScrollView(child: HardwareScreen()),
              // Data = historical bootstrap + CSV/Parquet import.
              SingleChildScrollView(child: DataBootstrapScreen()),
            ],
          ),
        ),
      ],
    );
  }
}

class _SettingsTabStrip extends StatelessWidget {
  final TabController controller;
  final List<(String, String, String)> tabs; // (label, icon, tooltip)
  const _SettingsTabStrip({required this.controller, required this.tabs});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Container(
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      padding: const EdgeInsets.only(right: 4),
      child: Row(
        children: [
          Expanded(
            child: TabBar(
              controller: controller,
              isScrollable: true,
              labelColor: NeoethosTokens.accent,
              unselectedLabelColor: NeoethosTokens.textMuted,
              indicatorColor: NeoethosTokens.accent,
              labelStyle: const TextStyle(
                fontSize: NeoethosTokens.fsBody,
                fontWeight: FontWeight.w700,
              ),
              unselectedLabelStyle: const TextStyle(
                fontSize: NeoethosTokens.fsBody,
                fontWeight: FontWeight.w500,
              ),
              tabs: [
                for (final (label, icon, tooltip) in tabs)
                  Tooltip(
                    message: tooltip,
                    waitDuration: const Duration(milliseconds: 600),
                    child: Tab(text: '$icon  $label'),
                  ),
              ],
            ),
          ),
          // Help link sits in the top-right so the user can always
          // jump to the F1 docs from inside Settings, without it
          // being a separate sub-tab (Help is its own full-screen
          // experience).
          TextButton.icon(
            onPressed: () => showHelpDialog(context),
            icon: const Icon(Icons.help_outline,
                size: 16, color: NeoethosTokens.textMuted),
            label: Text(
              l10n.settingsHelpF1,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                fontWeight: FontWeight.w600,
                color: NeoethosTokens.textMuted,
              ),
            ),
            style: TextButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              minimumSize: Size.zero,
              tapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
          ),
        ],
      ),
    );
  }
}

/// "App Settings" tab body — broker credentials + the 5 app-wide
/// settings (data dir, news source, OpenAI model, news-trading mode,
/// account picker) + the F-312 raw YAML editor.
///
/// **F-327 (2026-05-29 rebuild)**: was the public `SettingsScreen`
/// exposed directly to the sidebar. Now lives as one tab inside the
/// consolidated `SettingsScreen` (`settings_consolidated_screen.dart`).
/// The class was renamed but the body is unchanged — all existing
/// fields, providers, and tests still apply to it.
class AppSettingsScreen extends ConsumerStatefulWidget {
  const AppSettingsScreen({super.key});

  @override
  ConsumerState<AppSettingsScreen> createState() => _AppSettingsScreenState();
}

class _AppSettingsScreenState extends ConsumerState<AppSettingsScreen> {
  final _clientIdCtrl = TextEditingController();
  final _clientSecretCtrl = TextEditingController();
  final _accountIdCtrl = TextEditingController();
  String _environment = 'Demo';
  bool _busy = false;
  String? _resultMessage;
  bool _resultOk = false;
  bool _loaded = false;
  bool _secretConfigured = false;
  String _secretMask = '';

  @override
  void initState() {
    super.initState();
    _loadCurrent();
  }

  Future<void> _loadCurrent() async {
    try {
      final r = await ref.read(backendClientProvider).fetchBrokerCredentials();
      if (!mounted) return;
      setState(() {
        _clientIdCtrl.text = (r['clientId'] as String?) ?? '';
        _accountIdCtrl.text = (r['accountId'] as String?) ?? '';
        _environment =
            (r['environment'] as String?) == 'Live' ? 'Live' : 'Demo';
        _secretConfigured = (r['clientSecretConfigured'] as bool?) ?? false;
        _secretMask = (r['clientSecretMask'] as String?) ?? '';
        _loaded = true;
      });
    } catch (_) {
      if (mounted) setState(() => _loaded = true);
    }
  }

  @override
  void dispose() {
    _clientIdCtrl.dispose();
    _clientSecretCtrl.dispose();
    _accountIdCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final l10n = AppLocalizations.of(context)!;
    final clientId = _clientIdCtrl.text.trim();
    final clientSecret = _clientSecretCtrl.text.trim();
    final accountId = _accountIdCtrl.text.trim();

    // Empty-secret semantics:
    //   * Empty Client ID is OK if the server already has one saved
    //     (UI shows it pre-filled; user might clear and leave blank
    //     to keep the existing).
    //   * Empty Client Secret is OK ONLY when a secret is already
    //     saved server-side (`_secretConfigured == true`) — the form
    //     literally says "Leave blank to keep". This is the common
    //     case when the user only wants to change account-id or
    //     environment.
    //   * If neither side has anything we still need both — the
    //     backend will catch that and return 400 with a clear
    //     message, but we pre-check here for snappier UX.
    final clientIdMissing = clientId.isEmpty && _clientIdCtrl.text.isEmpty;
    final clientSecretMissing = clientSecret.isEmpty && !_secretConfigured;
    if (clientIdMissing || clientSecretMissing) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.sell,
          content: Text(l10n.settingsCredentialsRequired),
        ),
      );
      return;
    }
    setState(() => _busy = true);
    try {
      final r = await ref.read(backendClientProvider).saveBrokerCredentials(
            clientId: clientId,
            clientSecret: clientSecret,
            environment: _environment,
            accountId: accountId,
          );
      if (!mounted) return;
      final ok = r['ok'] == true;
      final msg = (r['message'] as String?) ??
          (ok ? l10n.settingsSaved : l10n.settingsUnknownResponse);
      setState(() {
        _resultOk = ok;
        _resultMessage = msg;
        _secretConfigured = true;
        _clientSecretCtrl.clear();
      });
      // Force broker-status to re-read with the new creds on next tick.
      ref.invalidate(brokerStatusProvider);
      ref.invalidate(accountSnapshotProvider);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ok ? NeoethosTokens.buy : NeoethosTokens.warning,
          content: Text(msg),
          duration: const Duration(seconds: 5),
        ),
      );
    } on DioException catch (e) {
      final msg = describeError(e);
      if (!mounted) return;
      setState(() {
        _resultOk = false;
        _resultMessage = msg;
      });
      showTranslatedErrorSnackbar(context, e, prefix: l10n.settingsSaveFailed);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  /// F-333: the operator picked a different cTID in the dropdown. Make it
  /// the *active* account server-side — the backend reorders
  /// broker_credentials.toml so `accounts.first()` (what the runtime
  /// trades) becomes this id. Runtime hot-swap isn't in scope yet, so we
  /// prompt for a restart, which is the honest MVP.
  ///
  /// Kept separate from `_save`: selection is a one-click action that
  /// shouldn't require re-submitting the whole credentials form, and it
  /// must NOT fire on the picker's automatic stale-id correction (that
  /// path calls `onPicked` only).
  Future<void> _selectAccount(String accountId) async {
    final l10n = AppLocalizations.of(context)!;
    final id = accountId.trim();
    if (id.isEmpty) return;
    try {
      final r = await ref
          .read(backendClientProvider)
          .selectBrokerAccount(accountId: id);
      if (!mounted) return;
      final ok = r['ok'] == true;
      final selected = (r['selectedAccountId'] as String?) ?? id;
      // Force broker-status + account snapshot to re-read on next tick so
      // the status bar reflects the pending change (the actual swap lands
      // after restart, but invalidating avoids showing stale-cached data).
      ref.invalidate(brokerStatusProvider);
      ref.invalidate(accountSnapshotProvider);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ok ? NeoethosTokens.buy : NeoethosTokens.warning,
          content: Text(
            ok
                ? l10n.accountSwitchedRestart(selected)
                : l10n.accountSwitchUnexpected,
          ),
          duration: const Duration(seconds: 6),
        ),
      );
    } on DioException catch (e) {
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e,
          prefix: l10n.settingsAccountSelectFailed);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final asyncSettings = ref.watch(settingsProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(
            title: l10n.settingsTitle,
            subtitle: l10n.settingsHeaderSubtitle,
          ),
          const _LanguageCard(),
          _credentialsCard(context),
          asyncSettings.when(
            data: (s) => _configCard(s),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(
                error: err, title: l10n.settingsCouldNotLoad),
          ),
          // **2026-05-25 — task #238 supersedes #193**: the live
          // knob editor (`/settings/knob-catalog`) replaces the
          // read-only YAML dump. Operators can now edit every
          // catalogued knob, apply presets (Conservative/Balanced/
          // Aggressive), and see per-knob help inline without
          // touching config.yaml. The legacy read-only viewer is
          // preserved below as a fallback for diagnostics.
          const _AdvancedKnobEditorCard(),
          const SizedBox(height: 16),
          const _AdvancedConfigCard(),
        ],
      ),
    );
  }

  Widget _credentialsCard(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    if (!_loaded) {
      return SectionCard(
        title: l10n.settingsCredentialsTitle,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Text(
            l10n.commonLoading,
            style:
                const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
          ),
        ),
      );
    }
    return SectionCard(
      title: l10n.settingsCredentialsTitleOptional,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            l10n.settingsCredentialsIntro,
            style:
                const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _clientIdCtrl,
            enabled: !_busy,
            decoration: const InputDecoration(
              labelText: 'Client ID',
              isDense: true,
              border: OutlineInputBorder(),
              hintText: 'e.g. 26884_ZJBPTG1PzFd0Pw48UvjTmjK8...',
            ),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: _clientSecretCtrl,
            enabled: !_busy,
            obscureText: true,
            decoration: InputDecoration(
              labelText: 'Client Secret',
              isDense: true,
              border: const OutlineInputBorder(),
              hintText: _secretConfigured
                  ? l10n.settingsClientSecretSavedHint(_secretMask)
                  : l10n.settingsClientSecretPasteHint,
              helperText: _secretConfigured
                  ? l10n.settingsClientSecretSavedHelper
                  : null,
            ),
          ),
          const SizedBox(height: 10),
          // Account picker — auto-populated from /broker/accounts so
          // the user never has to type a numeric cTID by hand. Falls
          // back to the free-text TextField when the catalog isn't
          // available yet (haven't OAuthed, or credentials still
          // blank, or network down).
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                  child: _AccountPicker(
                currentAccountId: _accountIdCtrl.text,
                enabled: !_busy,
                onPicked: (id) => setState(() {
                  _accountIdCtrl.text = id;
                }),
                onUserPicked: _selectAccount,
                fallback: TextField(
                  controller: _accountIdCtrl,
                  enabled: !_busy,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(
                    labelText: 'Account ID (cTID)',
                    isDense: true,
                    border: const OutlineInputBorder(),
                    hintText: l10n.settingsAccountIdHint,
                    helperText: l10n.settingsAccountIdHelper,
                  ),
                ),
              )),
              const SizedBox(width: 12),
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: DropdownButton<String>(
                  value: _environment,
                  items: const [
                    DropdownMenuItem(value: 'Demo', child: Text('Demo')),
                    DropdownMenuItem(value: 'Live', child: Text('Live')),
                  ],
                  onChanged: _busy
                      ? null
                      : (v) {
                          if (v != null) setState(() => _environment = v);
                        },
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              FilledButton.icon(
                onPressed: _busy ? null : _save,
                icon: _busy
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.save, size: 16),
                label: Text(_busy
                    ? l10n.settingsSavingCredentials
                    : l10n.settingsSaveCredentials),
              ),
              if (_resultMessage != null) ...[
                const SizedBox(width: 12),
                Flexible(
                  child: Text(
                    _resultMessage!,
                    style: TextStyle(
                      fontSize: 11,
                      color:
                          _resultOk ? NeoethosTokens.buy : NeoethosTokens.sell,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _configCard(SettingsSnapshot s) {
    // Delegate to a stateful child so the text controllers
    // (`_dataDirCtrl` etc.) survive parent rebuilds when the user
    // toggles the cTrader credentials form above.
    return _AppSettingsCard(snapshot: s);
  }
}

/// Settings → App: UI language picker (Stage 1a, 2026-06-03). Writes the
/// in-memory [localeProvider]; persistence to the backend config lands in
/// Stage 1b. Its own labels come from [AppLocalizations], so flipping the
/// segments re-renders this card — and the whole app — in the chosen language
/// immediately, which is the live proof the i18n wiring works.
class _LanguageCard extends ConsumerWidget {
  const _LanguageCard();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final current = ref.watch(localeProvider).languageCode;
    return SectionCard(
      title: l10n.language,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            l10n.languageHint,
            style: const TextStyle(
              fontSize: 11,
              color: NeoethosTokens.textMuted,
            ),
          ),
          const SizedBox(height: 10),
          SegmentedButton<String>(
            segments: [
              ButtonSegment(value: 'en', label: Text(l10n.languageEnglish)),
              ButtonSegment(value: 'el', label: Text(l10n.languageGreek)),
            ],
            selected: {current == 'el' ? 'el' : 'en'},
            showSelectedIcon: false,
            onSelectionChanged: (sel) async {
              final code = sel.first;
              // Apply immediately for a snappy switch, then persist to
              // config.yaml (system.ui_locale) so it survives restarts.
              ref.read(localeProvider.notifier).setLanguage(code);
              try {
                await ref
                    .read(backendClientProvider)
                    .saveSettings(uiLocale: code);
              } catch (_) {
                // Non-fatal: the in-memory switch already applied; the
                // choice just isn't persisted until the next successful save.
              }
            },
          ),
        ],
      ),
    );
  }
}

/// Editable App-Settings form. POSTs to `/settings` which merges the
/// 4 exposed fields into the on-disk `config.yaml` (leaving the
/// ~200+ unexposed fields untouched) and rewrites the YAML in place.
class _AppSettingsCard extends ConsumerStatefulWidget {
  final SettingsSnapshot snapshot;
  const _AppSettingsCard({required this.snapshot});

  @override
  ConsumerState<_AppSettingsCard> createState() => _AppSettingsCardState();
}

class _AppSettingsCardState extends ConsumerState<_AppSettingsCard> {
  late final TextEditingController _dataDirCtrl;
  late final TextEditingController _newsSourceCtrl;
  late bool _newsEnabled;

  /// Snake_case id matching `crate::config::NewsTradingMode`.
  /// Defaults to `block_on_news` (safe).
  late String _newsTradingMode;
  // Discovery search-budget knobs (models.prop_search_*).
  late final TextEditingController _searchPopCtrl;
  late final TextEditingController _searchGenCtrl;
  late final TextEditingController _searchMaxHoursCtrl;
  late final TextEditingController _searchMaxIndCtrl;
  late final TextEditingController _searchPortfolioCtrl;
  late final TextEditingController _searchCorrCtrl;
  late final TextEditingController _searchMaxRowsCtrl;
  bool _busy = false;
  String? _message;
  bool _messageOk = false;

  @override
  void initState() {
    super.initState();
    final s = widget.snapshot;
    _dataDirCtrl = TextEditingController(text: s.dataDir);
    _newsSourceCtrl = TextEditingController(text: s.newsCalendarSource);
    _newsEnabled = s.newsCalendarEnabled;
    _newsTradingMode =
        s.newsTradingMode.isEmpty ? 'block_on_news' : s.newsTradingMode;
    _searchPopCtrl = TextEditingController(text: '${s.searchPopulation}');
    _searchGenCtrl = TextEditingController(text: '${s.searchGenerations}');
    _searchMaxHoursCtrl =
        TextEditingController(text: s.searchMaxHours.toString());
    _searchMaxIndCtrl = TextEditingController(text: '${s.searchMaxIndicators}');
    _searchPortfolioCtrl =
        TextEditingController(text: '${s.searchPortfolioSize}');
    _searchCorrCtrl =
        TextEditingController(text: s.searchCorrThreshold.toString());
    _searchMaxRowsCtrl = TextEditingController(text: '${s.searchMaxRows}');
  }

  @override
  void didUpdateWidget(covariant _AppSettingsCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Keep the text fields in sync if the snapshot changed (e.g. the
    // user clicked Save and the provider re-emitted) — but never
    // clobber what the user is actively typing into a focused field.
    final s = widget.snapshot;
    if (!_busy && _dataDirCtrl.text != s.dataDir) {
      _dataDirCtrl.text = s.dataDir;
    }
    if (!_busy && _newsSourceCtrl.text != s.newsCalendarSource) {
      _newsSourceCtrl.text = s.newsCalendarSource;
    }
  }

  @override
  void dispose() {
    _dataDirCtrl.dispose();
    _newsSourceCtrl.dispose();
    _searchPopCtrl.dispose();
    _searchGenCtrl.dispose();
    _searchMaxHoursCtrl.dispose();
    _searchMaxIndCtrl.dispose();
    _searchPortfolioCtrl.dispose();
    _searchCorrCtrl.dispose();
    _searchMaxRowsCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final l10n = AppLocalizations.of(context)!;
    final dataDir = _dataDirCtrl.text.trim();
    final newsSource = _newsSourceCtrl.text.trim();
    if (dataDir.isEmpty) {
      _showSnack(l10n.dataBootstrapDirBlank, ok: false);
      return;
    }
    if (newsSource.isEmpty) {
      _showSnack(l10n.settingsNewsSourceBlank, ok: false);
      return;
    }
    setState(() => _busy = true);
    try {
      await ref.read(backendClientProvider).saveSettings(
            dataDir: dataDir,
            newsCalendarEnabled: _newsEnabled,
            newsCalendarSource: newsSource,
            newsTradingMode: _newsTradingMode,
            searchPopulation: int.tryParse(_searchPopCtrl.text.trim()),
            searchGenerations: int.tryParse(_searchGenCtrl.text.trim()),
            searchMaxHours: double.tryParse(_searchMaxHoursCtrl.text.trim()),
            searchMaxIndicators: int.tryParse(_searchMaxIndCtrl.text.trim()),
            searchPortfolioSize: int.tryParse(_searchPortfolioCtrl.text.trim()),
            searchCorrThreshold: double.tryParse(_searchCorrCtrl.text.trim()),
            searchMaxRows: int.tryParse(_searchMaxRowsCtrl.text.trim()),
          );
      if (!mounted) return;
      setState(() {
        _messageOk = true;
        _message = l10n.settingsConfigUpdated;
      });
      // Refresh the snapshot so the parent screen and any other
      // consumers of settingsProvider see the new value.
      ref.invalidate(settingsProvider);
      _showSnack(l10n.settingsSavedToConfig, ok: true);
    } on DioException catch (e) {
      if (!mounted) return;
      final msg = describeError(e);
      setState(() {
        _messageOk = false;
        _message = msg;
      });
      showTranslatedErrorSnackbar(context, e, prefix: l10n.settingsSaveFailed);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messageOk = false;
        _message = e.toString();
      });
      _showSnack(l10n.settingsSaveError(describeError(e)), ok: false);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _showSnack(String msg, {required bool ok}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: ok ? NeoethosTokens.buy : NeoethosTokens.sell,
        content: Text(msg),
        duration: const Duration(seconds: 4),
      ),
    );
  }

  /// Compact labeled number field for a discovery search knob.
  Widget _knob(TextEditingController c, String label, String help) => SizedBox(
        width: 168,
        child: TextField(
          controller: c,
          enabled: !_busy,
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
          decoration: InputDecoration(
            labelText: label,
            isDense: true,
            border: const OutlineInputBorder(),
            helperText: help,
            helperMaxLines: 3,
            helperStyle: const TextStyle(fontSize: 10),
          ),
        ),
      );

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return SectionCard(
      title: l10n.settingsAppSettingsTitle,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            l10n.settingsAppSettingsIntro,
            style:
                const TextStyle(fontSize: 11, color: NeoethosTokens.textMuted),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _dataDirCtrl,
            enabled: !_busy,
            decoration: InputDecoration(
              labelText: l10n.settingsDataDirLabel,
              isDense: true,
              border: const OutlineInputBorder(),
              helperText: l10n.settingsDataDirHelper,
            ),
          ),
          const SizedBox(height: 10),
          // ── Discovery search budget (models.prop_search_*) ──────────
          // ExpansionTile's header is a ListTile, which needs a Material
          // ancestor for its ink; this card's SectionCard paints a coloured
          // DecoratedBox between the tile and the page Material, so host the
          // ink on a local transparent Material (no visual change).
          Material(
            type: MaterialType.transparency,
            child: ExpansionTile(
              tilePadding: EdgeInsets.zero,
              childrenPadding: const EdgeInsets.only(bottom: 8),
              title: Text(
                l10n.settingsSearchBudgetTitle,
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
              subtitle: Text(
                l10n.settingsSearchBudgetSubtitle,
                style: const TextStyle(
                    fontSize: 11, color: NeoethosTokens.textMuted),
              ),
              children: [
                Wrap(
                  spacing: 12,
                  runSpacing: 12,
                  children: [
                    _knob(_searchMaxHoursCtrl, l10n.settingsKnobMaxHoursLabel,
                        l10n.settingsKnobMaxHoursHelp),
                    _knob(_searchGenCtrl, l10n.settingsKnobGenerationsLabel,
                        l10n.settingsKnobGenerationsHelp),
                    _knob(_searchPopCtrl, l10n.settingsKnobPopulationLabel,
                        l10n.settingsKnobPopulationHelp),
                    _knob(
                        _searchMaxIndCtrl,
                        l10n.settingsKnobMaxIndicatorsLabel,
                        l10n.settingsKnobMaxIndicatorsHelp),
                    _knob(
                        _searchPortfolioCtrl,
                        l10n.settingsKnobPortfolioSizeLabel,
                        l10n.settingsKnobPortfolioSizeHelp),
                    _knob(_searchCorrCtrl, l10n.settingsKnobCorrThresholdLabel,
                        l10n.settingsKnobCorrThresholdHelp),
                    _knob(_searchMaxRowsCtrl, l10n.settingsKnobMaxRowsLabel,
                        l10n.settingsKnobMaxRowsHelp),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _newsSourceCtrl,
                  enabled: !_busy,
                  decoration: InputDecoration(
                    labelText: l10n.settingsNewsSourceLabel,
                    isDense: true,
                    border: const OutlineInputBorder(),
                    helperText: l10n.settingsNewsSourceHelper,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Row(
                  children: [
                    Switch(
                      value: _newsEnabled,
                      onChanged: _busy
                          ? null
                          : (v) => setState(() => _newsEnabled = v),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _newsEnabled
                          ? l10n.settingsCalendarOn
                          : l10n.settingsCalendarOff,
                      style: TextStyle(
                        fontSize: 11,
                        color: _newsEnabled
                            ? NeoethosTokens.buy
                            : NeoethosTokens.textFaint,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          // News-trading mode picker (#117). Default is BlockOnNews —
          // pause new orders inside the kill window. AllowAlways and
          // WarnOnly are opt-in for operators with event-driven
          // strategies (breakout-on-news, news-fade) who explicitly
          // want to trade through high-impact events.
          Text(
            l10n.settingsNewsModeTitle,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: NeoethosTokens.textPrimary,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            l10n.settingsNewsModeDescription,
            style:
                const TextStyle(fontSize: 11, color: NeoethosTokens.textMuted),
          ),
          const SizedBox(height: 8),
          _NewsTradingModeRow(
            id: 'block_on_news',
            label: l10n.settingsNewsModeBlockLabel,
            description: l10n.settingsNewsModeBlockDescription,
            selected: _newsTradingMode == 'block_on_news',
            busy: _busy,
            onPick: () => setState(() => _newsTradingMode = 'block_on_news'),
          ),
          _NewsTradingModeRow(
            id: 'allow_always',
            label: l10n.settingsNewsModeAllowLabel,
            description: l10n.settingsNewsModeAllowDescription,
            selected: _newsTradingMode == 'allow_always',
            busy: _busy,
            onPick: () => setState(() => _newsTradingMode = 'allow_always'),
          ),
          _NewsTradingModeRow(
            id: 'warn_only',
            label: l10n.settingsNewsModeWarnLabel,
            description: l10n.settingsNewsModeWarnDescription,
            selected: _newsTradingMode == 'warn_only',
            busy: _busy,
            onPick: () => setState(() => _newsTradingMode = 'warn_only'),
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              FilledButton.icon(
                onPressed: _busy ? null : _save,
                icon: _busy
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.save, size: 16),
                label: Text(_busy
                    ? l10n.settingsSavingSettings
                    : l10n.settingsSaveSettings),
              ),
              if (_message != null) ...[
                const SizedBox(width: 12),
                Flexible(
                  child: Text(
                    _message!,
                    style: TextStyle(
                      fontSize: 11,
                      color:
                          _messageOk ? NeoethosTokens.buy : NeoethosTokens.sell,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }
}

/// Account picker — Dropdown sourced from `/broker/accounts`. Falls
/// back to the parent's `fallback` TextField when:
///   * The OAuth token isn't saved yet (Re-authenticate must run first)
///   * /broker/accounts returns an error
///   * The list comes back empty (token granted zero accounts)
///
/// This is the visible cure for the `CH_ACCESS_TOKEN_INVALID` loop —
/// users now pick a *real* cTID from the granted set instead of
/// guessing at a stale value left over in broker_credentials.toml.
class _AccountPicker extends ConsumerWidget {
  final String currentAccountId;
  final bool enabled;

  /// Fired for AUTOMATIC corrections only — e.g. the saved cTID isn't in
  /// the granted set, so we snap the local field to the first available.
  /// This must stay side-effect-free (local state only); it runs in a
  /// post-frame callback on every build and must NOT hit the backend.
  final ValueChanged<String> onPicked;

  /// Fired only when the OPERATOR actively chooses a row from the
  /// dropdown. F-333: this is the one that promotes the account to
  /// active server-side (reorders broker_credentials.toml).
  final ValueChanged<String> onUserPicked;
  final Widget fallback;
  const _AccountPicker({
    required this.currentAccountId,
    required this.enabled,
    required this.onPicked,
    required this.onUserPicked,
    required this.fallback,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(brokerAccountsProvider);
    return async.when(
      data: (snap) {
        if (snap.accounts.isEmpty) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              fallback,
              const SizedBox(height: 4),
              Text(
                l10n.settingsAccountPickerEmpty,
                style: const TextStyle(
                  fontSize: 10,
                  color: NeoethosTokens.warning,
                ),
              ),
            ],
          );
        }
        // If the saved account_id isn't in the granted set, fall back
        // to first available — that's typically what the user meant
        // anyway, and the consent screen has already accepted it.
        final ids = snap.accounts.map((a) => a.accountId).toList();
        final selected =
            ids.contains(currentAccountId) ? currentAccountId : ids.first;
        if (selected != currentAccountId) {
          // Post-frame nudge so we don't setState during build.
          WidgetsBinding.instance
              .addPostFrameCallback((_) => onPicked(selected));
        }
        return InputDecorator(
          decoration: InputDecoration(
            labelText: l10n.settingsAccountPickerLabel(snap.accountCount),
            isDense: true,
            border: const OutlineInputBorder(),
            helperText: l10n.settingsAccountPickerHelper,
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              isExpanded: true,
              value: selected,
              items: [
                for (final a in snap.accounts)
                  DropdownMenuItem(
                    value: a.accountId,
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 1,
                          ),
                          decoration: BoxDecoration(
                            color: (a.isLive == true)
                                ? NeoethosTokens.sell.withValues(alpha: 0.25)
                                : NeoethosTokens.buy.withValues(alpha: 0.2),
                            borderRadius: BorderRadius.circular(3),
                          ),
                          child: Text(
                            (a.isLive == true) ? 'LIVE' : 'DEMO',
                            style: const TextStyle(
                              fontSize: 9,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            a.dropdownLabel,
                            style: const TextStyle(fontSize: 12),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
              onChanged: enabled
                  ? (v) {
                      if (v != null && v != selected) {
                        // Update the local field immediately, then
                        // promote the account server-side (F-333).
                        onPicked(v);
                        onUserPicked(v);
                      }
                    }
                  : null,
            ),
          ),
        );
      },
      loading: () => Stack(
        children: [
          fallback,
          const Positioned(
            right: 8,
            top: 16,
            child: SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
          ),
        ],
      ),
      error: (err, _) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          fallback,
          const SizedBox(height: 4),
          Text(
            l10n.settingsAccountPickerError('$err'),
            style: const TextStyle(
              fontSize: 10,
              color: NeoethosTokens.warning,
            ),
          ),
        ],
      ),
    );
  }
}

/// One radio-style row in the news-trading-mode picker.
class _NewsTradingModeRow extends StatelessWidget {
  final String id;
  final String label;
  final String description;
  final bool selected;
  final bool busy;
  final VoidCallback onPick;
  const _NewsTradingModeRow({
    required this.id,
    required this.label,
    required this.description,
    required this.selected,
    required this.busy,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: busy ? null : onPick,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 3),
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: selected
              ? NeoethosTokens.accent.withValues(alpha: 0.12)
              : NeoethosTokens.surfaceBg,
          border: Border.all(
            color: selected ? NeoethosTokens.accent : NeoethosTokens.border,
          ),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 14,
              height: 14,
              margin: const EdgeInsets.only(right: 10, top: 2),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color:
                      selected ? NeoethosTokens.accent : NeoethosTokens.border,
                  width: 2,
                ),
                color: selected ? NeoethosTokens.accent : Colors.transparent,
              ),
            ),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label,
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      color: selected
                          ? NeoethosTokens.accent
                          : NeoethosTokens.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    description,
                    style: const TextStyle(
                      fontSize: 10,
                      color: NeoethosTokens.textMuted,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 16),
        child: Text(
          AppLocalizations.of(context)!.settingsLoadingSettings,
          style: const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

/// **2026-05-25 — task #238**: live launcher for the
/// AdvancedSettings screen. Renders a compact card with a "Open
/// advanced editor" CTA. Clicking pushes the 2-pane knob editor
/// where the operator can apply presets + tweak any of the ~42
/// catalogued runtime knobs with inline help.
class _AdvancedKnobEditorCard extends StatelessWidget {
  const _AdvancedKnobEditorCard();

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return SectionCard(
      title: l10n.settingsAdvancedKnobsTitle,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            l10n.settingsAdvancedKnobsIntro,
            style: const TextStyle(
              fontSize: 12,
              color: NeoethosTokens.textMuted,
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              FilledButton.icon(
                onPressed: () {
                  Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => Scaffold(
                        appBar: AppBar(
                          title: Text(l10n.settingsAdvancedSettingsTitle),
                        ),
                        body: const AdvancedSettingsScreen(),
                      ),
                    ),
                  );
                },
                icon: const Icon(Icons.tune, size: 16),
                label: Text(l10n.settingsOpenAdvancedEditor),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// #193: surface the raw config.yaml contents so an operator can
/// inspect the 200+ knobs the typed `/settings` DTO can't enumerate.
/// The on-disk path is shown so the user knows which file to edit if
/// they want to change something the UI doesn't yet expose as a typed
/// control.
///
/// F-312 (2026-05-29): previously read-only. Now editable + Save —
/// closes the silent-drop hole where the typed knob save path
/// dropped any edits outside its 5-field allowlist. The Save button
/// pushes the entire YAML body to `POST /settings/raw`, which
/// schema-validates against the `Settings` struct before writing,
/// so a typo'd field surfaces as a 400 with an actionable error
/// message instead of waiting until the next discovery start.
class _AdvancedConfigCard extends ConsumerStatefulWidget {
  const _AdvancedConfigCard();

  @override
  ConsumerState<_AdvancedConfigCard> createState() =>
      _AdvancedConfigCardState();
}

class _AdvancedConfigCardState extends ConsumerState<_AdvancedConfigCard> {
  /// Last value the backend confirmed on disk. Compared against
  /// [_controller.text] to detect dirty edits.
  String? _yamlOnDisk;
  String? _path;
  String? _error;

  /// Structured remediation hint from the backend's 400 body — e.g.
  /// "Common causes: typo in a field name, wrong type, missing
  /// required section." Surfaced inline next to the error so the
  /// operator doesn't have to dig through logs.
  String? _errorHint;
  bool _open = false;
  bool _loading = false;
  bool _saving = false;

  /// The text-field controller backs the editable view. Initialised
  /// lazily on first load + kept across rebuilds so the user's edits
  /// survive a re-render (e.g. theme switch) without being clobbered.
  final TextEditingController _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  bool get _dirty => _yamlOnDisk != null && _controller.text != _yamlOnDisk;

  Future<void> _load() async {
    final l10n = AppLocalizations.of(context)!;
    setState(() {
      _loading = true;
      _error = null;
      _errorHint = null;
    });
    try {
      final r = await ref.read(backendClientProvider).fetchRawConfigYaml();
      if (!mounted) return;
      setState(() {
        _yamlOnDisk = (r['yaml'] as String?) ?? '';
        _path = (r['path'] as String?) ?? '';
        _controller.text = _yamlOnDisk!;
      });
    } catch (err) {
      if (mounted) {
        setState(() => _error = l10n.settingsYamlReadError(describeError(err)));
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _save() async {
    final l10n = AppLocalizations.of(context)!;
    if (_yamlOnDisk == null || !_dirty) return;
    setState(() {
      _saving = true;
      _error = null;
      _errorHint = null;
    });
    try {
      final r = await ref
          .read(backendClientProvider)
          .saveRawConfigYaml(_controller.text);
      if (!mounted) return;
      // Backend returns `{ok: true, ...}` on success or
      // `{error: "...", code: "...", hint: "..."}` on validation
      // failure (4xx still flows through `validateStatus < 500`).
      if (r['ok'] == true) {
        setState(() {
          _yamlOnDisk = _controller.text;
          _error = null;
          _errorHint = null;
        });
        if (mounted) {
          final bytes = (r['bytesWritten'] as num?)?.toInt() ?? 0;
          final backup = r['backupPath'] != null
              ? l10n.settingsYamlBackupAt('${r['backupPath']}')
              : '';
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(l10n.settingsYamlSaved(bytes, backup)),
              duration: const Duration(seconds: 4),
            ),
          );
        }
      } else {
        setState(() {
          _error = (r['error'] as String?) ?? l10n.settingsUnknownError;
          _errorHint = r['hint'] as String?;
        });
      }
    } catch (err) {
      if (mounted) {
        setState(
            () => _error = l10n.settingsYamlWriteError(describeError(err)));
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  void _revert() {
    if (_yamlOnDisk == null) return;
    setState(() {
      _controller.text = _yamlOnDisk!;
      _error = null;
      _errorHint = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: ExpansionTile(
        title: Row(
          children: [
            Text(
              l10n.settingsRawConfigTitle,
              style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
            ),
            if (_dirty) ...[
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: const Color(0xFFE65100),
                  borderRadius: BorderRadius.circular(3),
                ),
                child: Text(
                  l10n.settingsUnsavedBadge,
                  style: const TextStyle(
                    fontSize: 9,
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ],
        ),
        subtitle: Text(
          _path == null
              ? l10n.settingsRawConfigSubtitle
              : l10n.settingsRawConfigSource('$_path'),
          style: const TextStyle(
            fontSize: 11,
            color: NeoethosTokens.textMuted,
          ),
        ),
        initiallyExpanded: _open,
        onExpansionChanged: (v) {
          setState(() => _open = v);
          if (v && _yamlOnDisk == null && !_loading) _load();
        },
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // ── Action bar: Reload + Save + Revert ─────────────
                Row(
                  children: [
                    OutlinedButton.icon(
                      onPressed: (_loading || _saving) ? null : _load,
                      icon: _loading
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.refresh, size: 16),
                      label: Text(l10n.settingsReloadFromDisk),
                    ),
                    const SizedBox(width: 8),
                    FilledButton.icon(
                      onPressed: (_saving || !_dirty || _yamlOnDisk == null)
                          ? null
                          : _save,
                      icon: _saving
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Icon(Icons.save, size: 16),
                      label: Text(l10n.settingsSaveChanges),
                    ),
                    const SizedBox(width: 8),
                    if (_dirty)
                      OutlinedButton.icon(
                        onPressed: _saving ? null : _revert,
                        icon: const Icon(Icons.undo, size: 16),
                        label: Text(l10n.settingsRevert),
                      ),
                  ],
                ),
                const SizedBox(height: 8),
                // ── Inline error block (validation failures land here) ──
                if (_error != null) ...[
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0x33B71C1C),
                      border: Border.all(color: const Color(0xFFB71C1C)),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _error!,
                          style: const TextStyle(
                            fontSize: 11,
                            color: NeoethosTokens.sell,
                          ),
                        ),
                        if (_errorHint != null) ...[
                          const SizedBox(height: 4),
                          Text(
                            _errorHint!,
                            style: const TextStyle(
                              fontSize: 10,
                              fontStyle: FontStyle.italic,
                              color: NeoethosTokens.textMuted,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(height: 8),
                ],
                // ── The YAML editor itself ─────────────────────────
                if (_yamlOnDisk != null)
                  Container(
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: NeoethosTokens.surfaceBg,
                      border: Border.all(
                        color: _dirty
                            ? const Color(0xFFE65100)
                            : NeoethosTokens.border,
                      ),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: TextField(
                      controller: _controller,
                      minLines: 18,
                      maxLines: 40,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 11,
                        color: NeoethosTokens.textPrimary,
                      ),
                      decoration: const InputDecoration(
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.all(10),
                        isCollapsed: true,
                      ),
                      onChanged: (_) {
                        // Force a rebuild so the UNSAVED badge + Save/
                        // Revert button enablement reflect the dirty
                        // state on every keystroke. TextField itself
                        // doesn't trigger setState for content changes.
                        setState(() {});
                      },
                    ),
                  )
                else if (_loading)
                  Text(
                    l10n.settingsFetchingConfig,
                    style: const TextStyle(
                      fontSize: 11,
                      color: NeoethosTokens.textMuted,
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
