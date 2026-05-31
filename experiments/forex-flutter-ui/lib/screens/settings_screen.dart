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
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
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
  static const _tabs = [
    ('Account', '🔐',
        'cTrader OAuth + saved client_id/secret'),
    ('App', '⚙',
        'Data dir, news source, LLM model, news-trading mode, raw YAML'),
    ('Risk', '⚠',
        'Prop-firm preset + drawdown caps + per-trade risk'),
    ('Advanced', '🛠',
        '42+ search-pipeline knobs — population, GA params, thresholds'),
    ('Hardware', '🖥',
        'CPU / GPU detection (read-only)'),
    ('Data', '📂',
        'Historical bootstrap + CSV/Parquet import'),
  ];

  @override
  void initState() {
    super.initState();
    _controller = TabController(length: _tabs.length, vsync: this);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _SettingsTabStrip(controller: _controller, tabs: _tabs),
        const SizedBox(height: ForexAiTokens.spSm),
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
    return Container(
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      padding: const EdgeInsets.only(right: 4),
      child: Row(
        children: [
          Expanded(
            child: TabBar(
              controller: controller,
              isScrollable: true,
              labelColor: ForexAiTokens.accent,
              unselectedLabelColor: ForexAiTokens.textMuted,
              indicatorColor: ForexAiTokens.accent,
              labelStyle: const TextStyle(
                fontSize: ForexAiTokens.fsBody,
                fontWeight: FontWeight.w700,
              ),
              unselectedLabelStyle: const TextStyle(
                fontSize: ForexAiTokens.fsBody,
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
                size: 16, color: ForexAiTokens.textMuted),
            label: const Text(
              'Help (F1)',
              style: TextStyle(
                fontSize: ForexAiTokens.fsCaption,
                fontWeight: FontWeight.w600,
                color: ForexAiTokens.textMuted,
              ),
            ),
            style: TextButton.styleFrom(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
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
        _environment = (r['environment'] as String?) == 'Live' ? 'Live' : 'Demo';
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
        const SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text(
            'Client ID and Client Secret are required (no saved value to fall back on)',
          ),
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
      final msg = (r['message'] as String?) ?? (ok ? 'Saved' : 'Unknown response');
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
          backgroundColor: ok ? ForexAiTokens.buy : ForexAiTokens.warning,
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
      showTranslatedErrorSnackbar(context, e, prefix: 'Save failed');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final asyncSettings = ref.watch(settingsProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Settings',
            subtitle: 'cTrader credentials · app configuration',
          ),
          _credentialsCard(),
          asyncSettings.when(
            data: (s) => _configCard(s),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
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

  Widget _credentialsCard() {
    if (!_loaded) {
      return const SectionCard(
        title: 'cTrader Credentials',
        child: Padding(
          padding: EdgeInsets.symmetric(vertical: 12),
          child: Text(
            'Loading…',
            style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
          ),
        ),
      );
    }
    return SectionCard(
      title: 'cTrader Credentials',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Get these from the Spotware Open API portal '
            '(https://openapi.ctrader.com → Applications → your app). '
            'They are saved to %APPDATA%/neoethos/broker_credentials.toml '
            '— never committed to git. After saving, open Broker Setup '
            '→ Re-authenticate to fetch a fresh OAuth token.',
            style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
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
                  ? 'Saved: $_secretMask (leave blank to keep)'
                  : 'Paste your secret here',
              helperText: _secretConfigured
                  ? 'A secret is already saved. Leave blank to keep it.'
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
              Expanded(child: _AccountPicker(
                currentAccountId: _accountIdCtrl.text,
                enabled: !_busy,
                onPicked: (id) => setState(() {
                  _accountIdCtrl.text = id;
                }),
                fallback: TextField(
                  controller: _accountIdCtrl,
                  enabled: !_busy,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    labelText: 'Account ID (cTID)',
                    isDense: true,
                    border: OutlineInputBorder(),
                    hintText: 'numeric, e.g. 5789955',
                    helperText:
                        'Will switch to a live dropdown once cTrader OAuth completes.',
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
                label: Text(_busy ? 'Saving…' : 'Save credentials'),
              ),
              if (_resultMessage != null) ...[
                const SizedBox(width: 12),
                Flexible(
                  child: Text(
                    _resultMessage!,
                    style: TextStyle(
                      fontSize: 11,
                      color: _resultOk
                          ? ForexAiTokens.buy
                          : ForexAiTokens.sell,
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
  late final TextEditingController _openaiModelCtrl;
  late bool _newsEnabled;
  /// Snake_case id matching `crate::config::NewsTradingMode`.
  /// Defaults to `block_on_news` (safe).
  late String _newsTradingMode;
  bool _busy = false;
  String? _message;
  bool _messageOk = false;

  @override
  void initState() {
    super.initState();
    final s = widget.snapshot;
    _dataDirCtrl = TextEditingController(text: s.dataDir);
    _newsSourceCtrl = TextEditingController(text: s.newsCalendarSource);
    _openaiModelCtrl = TextEditingController(text: s.openaiModel);
    _newsEnabled = s.newsCalendarEnabled;
    _newsTradingMode = s.newsTradingMode.isEmpty
        ? 'block_on_news'
        : s.newsTradingMode;
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
    if (!_busy && _openaiModelCtrl.text != s.openaiModel) {
      _openaiModelCtrl.text = s.openaiModel;
    }
  }

  @override
  void dispose() {
    _dataDirCtrl.dispose();
    _newsSourceCtrl.dispose();
    _openaiModelCtrl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final dataDir = _dataDirCtrl.text.trim();
    final newsSource = _newsSourceCtrl.text.trim();
    // openai_model is allowed blank intentionally — see backend
    // doc-comment in server/settings.rs::update_settings.
    if (dataDir.isEmpty) {
      _showSnack('Data directory cannot be blank', ok: false);
      return;
    }
    if (newsSource.isEmpty) {
      _showSnack('News calendar source cannot be blank', ok: false);
      return;
    }
    setState(() => _busy = true);
    try {
      await ref.read(backendClientProvider).saveSettings(
            dataDir: dataDir,
            newsCalendarEnabled: _newsEnabled,
            newsCalendarSource: newsSource,
            openaiModel: _openaiModelCtrl.text.trim(),
            newsTradingMode: _newsTradingMode,
          );
      if (!mounted) return;
      setState(() {
        _messageOk = true;
        _message = 'Saved · config.yaml updated';
      });
      // Refresh the snapshot so the parent screen and any other
      // consumers of settingsProvider see the new value.
      ref.invalidate(settingsProvider);
      _showSnack('Settings saved to config.yaml', ok: true);
    } on DioException catch (e) {
      if (!mounted) return;
      final msg = describeError(e);
      setState(() {
        _messageOk = false;
        _message = msg;
      });
      showTranslatedErrorSnackbar(context, e, prefix: 'Save failed');
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _messageOk = false;
        _message = e.toString();
      });
      _showSnack('Save failed: $e', ok: false);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _showSnack(String msg, {required bool ok}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: ok ? ForexAiTokens.buy : ForexAiTokens.sell,
        content: Text(msg),
        duration: const Duration(seconds: 4),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SectionCard(
      title: 'App Settings',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'These fields write directly into config.yaml. Unchanged '
            'lines (risk, models, the 200+ knobs the UI doesn\'t '
            'show) are preserved on every save.',
            style: TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _dataDirCtrl,
            enabled: !_busy,
            decoration: const InputDecoration(
              labelText: 'Data directory',
              isDense: true,
              border: OutlineInputBorder(),
              helperText:
                  'Where Vortex bars + discovery artifacts live. '
                  'Relative paths resolve against the binary\'s CWD.',
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _newsSourceCtrl,
                  enabled: !_busy,
                  decoration: const InputDecoration(
                    labelText: 'News calendar source',
                    isDense: true,
                    border: OutlineInputBorder(),
                    helperText: 'e.g. "forexfactory", "investing", "test".',
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
                      _newsEnabled ? 'Calendar ON' : 'Calendar OFF',
                      style: TextStyle(
                        fontSize: 11,
                        color: _newsEnabled
                            ? ForexAiTokens.buy
                            : ForexAiTokens.textFaint,
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
          const Text(
            'News-trading mode',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: ForexAiTokens.textPrimary,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'How the gate treats high-impact news. The kill window is '
            'set by `news_kill_window_min` in config.yaml (30 min '
            'default). The default is to pause new orders; flip to '
            'one of the others if your strategy is event-driven.',
            style: TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
          ),
          const SizedBox(height: 8),
          _NewsTradingModeRow(
            id: 'block_on_news',
            label: 'Pause during news (safe default)',
            description:
                'No new orders inside the kill window. Existing '
                'positions keep their SL/TP.',
            selected: _newsTradingMode == 'block_on_news',
            busy: _busy,
            onPick: () => setState(() => _newsTradingMode = 'block_on_news'),
          ),
          _NewsTradingModeRow(
            id: 'allow_always',
            label: 'Play through news (event-driven strategies)',
            description:
                'No automatic block. Use for breakout-on-news, '
                'fade-the-spike, or any strategy whose edge IS the '
                'news event. The UI still shows a banner while a '
                'high-impact print is imminent.',
            selected: _newsTradingMode == 'allow_always',
            busy: _busy,
            onPick: () => setState(() => _newsTradingMode = 'allow_always'),
          ),
          _NewsTradingModeRow(
            id: 'warn_only',
            label: 'Warn only — don\'t block',
            description:
                'Visual warning in the kill window but no order block. '
                'For operators who want a heads-up without the gate '
                'overriding their judgment.',
            selected: _newsTradingMode == 'warn_only',
            busy: _busy,
            onPick: () => setState(() => _newsTradingMode = 'warn_only'),
          ),
          const SizedBox(height: 14),
          TextField(
            controller: _openaiModelCtrl,
            enabled: !_busy,
            decoration: const InputDecoration(
              labelText: 'LLM model name (legacy "openai_model" field)',
              isDense: true,
              border: OutlineInputBorder(),
              helperText:
                  'Used by the news pipeline. Leave blank to disable LLM '
                  'news ingestion.',
            ),
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
                label: Text(_busy ? 'Saving…' : 'Save settings'),
              ),
              if (_message != null) ...[
                const SizedBox(width: 12),
                Flexible(
                  child: Text(
                    _message!,
                    style: TextStyle(
                      fontSize: 11,
                      color: _messageOk
                          ? ForexAiTokens.buy
                          : ForexAiTokens.sell,
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
  final ValueChanged<String> onPicked;
  final Widget fallback;
  const _AccountPicker({
    required this.currentAccountId,
    required this.enabled,
    required this.onPicked,
    required this.fallback,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(brokerAccountsProvider);
    return async.when(
      data: (snap) {
        if (snap.accounts.isEmpty) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              fallback,
              const SizedBox(height: 4),
              const Text(
                'Token granted access to 0 accounts. Open Broker Setup '
                '→ Re-authenticate and tick at least one account on '
                'the Spotware consent screen.',
                style: TextStyle(
                  fontSize: 10,
                  color: ForexAiTokens.warning,
                ),
              ),
            ],
          );
        }
        // If the saved account_id isn't in the granted set, fall back
        // to first available — that's typically what the user meant
        // anyway, and the consent screen has already accepted it.
        final ids = snap.accounts.map((a) => a.accountId).toList();
        final selected = ids.contains(currentAccountId)
            ? currentAccountId
            : ids.first;
        if (selected != currentAccountId) {
          // Post-frame nudge so we don't setState during build.
          WidgetsBinding.instance
              .addPostFrameCallback((_) => onPicked(selected));
        }
        return InputDecorator(
          decoration: InputDecoration(
            labelText:
                'Account · ${snap.accountCount} from /broker/accounts (live)',
            isDense: true,
            border: const OutlineInputBorder(),
            helperText:
                'Picked from the cTrader OAuth grant — no more typing cTIDs by hand.',
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
                                ? ForexAiTokens.sell.withValues(alpha: 0.25)
                                : ForexAiTokens.buy.withValues(alpha: 0.2),
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
                      if (v != null) onPicked(v);
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
            'Account picker unavailable: $err\n'
            'Save credentials, then Broker Setup → Re-authenticate '
            'before the dropdown can populate.',
            style: const TextStyle(
              fontSize: 10,
              color: ForexAiTokens.warning,
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
              ? ForexAiTokens.accent.withValues(alpha: 0.12)
              : ForexAiTokens.surfaceBg,
          border: Border.all(
            color: selected ? ForexAiTokens.accent : ForexAiTokens.border,
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
                  color: selected
                      ? ForexAiTokens.accent
                      : ForexAiTokens.border,
                  width: 2,
                ),
                color: selected ? ForexAiTokens.accent : Colors.transparent,
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
                          ? ForexAiTokens.accent
                          : ForexAiTokens.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    description,
                    style: const TextStyle(
                      fontSize: 10,
                      color: ForexAiTokens.textMuted,
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
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'Loading settings…',
          style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
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
    return SectionCard(
      title: 'Advanced runtime knobs',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Live editor for the ~42 catalogued knobs. Apply a preset '
            '(Conservative / Balanced / Aggressive), or tweak any '
            'individual knob with inline help.',
            style: TextStyle(
              fontSize: 12,
              color: ForexAiTokens.textMuted,
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
                          title: const Text('Advanced settings'),
                        ),
                        body: const AdvancedSettingsScreen(),
                      ),
                    ),
                  );
                },
                icon: const Icon(Icons.tune, size: 16),
                label: const Text('Open advanced editor'),
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
      if (mounted) setState(() => _error = err.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _save() async {
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
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'config.yaml saved ($bytes bytes). '
                '${r['backupPath'] != null ? "Backup at ${r['backupPath']}" : ""}',
              ),
              duration: const Duration(seconds: 4),
            ),
          );
        }
      } else {
        setState(() {
          _error = (r['error'] as String?) ?? 'unknown error';
          _errorHint = r['hint'] as String?;
        });
      }
    } catch (err) {
      if (mounted) setState(() => _error = err.toString());
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
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: ExpansionTile(
        title: Row(
          children: [
            const Text(
              'Advanced: full config.yaml',
              style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
            ),
            if (_dirty) ...[
              const SizedBox(width: 8),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: const Color(0xFFE65100),
                  borderRadius: BorderRadius.circular(3),
                ),
                child: const Text(
                  'UNSAVED',
                  style: TextStyle(
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
              ? 'Expand to edit every config knob. Save validates against '
                  'the typed Settings schema before writing — a typo here '
                  'surfaces as an error, not a silent failure later.'
              : 'Source: $_path',
          style: const TextStyle(
            fontSize: 11,
            color: ForexAiTokens.textMuted,
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
                              child: CircularProgressIndicator(
                                  strokeWidth: 2),
                            )
                          : const Icon(Icons.refresh, size: 16),
                      label: const Text('Reload from disk'),
                    ),
                    const SizedBox(width: 8),
                    FilledButton.icon(
                      onPressed:
                          (_saving || !_dirty || _yamlOnDisk == null)
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
                      label: const Text('Save changes'),
                    ),
                    const SizedBox(width: 8),
                    if (_dirty)
                      OutlinedButton.icon(
                        onPressed: _saving ? null : _revert,
                        icon: const Icon(Icons.undo, size: 16),
                        label: const Text('Revert'),
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
                            color: ForexAiTokens.sell,
                          ),
                        ),
                        if (_errorHint != null) ...[
                          const SizedBox(height: 4),
                          Text(
                            _errorHint!,
                            style: const TextStyle(
                              fontSize: 10,
                              fontStyle: FontStyle.italic,
                              color: ForexAiTokens.textMuted,
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
                      color: ForexAiTokens.surfaceBg,
                      border: Border.all(
                        color: _dirty
                            ? const Color(0xFFE65100)
                            : ForexAiTokens.border,
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
                        color: ForexAiTokens.textPrimary,
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
                  const Text(
                    'Fetching config.yaml…',
                    style: TextStyle(
                      fontSize: 11,
                      color: ForexAiTokens.textMuted,
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

class _Error extends StatelessWidget {
  final String error;
  const _Error({required this.error});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Text(
          'Backend unreachable: $error',
          style: const TextStyle(color: ForexAiTokens.sell, fontSize: 12),
        ),
      );
}

