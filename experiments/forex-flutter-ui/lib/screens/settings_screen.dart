// Settings — credentials form + read-only config view.
//
// The credentials form posts to /broker/credentials which writes
// broker_credentials.toml under %APPDATA%/neoethos/. After save, the
// operator goes to Broker Setup → Re-authenticate to do the actual
// OAuth flow against the freshly-saved client_id/secret.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
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
                  'news ingestion. Gemma chat does not read this.',
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
