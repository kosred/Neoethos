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

    if (clientId.isEmpty || clientSecret.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Client ID and Client Secret are required'),
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
      final body = e.response?.data;
      final msg = (body is Map && body['error'] is String)
          ? body['error'] as String
          : e.message ?? e.toString();
      if (!mounted) return;
      setState(() {
        _resultOk = false;
        _resultMessage = msg;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Save failed: $msg'),
          duration: const Duration(seconds: 6),
        ),
      );
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
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _accountIdCtrl,
                  enabled: !_busy,
                  keyboardType: TextInputType.number,
                  decoration: const InputDecoration(
                    labelText: 'Account ID (cTID)',
                    isDense: true,
                    border: OutlineInputBorder(),
                    hintText: 'numeric, e.g. 46774385',
                  ),
                ),
              ),
              const SizedBox(width: 12),
              DropdownButton<String>(
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
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Data',
          child: _Row('Data directory', s.dataDir),
        ),
        SectionCard(
          title: 'News & Calendar',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row(
                'Calendar enabled',
                s.newsCalendarEnabled ? 'ON' : 'OFF',
                accent: s.newsCalendarEnabled
                    ? ForexAiTokens.buy
                    : ForexAiTokens.textFaint,
              ),
              _Row('Calendar source', s.newsCalendarSource),
            ],
          ),
        ),
        SectionCard(
          title: 'LLM',
          child: _Row('OpenAI model (legacy field)', s.openaiModel),
        ),
      ],
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  final Color? accent;
  const _Row(this.label, this.value, {this.accent});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          children: [
            SizedBox(
              width: 200,
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 12,
                  color: ForexAiTokens.textMuted,
                ),
              ),
            ),
            Expanded(
              child: Text(
                value,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: accent ?? ForexAiTokens.textPrimary,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      );
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
