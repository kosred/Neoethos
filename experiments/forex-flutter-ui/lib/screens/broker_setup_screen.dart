import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class BrokerSetupScreen extends ConsumerWidget {
  const BrokerSetupScreen({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(brokerStatusProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Broker Setup',
            subtitle: 'cTrader / DXtrade · OAuth + account targets',
          ),
          async.when(
            data: (b) => _Body(status: b),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }
}

class _Body extends ConsumerStatefulWidget {
  final BrokerStatus status;
  const _Body({required this.status});
  @override
  ConsumerState<_Body> createState() => _BodyState();
}

class _BodyState extends ConsumerState<_Body> {
  bool _reauthBusy = false;

  Future<void> _onReauth() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Re-authenticate with cTrader?'),
        content: const Text(
          'A browser window will open on the Spotware consent screen. '
          'After you click "Continue", the new token is saved to the OS '
          'keyring and the existing session picks it up on the next '
          '5-second refresh — no restart needed.\n\n'
          'Typical wall-clock time: 10–30 seconds.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Open browser'),
          ),
        ],
      ),
    );
    if (confirm != true) return;
    if (!mounted) return;

    setState(() => _reauthBusy = true);
    try {
      final result = await ref.read(backendClientProvider).reauthBroker();
      if (!mounted) return;
      // Bump status + account so the new token is reflected.
      ref.invalidate(brokerStatusProvider);
      ref.invalidate(accountSnapshotProvider);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.buy,
          content: Text(
            (result['message'] as String?) ?? 'OAuth refresh complete',
          ),
          duration: const Duration(seconds: 4),
        ),
      );
    } on DioException catch (e) {
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e, prefix: 'Re-auth failed');
    } finally {
      if (mounted) setState(() => _reauthBusy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = widget.status;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Active Session',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: status.connected
                          ? ForexAiTokens.buy
                          : ForexAiTokens.sell,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    status.connected ? 'Connected' : 'Disconnected',
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: status.connected
                          ? ForexAiTokens.buy
                          : ForexAiTokens.sell,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              _Row('Adapter', status.adapter),
              _Row('Environment', status.environment),
              _Row('Account ID', status.accountId),
              _Row('Client ID prefix', status.clientIdPrefix),
            ],
          ),
        ),
        SectionCard(
          title: 'Re-authenticate',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Run the OAuth flow if the token expired or you see '
                'RET_ACCOUNT_DISABLED in the logs. Opens the Spotware '
                'consent screen in your default browser, captures the '
                'redirect on loopback, swaps the auth code for a fresh '
                'trading-scope token, and writes it to the OS keyring.',
                style: TextStyle(
                  color: ForexAiTokens.textMuted,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed: _reauthBusy ? null : _onReauth,
                    icon: const Icon(Icons.refresh, size: 18),
                    label: const Text('Re-authenticate'),
                  ),
                  if (_reauthBusy) ...[
                    const SizedBox(width: 16),
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    const SizedBox(width: 8),
                    const Text(
                      'Waiting for browser approval…',
                      style: TextStyle(
                        color: ForexAiTokens.textMuted,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  const _Row(this.label, this.value);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          children: [
            SizedBox(
              width: 160,
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
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: ForexAiTokens.textPrimary,
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
          'Loading broker status…',
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
