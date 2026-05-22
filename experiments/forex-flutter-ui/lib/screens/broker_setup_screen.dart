import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
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

class _Body extends StatelessWidget {
  final BrokerStatus status;
  const _Body({required this.status});
  @override
  Widget build(BuildContext context) {
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
        const SectionCard(
          title: 'Re-authenticate',
          child: Text(
            'To refresh the OAuth token (e.g. after RET_ACCOUNT_DISABLED '
            'or a 24h token expiry), close this app and run:\n\n'
            '    neoethos-app --reauth\n\n'
            'That opens the Spotware consent screen in your default '
            'browser, captures the redirect on loopback, swaps the '
            'auth code for a fresh trading-scope token, and writes it '
            'to the OS keyring. The button equivalent ships when the '
            'POST /broker/reauth control endpoint lands.',
            style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
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
