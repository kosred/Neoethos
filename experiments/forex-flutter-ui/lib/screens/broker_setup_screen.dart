import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../widgets/backend_error_widget.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class BrokerSetupScreen extends ConsumerWidget {
  const BrokerSetupScreen({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(brokerStatusProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(
            title: l10n.brokerSetupTitle,
            subtitle: l10n.brokerSetupSubtitle,
          ),
          async.when(
            data: (b) => _Body(status: b),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(error: err, title: l10n.brokerSetupUnreachable),
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
    final l10n = AppLocalizations.of(context)!;
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(l10n.brokerSetupReauthDialogTitle),
        content: Text(l10n.brokerSetupReauthDialogBody),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: Text(l10n.commonCancel),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: Text(l10n.brokerSetupOpenBrowser),
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
          backgroundColor: NeoethosTokens.buy,
          content: Text(
            (result['message'] as String?) ?? l10n.brokerSetupOauthRefreshComplete,
          ),
          duration: const Duration(seconds: 4),
        ),
      );
    } on DioException catch (e) {
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e, prefix: l10n.brokerSetupReauthFailed);
    } finally {
      if (mounted) setState(() => _reauthBusy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final status = widget.status;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: l10n.brokerSetupActiveSession,
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
                          ? NeoethosTokens.buy
                          : NeoethosTokens.sell,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    status.connected
                        ? l10n.brokerSetupConnected
                        : l10n.brokerSetupDisconnected,
                    style: TextStyle(
                      fontWeight: FontWeight.w700,
                      color: status.connected
                          ? NeoethosTokens.buy
                          : NeoethosTokens.sell,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              _Row(l10n.brokerSetupRowAdapter, status.adapter),
              _Row(l10n.brokerSetupRowEnvironment, status.environment),
              _Row('Account ID', status.accountId),
              _Row(l10n.brokerSetupRowClientIdPrefix, status.clientIdPrefix),
            ],
          ),
        ),
        SectionCard(
          title: l10n.brokerSetupReauthTitle,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                l10n.brokerSetupReauthDescription,
                style: const TextStyle(
                  color: NeoethosTokens.textMuted,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed: _reauthBusy ? null : _onReauth,
                    icon: const Icon(Icons.refresh, size: 18),
                    label: Text(l10n.brokerSetupReauthButton),
                  ),
                  if (_reauthBusy) ...[
                    const SizedBox(width: 16),
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      l10n.brokerSetupWaitingApproval,
                      style: const TextStyle(
                        color: NeoethosTokens.textMuted,
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
                  color: NeoethosTokens.textMuted,
                ),
              ),
            ),
            Expanded(
              child: Text(
                value,
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: NeoethosTokens.textPrimary,
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
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 16),
        child: Text(
          AppLocalizations.of(context)!.brokerSetupLoadingStatus,
          style: const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

