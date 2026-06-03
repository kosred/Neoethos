// StatusBar — bottom strip (broker / engine / blackout / version).
//
// All four items now reflect live state — no hardcoded "Live" lying
// about a Demo session, no "Idle" engine when a job is actually
// running. Sources:
//   - Broker: `/broker/status` (adapter + environment + connected flag)
//   - Engine: `/engines/status` (whichever of Discovery/Training/AutoTrader
//             is running takes the label; "Idle" only when all three are).
//   - Blackout: still "—" because the news-blackout endpoint is part of
//             the Gemma News work and the state we'd render here lives
//             behind a follow-up.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';

class StatusBar extends ConsumerWidget {
  const StatusBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final asyncAccount = ref.watch(accountSnapshotProvider);
    final brokerAsync = ref.watch(brokerStatusProvider);
    final enginesAsync = ref.watch(enginesProvider);

    // Broker label: prefer `/broker/status` (gives adapter + environment
    // + connected). If that hasn't loaded yet, fall back to inferring
    // from accountSnapshotProvider so the bar isn't empty on cold start.
    final (brokerValue, brokerOk) = brokerAsync.maybeWhen(
      data: (b) => (
        '${b.adapter} · ${b.environment}${b.connected ? "" : " · ${l10n.statusOffline}"}',
        b.connected,
      ),
      orElse: () => switch (asyncAccount) {
        AsyncData() => ('cTrader · ${l10n.statusConnecting}', false),
        AsyncError(error: final e) when e is BrokerNotReadyException =>
          ('cTrader · ${l10n.statusConnecting}', false),
        AsyncError() => ('cTrader · ${l10n.statusOffline}', false),
        _ => ('cTrader · ${l10n.statusConnecting}', false),
      },
    );

    final (engineValue, engineRunning) = enginesAsync.maybeWhen(
      data: (e) {
        bool running(String s) => s.toLowerCase() == 'running';
        if (running(e.discovery)) {
          return ('${l10n.engineDiscovery} · ${l10n.statusRunning}', true);
        }
        if (running(e.training)) {
          return ('${l10n.engineTraining} · ${l10n.statusRunning}', true);
        }
        if (running(e.autoTrader)) {
          return ('${l10n.engineAutoTrader} · ${l10n.statusRunning}', true);
        }
        return (l10n.statusIdle, false);
      },
      orElse: () => ('—', false),
    );

    return Container(
      height: NeoethosTokens.statusbarHeight,
      decoration: const BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border(top: BorderSide(color: NeoethosTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: NeoethosTokens.spMd),
      child: Row(
        children: [
          _StatusItem(label: 'Broker', value: brokerValue, success: brokerOk),
          const _StatusSep(),
          _StatusItem(
            label: l10n.statusEngine,
            value: engineValue,
            success: engineRunning,
          ),
          const _StatusSep(),
          _StatusItem(label: l10n.statusNewsBlackout, value: '—'),
          const Spacer(),
          const _StatusItem(label: 'v0.4.36'),
        ],
      ),
    );
  }
}

class _StatusItem extends StatelessWidget {
  final String label;
  final String? value;
  final bool success;
  const _StatusItem({required this.label, this.value, this.success = false});

  @override
  Widget build(BuildContext context) {
    final color = success ? NeoethosTokens.buy : NeoethosTokens.textPrimary;
    return Row(
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: NeoethosTokens.fsCaption,
            color: NeoethosTokens.textMuted,
          ),
        ),
        if (value != null) ...[
          const SizedBox(width: 6),
          Text(
            value!,
            style: TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ],
      ],
    );
  }
}

class _StatusSep extends StatelessWidget {
  const _StatusSep();
  @override
  Widget build(BuildContext context) => Container(
        width: 1,
        height: 12,
        color: NeoethosTokens.border,
        margin: const EdgeInsets.symmetric(horizontal: NeoethosTokens.spMd),
      );
}
