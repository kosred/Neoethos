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
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';

class StatusBar extends ConsumerWidget {
  const StatusBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final asyncAccount = ref.watch(accountSnapshotProvider);
    final brokerAsync = ref.watch(brokerStatusProvider);
    final enginesAsync = ref.watch(enginesProvider);

    // Broker label: prefer `/broker/status` (gives adapter + environment
    // + connected). If that hasn't loaded yet, fall back to inferring
    // from accountSnapshotProvider so the bar isn't empty on cold start.
    final (brokerValue, brokerOk) = brokerAsync.maybeWhen(
      data: (b) => (
        '${b.adapter} · ${b.environment}${b.connected ? "" : " · offline"}',
        b.connected,
      ),
      orElse: () => switch (asyncAccount) {
        AsyncData() => ('cTrader · connecting', false),
        AsyncError(error: final e) when e is BrokerNotReadyException =>
          ('cTrader · connecting', false),
        AsyncError() => ('cTrader · offline', false),
        _ => ('cTrader · connecting', false),
      },
    );

    final (engineValue, engineRunning) = enginesAsync.maybeWhen(
      data: (e) {
        bool running(String s) => s.toLowerCase() == 'running';
        if (running(e.discovery)) return ('Discovery · running', true);
        if (running(e.training)) return ('Training · running', true);
        if (running(e.autoTrader)) return ('Auto-trader · running', true);
        return ('Idle', false);
      },
      orElse: () => ('—', false),
    );

    return Container(
      height: ForexAiTokens.statusbarHeight,
      decoration: const BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border(top: BorderSide(color: ForexAiTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spMd),
      child: Row(
        children: [
          _StatusItem(label: 'Broker', value: brokerValue, success: brokerOk),
          const _StatusSep(),
          _StatusItem(
            label: 'Engine',
            value: engineValue,
            success: engineRunning,
          ),
          const _StatusSep(),
          const _StatusItem(label: 'News blackout', value: '—'),
          const Spacer(),
          const _StatusItem(label: 'v0.4.35'),
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
    final color = success ? ForexAiTokens.buy : ForexAiTokens.textPrimary;
    return Row(
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: ForexAiTokens.fsCaption,
            color: ForexAiTokens.textMuted,
          ),
        ),
        if (value != null) ...[
          const SizedBox(width: 6),
          Text(
            value!,
            style: TextStyle(
              fontSize: ForexAiTokens.fsCaption,
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
        color: ForexAiTokens.border,
        margin: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spMd),
      );
}
