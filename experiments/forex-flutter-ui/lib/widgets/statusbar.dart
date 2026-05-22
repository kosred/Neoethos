// StatusBar — bottom strip (broker / engine / blackout / version).
//
// Broker badge now reflects the live connection state from
// `accountSnapshotProvider`. Engine + blackout stay as static
// placeholders until their respective endpoints ship.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../theme/theme.dart';

class StatusBar extends ConsumerWidget {
  const StatusBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final asyncSnapshot = ref.watch(accountSnapshotProvider);
    final (brokerValue, brokerOk) = switch (asyncSnapshot) {
      AsyncData() => ('cTrader · Live', true),
      AsyncError(error: final e) when e is BrokerNotReadyException =>
        ('cTrader · connecting', false),
      AsyncError() => ('cTrader · offline', false),
      _ => ('cTrader · connecting', false),
    };

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
          // Engine / blackout / latency stay static for now — their
          // backing endpoints (/engines/status, /news/state) are
          // scheduled in the next session.
          const _StatusItem(label: 'Engine', value: 'Idle'),
          const _StatusSep(),
          const _StatusItem(label: 'News blackout', value: '—'),
          const Spacer(),
          const _StatusItem(label: 'v0.4.20'),
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
