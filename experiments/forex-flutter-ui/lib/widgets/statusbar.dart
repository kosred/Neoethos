// StatusBar — bottom strip (broker / engine / blackout / latency).
// Mirrors the .statusbar block in mockups/ui_mockup.html.

import 'package:flutter/material.dart';

import '../theme/theme.dart';

class StatusBar extends StatelessWidget {
  const StatusBar({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: ForexAiTokens.statusbarHeight,
      decoration: const BoxDecoration(
        // Same fix as TopBar: Container.color + BoxDecoration is an
        // assert in Flutter 3.44+. Fold the bg into the decoration.
        color: ForexAiTokens.panelBg,
        border: Border(top: BorderSide(color: ForexAiTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spMd),
      child: Row(
        children: const [
          _StatusItem(label: 'Broker', value: 'cTrader · Live', success: true),
          _StatusSep(),
          _StatusItem(label: 'Engine', value: 'Running', success: true),
          _StatusSep(),
          _StatusItem(label: 'News blackout', value: 'CLEAR', success: true),
          _StatusSep(),
          _StatusItem(label: 'Latency', value: '83 ms'),
          Spacer(),
          _StatusItem(label: 'v0.4.5'),
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
  Widget build(BuildContext context) {
    return Container(
      width: 1,
      height: 12,
      color: ForexAiTokens.border,
      margin: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spMd),
    );
  }
}
