// Shared building blocks for placeholder screens. Each per-panel
// screen file uses these helpers so the look stays consistent
// while the real backend wiring lands later.

import 'package:flutter/material.dart';
import '../theme/theme.dart';

class ViewHeader extends StatelessWidget {
  final String title;
  final String? subtitle;
  const ViewHeader({super.key, required this.title, this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: NeoethosTokens.spSm),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          Text(title, style: Theme.of(context).textTheme.titleLarge),
          if (subtitle != null) ...[
            const SizedBox(width: 8),
            Text(
              subtitle!,
              style: const TextStyle(
                fontSize: 12,
                color: NeoethosTokens.textMuted,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class StatCard extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;
  const StatCard({
    super.key,
    required this.label,
    required this.value,
    this.valueColor,
  });
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(NeoethosTokens.spMd),
      constraints: const BoxConstraints(minHeight: 60),
      decoration: BoxDecoration(
        color: NeoethosTokens.surfaceBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rMd),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontSize: 10,
              letterSpacing: 0.4,
              color: NeoethosTokens.textMuted,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: valueColor ?? NeoethosTokens.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}

class SectionCard extends StatelessWidget {
  final String title;
  final Widget child;
  const SectionCard({super.key, required this.title, required this.child});
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(NeoethosTokens.spMd),
      margin: const EdgeInsets.only(top: NeoethosTokens.spSm),
      decoration: BoxDecoration(
        color: NeoethosTokens.surfaceAlt,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: NeoethosTokens.textPrimary,
            ),
          ),
          const SizedBox(height: NeoethosTokens.spXs),
          child,
        ],
      ),
    );
  }
}

