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
      padding: const EdgeInsets.only(bottom: ForexAiTokens.spSm),
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
                color: ForexAiTokens.textMuted,
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
      padding: const EdgeInsets.all(ForexAiTokens.spMd),
      constraints: const BoxConstraints(minHeight: 60),
      decoration: BoxDecoration(
        color: ForexAiTokens.surfaceBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rMd),
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
              color: ForexAiTokens.textMuted,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: valueColor ?? ForexAiTokens.textPrimary,
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
      padding: const EdgeInsets.all(ForexAiTokens.spMd),
      margin: const EdgeInsets.only(top: ForexAiTokens.spSm),
      decoration: BoxDecoration(
        color: ForexAiTokens.surfaceAlt,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: ForexAiTokens.textPrimary,
            ),
          ),
          const SizedBox(height: ForexAiTokens.spXs),
          child,
        ],
      ),
    );
  }
}

class PendingStub extends StatelessWidget {
  final String title;
  final String subtitle;
  const PendingStub({super.key, required this.title, required this.subtitle});
  @override
  Widget build(BuildContext context) {
    // Wrap in SingleChildScrollView so the placeholder copy + section
    // card fit cleanly in narrow / short test viewports (the default
    // 800x600 flutter_test surface trips a RenderFlex overflow
    // otherwise). The dock panel already supplies horizontal padding,
    // so we only need vertical scroll here.
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(title: title, subtitle: subtitle),
          const SectionCard(
            title: 'Pending — Flutter wiring',
            child: Text(
              'Αυτό το panel θα συνδεθεί στο Rust backend μέσω REST/SSE '
              'στην επόμενη φάση. Το shell (topbar, sidebar, statusbar, '
              'theme) λειτουργεί ήδη — εδώ θα έρθει το πραγματικό '
              'wiring με data από το neoethos backend + Gemma.',
              style: TextStyle(color: ForexAiTokens.textMuted),
            ),
          ),
        ],
      ),
    );
  }
}
