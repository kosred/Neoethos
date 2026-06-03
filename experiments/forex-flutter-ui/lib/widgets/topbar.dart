// TopBar — brand + connection badges + live ribbon items.
//
// Mirrors the .topbar block in mockups/ui_mockup.html, but every
// numeric / status value now reads from `accountSnapshotProvider`
// instead of hardcoded mockup figures.
//
// Render rules:
//   - Brand:            always "NeoEthos".
//   - LIVE / OFFLINE:   derives from `accountSnapshotProvider` state.
//                         data        → green LIVE
//                         BrokerNot…  → muted CONNECTING
//                         other error → red OFFLINE
//                         loading     → faint CONNECTING (first launch)
//   - Ribbon (Balance/Equity/Free Margin): em-dash when no data,
//     real numbers once the bridge has filled the cache, last-known
//     numbers preserved during transient refresh errors so the
//     operator doesn't lose situational awareness.
//   - Auto pills:       static for now (Discovery / Training engine
//                       state lives behind endpoints that ship in a
//                       follow-up session).

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../api/currency_format.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../screens/help_screen.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import 'backend_diagnostics_dialog.dart';
import 'report_issue.dart';

class TopBar extends ConsumerWidget {
  const TopBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final asyncSnapshot = ref.watch(accountSnapshotProvider);
    final brokerStatus = ref.watch(brokerStatusProvider);
    final engines = ref.watch(enginesProvider).valueOrNull;
    final discoveryOn = (engines?.discovery.toLowerCase() ?? 'idle') == 'running';
    final trainingOn = (engines?.training.toLowerCase() ?? 'idle') == 'running';

    // **2026-05-26 fix (Κωνσταντίνος)**: badge previously always said
    // "LIVE" whenever AccountSnapshot had data, regardless of whether
    // the broker session was Demo or Live. Now reads /broker/status's
    // `environment` field directly: shows "DEMO" for Demo accounts,
    // "LIVE" only for real-money accounts. Account-snapshot
    // availability still drives the CONNECTING/OFFLINE states for
    // back-compat with the no-broker-status case.
    final (badgeLabel, badgeKind) = switch (brokerStatus) {
      AsyncData(value: final b) when b.connected =>
        b.environment.toLowerCase() == 'live'
            ? ('LIVE', _BadgeKind.live)
            : ('DEMO', _BadgeKind.idle),
      AsyncData() => (l10n.badgeOffline, _BadgeKind.offline),
      AsyncError() => (l10n.badgeOffline, _BadgeKind.offline),
      _ => switch (asyncSnapshot) {
          AsyncData() => (l10n.badgeConnecting, _BadgeKind.idle),
          AsyncError(error: final e) when e is BrokerNotReadyException =>
            (l10n.badgeConnecting, _BadgeKind.idle),
          AsyncError() => (l10n.badgeOffline, _BadgeKind.offline),
          _ => (l10n.badgeConnecting, _BadgeKind.idle),
        },
    };

    final snap = asyncSnapshot.valueOrNull;
    final currencySymbol = currencyGlyph(snap?.currency ?? 'EUR');
    final fmt = NumberFormat.currency(symbol: currencySymbol, decimalDigits: 2);
    String ribbonValue(double? v) => v == null ? '—' : fmt.format(v);
    final equityAccent = snap == null
        ? _ValueAccent.plain
        : snap.equity > snap.balance
            ? _ValueAccent.success
            : snap.equity < snap.balance
                ? _ValueAccent.danger
                : _ValueAccent.plain;

    return Container(
      height: NeoethosTokens.topbarHeight,
      padding: const EdgeInsets.symmetric(horizontal: NeoethosTokens.spLg),
      decoration: const BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border(bottom: BorderSide(color: NeoethosTokens.border)),
      ),
      child: Row(
        children: [
          const Text(
            'NeoEthos',
            style: TextStyle(
              fontSize: NeoethosTokens.fsSubtitle + 1,
              fontWeight: FontWeight.w700,
              color: NeoethosTokens.textPrimary,
            ),
          ),
          const SizedBox(width: NeoethosTokens.spSm),
          const _Badge(label: 'PRO', kind: _BadgeKind.pro),
          const SizedBox(width: NeoethosTokens.spXs),
          _Badge(label: badgeLabel, kind: badgeKind),
          const SizedBox(width: NeoethosTokens.spXs),
          const _AccountSwitcher(),
          const _VSep(),
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _RibbonItem(
                    label: l10n.ribbonBalance,
                    value: ribbonValue(snap?.balance),
                  ),
                  _RibbonItem(
                    label: 'Equity',
                    value: ribbonValue(snap?.equity),
                    valueAccent: equityAccent,
                  ),
                  _RibbonItem(
                    label: 'Free Margin',
                    value: ribbonValue(snap?.freeMargin),
                  ),
                  _RibbonItem(
                    label: l10n.ribbonOpen,
                    value: snap == null ? '—' : '${snap.positions.length}',
                    valueAccent: snap != null && snap.positions.isNotEmpty
                        ? _ValueAccent.accent
                        : _ValueAccent.plain,
                  ),
                ],
              ),
            ),
          ),
          // Auto-Discover / Auto-Train pills are now backed by
          // `/engines/status`. Until the POST start/stop endpoints
          // land they're read-only mirrors of whatever the bridge is
          // doing, but they no longer lie about state.
          _AutoPill(label: l10n.autoDiscover, on: discoveryOn),
          const SizedBox(width: NeoethosTokens.spXs),
          _AutoPill(label: l10n.autoTrain, on: trainingOn),
          const SizedBox(width: NeoethosTokens.spSm),
          IconButton(
            onPressed: () => ref
                .read(accountSnapshotProvider.notifier)
                .refreshNow(),
            tooltip: l10n.tooltipRefreshSnapshot,
            icon: const Icon(Icons.refresh, color: NeoethosTokens.textMuted),
          ),
          // Help — F1 keyboard shortcut also opens this. Greek + English
          // docs across 6 sections (Welcome, Trading, AI Engine, Risk,
          // Shortcuts, FAQ). Lifted from the Codex mockup in F-329.
          IconButton(
            onPressed: () => showHelpDialog(context),
            tooltip: l10n.tooltipHelp,
            icon: const Icon(Icons.help_outline,
                color: NeoethosTokens.textMuted),
          ),
          // Report Issue — single, always-visible entry point. End
          // users can't rebuild the app, so this is the canonical way
          // to bundle today's logs + redacted config and email them
          // to NeoEthos support. Wired here so the button is reachable
          // from any screen, even when a panel further down is the one
          // that broke.
          IconButton(
            onPressed: () => showReportIssueDialog(context),
            tooltip: l10n.tooltipReportIssue,
            icon: const Icon(Icons.bug_report_outlined,
                color: NeoethosTokens.textMuted),
          ),
          // Backend diagnostics — supervisor.log tail + Restart button.
          // Always visible (not gated on degraded state) so the
          // operator can audit a healthy backend too.
          IconButton(
            onPressed: () => showBackendDiagnosticsDialog(context),
            tooltip: l10n.tooltipBackendDiagnostics,
            icon: const Icon(Icons.monitor_heart_outlined,
                color: NeoethosTokens.textMuted),
          ),
          // **2026-05-25 — task #241**: dead "Notifications (TODO)"
          // button removed. The Backend diagnostics icon above is the
          // canonical "something happened, see what" entry-point; the
          // PendingActionsBanner handles LLM-proposed trades; the
          // BackendHealthBanner handles connectivity. A separate
          // top-level notifications inbox would be 4th-tier UX clutter
          // until we have a real notification stream to populate it.
        ],
      ),
    );
  }
}

/// Accessible cTrader account switcher in the top bar (operator
/// feedback: the 7-account picker was buried under Settings → App). Lists
/// every account the OAuth grant exposes (DEMO/LIVE badged), marks the
/// active one, and switches on tap. The switch reorders
/// broker_credentials.toml server-side and takes effect on the next
/// backend start — the SnackBar says so.
class _AccountSwitcher extends ConsumerWidget {
  const _AccountSwitcher();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final accounts =
        ref.watch(brokerAccountsProvider).valueOrNull?.accounts ?? const [];
    final currentId =
        ref.watch(brokerStatusProvider).valueOrNull?.accountId ?? '';
    if (accounts.isEmpty) {
      // No OAuth grant loaded yet — nothing to switch between.
      return const SizedBox.shrink();
    }
    return PopupMenuButton<String>(
      tooltip: l10n.accountSwitcherTooltip,
      offset: const Offset(0, 36),
      color: NeoethosTokens.panelBg,
      onSelected: (id) => _select(context, ref, id),
      itemBuilder: (_) => [
        for (final a in accounts)
          PopupMenuItem<String>(
            value: a.accountId,
            height: 38,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _Badge(
                  label: (a.isLive ?? false) ? 'LIVE' : 'DEMO',
                  kind: (a.isLive ?? false)
                      ? _BadgeKind.live
                      : _BadgeKind.idle,
                ),
                const SizedBox(width: 8),
                Text(
                  a.accountId,
                  style: TextStyle(
                    fontSize: NeoethosTokens.fsBody,
                    fontWeight: a.accountId == currentId
                        ? FontWeight.w800
                        : FontWeight.w500,
                    color: NeoethosTokens.textPrimary,
                  ),
                ),
                if (a.accountId == currentId) ...[
                  const SizedBox(width: 8),
                  const Icon(Icons.check, size: 14, color: NeoethosTokens.buy),
                ],
              ],
            ),
          ),
      ],
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 3, horizontal: 8),
        decoration: BoxDecoration(
          color: NeoethosTokens.appBg,
          border: Border.all(color: NeoethosTokens.border),
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.account_circle_outlined,
                size: 14, color: NeoethosTokens.textMuted),
            const SizedBox(width: 5),
            Text(
              currentId.isEmpty ? l10n.accountLabel : currentId,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.textPrimary,
              ),
            ),
            const Icon(Icons.arrow_drop_down,
                size: 16, color: NeoethosTokens.textMuted),
          ],
        ),
      ),
    );
  }

  Future<void> _select(BuildContext context, WidgetRef ref, String id) async {
    final l10n = AppLocalizations.of(context)!;
    final messenger = ScaffoldMessenger.of(context);
    try {
      final r = await ref
          .read(backendClientProvider)
          .selectBrokerAccount(accountId: id);
      final ok = r['ok'] == true;
      ref.invalidate(brokerStatusProvider);
      ref.invalidate(accountSnapshotProvider);
      messenger.showSnackBar(SnackBar(
        backgroundColor: ok ? NeoethosTokens.buy : NeoethosTokens.warning,
        duration: const Duration(seconds: 5),
        content: Text(ok
            ? l10n.accountSwitchedRestart(id)
            : l10n.accountSwitchUnexpected),
      ));
    } catch (e) {
      messenger.showSnackBar(SnackBar(
        backgroundColor: NeoethosTokens.sell,
        duration: const Duration(seconds: 6),
        content: Text(l10n.accountSwitchFailed(describeError(e))),
      ));
    }
  }
}

enum _BadgeKind { pro, live, offline, local, blackout, idle }

class _Badge extends StatelessWidget {
  final String label;
  final _BadgeKind kind;
  const _Badge({required this.label, required this.kind});

  @override
  Widget build(BuildContext context) {
    final (fg, bg, border) = switch (kind) {
      _BadgeKind.pro => (
          NeoethosTokens.accent,
          const Color(0x29002962),
          NeoethosTokens.accent.withValues(alpha: 0.6),
        ),
      _BadgeKind.live => (
          NeoethosTokens.buy,
          NeoethosTokens.buy.withValues(alpha: 0.16),
          NeoethosTokens.buy.withValues(alpha: 0.6),
        ),
      _BadgeKind.offline => (
          NeoethosTokens.sell,
          NeoethosTokens.sell.withValues(alpha: 0.16),
          NeoethosTokens.sell.withValues(alpha: 0.6),
        ),
      _BadgeKind.local => (
          NeoethosTokens.warning,
          NeoethosTokens.warning.withValues(alpha: 0.16),
          NeoethosTokens.warning.withValues(alpha: 0.6),
        ),
      _BadgeKind.blackout => (
          NeoethosTokens.sell,
          NeoethosTokens.sell.withValues(alpha: 0.16),
          NeoethosTokens.sell.withValues(alpha: 0.6),
        ),
      _BadgeKind.idle => (
          NeoethosTokens.textFaint,
          NeoethosTokens.textFaint.withValues(alpha: 0.16),
          NeoethosTokens.textFaint.withValues(alpha: 0.6),
        ),
    };
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 2, horizontal: 8),
      decoration: BoxDecoration(
        color: bg,
        border: Border.all(color: border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: NeoethosTokens.fsCaption - 0.5,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.4,
          color: fg,
        ),
      ),
    );
  }
}

class _VSep extends StatelessWidget {
  const _VSep();
  @override
  Widget build(BuildContext context) => Container(
        width: 1,
        height: 28,
        color: NeoethosTokens.border,
        margin: const EdgeInsets.symmetric(horizontal: NeoethosTokens.spSm),
      );
}

enum _ValueAccent { plain, success, danger, accent }

class _RibbonItem extends StatelessWidget {
  final String label;
  final String value;
  final _ValueAccent valueAccent;
  const _RibbonItem({
    required this.label,
    required this.value,
    this.valueAccent = _ValueAccent.plain,
  });

  @override
  Widget build(BuildContext context) {
    final color = switch (valueAccent) {
      _ValueAccent.plain => NeoethosTokens.textPrimary,
      _ValueAccent.success => NeoethosTokens.buy,
      _ValueAccent.danger => NeoethosTokens.sell,
      _ValueAccent.accent => NeoethosTokens.accent,
    };
    return Padding(
      padding: const EdgeInsets.only(right: NeoethosTokens.spLg),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption - 1,
              letterSpacing: 0.8,
              fontWeight: FontWeight.w700,
              color: NeoethosTokens.textMuted,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: NeoethosTokens.fsBody,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

class _AutoPill extends StatelessWidget {
  final String label;
  final bool on;
  const _AutoPill({required this.label, required this.on});

  @override
  Widget build(BuildContext context) {
    final fg = on ? NeoethosTokens.buy : NeoethosTokens.textFaint;
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 3, horizontal: 8),
      decoration: BoxDecoration(
        color: fg.withValues(alpha: 0.15),
        border: Border.all(color: fg.withValues(alpha: 0.55)),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: NeoethosTokens.fsCaption,
          fontWeight: FontWeight.w700,
          color: fg,
        ),
      ),
    );
  }
}
