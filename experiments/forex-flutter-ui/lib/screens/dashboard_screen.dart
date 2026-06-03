import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../api/currency_format.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

/// Dashboard — live numbers from the Rust HTTP server.
///
/// The screen is a thin `ConsumerWidget` over
/// [accountSnapshotProvider]. Three render paths:
///   - loading (no data ever): all four stat cards show `—`
///   - data: real balance / equity / free margin / open positions
///   - error: banner above the stats explaining what's wrong;
///     last-known data still renders underneath so the operator
///     keeps situational awareness across a transient network blip.
class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final snapshot = ref.watch(accountSnapshotProvider);
    final brokerAsync = ref.watch(brokerStatusProvider);
    return SingleChildScrollView(
      child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // **2026-05-25 — task #241**: header row with the refresh
        // button on the right. The backend's `/account/snapshot/refresh`
        // skips the bridge's 5 s safety timer; the resulting fresh
        // snapshot arrives over the SSE within ~750 ms so the
        // operator sees the new state nearly instantly.
        Row(
          children: [
            Expanded(
              child: ViewHeader(
                title: l10n.dashboardTitle,
                subtitle: l10n.dashboardSubtitle,
              ),
            ),
            IconButton(
              tooltip: l10n.dashboardRefreshTooltip,
              icon: snapshot.isLoading
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: NeoethosTokens.textMuted,
                      ),
                    )
                  : const Icon(Icons.refresh,
                      color: NeoethosTokens.textMuted, size: 20),
              onPressed: snapshot.isLoading
                  ? null
                  : () => ref
                      .read(accountSnapshotProvider.notifier)
                      .refreshNow(),
            ),
          ],
        ),
        // F-328 (2026-05-29): account-context strip directly under the
        // header, so the operator always sees which account they're
        // looking at before reading any number underneath.
        _AccountContextStrip(
          accountAsync: snapshot,
          brokerAsync: brokerAsync,
        ),
        if (snapshot.hasError) _ErrorBanner(error: snapshot.error!),
        _StatRow(snapshot: snapshot),
        // (The old "Growth Mode" card was removed 2026-06-03: it was a
        // guess-fed projection surface and a duplicate of Risky Mode, which is
        // now a first-class Trading Mode set from the Risk screen — the goal
        // there pressures the discovery search directly.)
        SectionCard(
          title: l10n.openPositions,
          child: _PositionsTable(snapshot: snapshot),
        ),
        SectionCard(
          title: l10n.engineHealth,
          child: const _EngineHealthRow(),
        ),
      ],
      ),
    );
  }
}

class _StatRow extends StatelessWidget {
  final AsyncValue<AccountSnapshot> snapshot;
  const _StatRow({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    // Use `valueOrNull` so an error during a periodic refresh keeps
    // the previous numbers on screen — we only fall back to em-dash
    // placeholders if no data has ever loaded.
    final data = snapshot.valueOrNull;
    final isFirstLoad = snapshot.isLoading && data == null;

    String fmt(double v) {
      final f = NumberFormat.currency(
        symbol: currencyGlyph(data?.currency ?? 'EUR'),
        decimalDigits: 2,
      );
      return f.format(v);
    }

    final balance = data == null
        ? (isFirstLoad ? '…' : '—')
        : fmt(data.balance);
    final equity = data == null
        ? (isFirstLoad ? '…' : '—')
        : fmt(data.equity);
    final freeMargin = data == null
        ? (isFirstLoad ? '…' : '—')
        : fmt(data.freeMargin);
    final usedMargin = data == null
        ? (isFirstLoad ? '…' : '—')
        : fmt(data.usedMargin);
    // F-328: margin level (equity / used margin × 100 %) — the standard
    // forex broker health metric. Above 200 % = comfortable, 50-100 % =
    // margin-call zone, below 50 % = stop-out zone for most brokers.
    final marginLevel = data == null || data.usedMargin == 0
        ? (isFirstLoad ? '…' : '—')
        : '${(data.equity / data.usedMargin * 100).toStringAsFixed(0)}%';
    final marginLevelColor = data == null || data.usedMargin == 0
        ? null
        : data.equity / data.usedMargin >= 2.0
            ? NeoethosTokens.buy
            : data.equity / data.usedMargin >= 1.0
                ? NeoethosTokens.warning
                : NeoethosTokens.sell;
    final openCount = data == null
        ? (isFirstLoad ? '…' : '—')
        : '${data.positions.length}';

    // Color equity green when up vs balance, red when down — gives the
    // operator a one-glance read on session PnL without staring at the
    // raw number.
    final equityColor = data == null
        ? null
        : data.equity > data.balance
            ? NeoethosTokens.buy
            : data.equity < data.balance
                ? NeoethosTokens.sell
                : null;

    // Freshness badge — local time of the last successful refresh.
    // Mirrors what the server stamped into the snapshot; we localise
    // here so each user sees their wall-clock, not UTC. Hidden until
    // the first snapshot lands.
    final asOf = data?.fetchedAtUnixMs == null
        ? null
        : DateTime.fromMillisecondsSinceEpoch(data!.fetchedAtUnixMs!).toLocal();
    final asOfLabel = asOf == null
        ? null
        : l10n.dashboardAsOf(DateFormat('HH:mm:ss').format(asOf));

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        GridView.count(
          // F-328: 6 cards (Balance · Equity · Free Margin · Used Margin
          // · Margin Level · Open Positions). The first row is the cash
          // ladder; the second adds the broker-health margin level + the
          // open-positions counter so the operator gets the full account
          // picture before scrolling further.
          crossAxisCount: 3,
          crossAxisSpacing: 8,
          mainAxisSpacing: 8,
          childAspectRatio: 3.2,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          children: [
            StatCard(label: l10n.ribbonBalance, value: balance),
            StatCard(label: 'Equity', value: equity, valueColor: equityColor),
            StatCard(label: 'Free Margin', value: freeMargin),
            StatCard(label: 'Used Margin', value: usedMargin),
            StatCard(
              label: 'Margin Level',
              value: marginLevel,
              valueColor: marginLevelColor,
            ),
            StatCard(label: l10n.openPositions, value: openCount),
          ],
        ),
        if (asOfLabel != null) ...[
          const SizedBox(height: 4),
          Text(
            asOfLabel,
            style: const TextStyle(
              fontSize: 10,
              color: NeoethosTokens.textFaint,
            ),
          ),
        ],
      ],
    );
  }
}

class _PositionsTable extends StatelessWidget {
  final AsyncValue<AccountSnapshot> snapshot;
  const _PositionsTable({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final positions = snapshot.valueOrNull?.positions ?? const [];
    if (positions.isEmpty) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Text(
          l10n.dashboardNoPositions,
          style: const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
    }

    final pipFmt = NumberFormat('+#,##0.0;-#,##0.0', 'en_US');
    final usdFmt = NumberFormat.currency(symbol: r'$', decimalDigits: 2);

    // "Since" formatter — converts UTC openTimestampMs into local
    // wall-clock. Falls back to "—" when the broker didn't stamp the
    // fill yet (rare race). Short HH:mm format saves table real estate.
    final sinceFmt = DateFormat('HH:mm');
    String sinceLabel(int? ms) {
      if (ms == null) return '—';
      return sinceFmt.format(DateTime.fromMillisecondsSinceEpoch(ms).toLocal());
    }

    return Table(
      defaultVerticalAlignment: TableCellVerticalAlignment.middle,
      columnWidths: const {
        0: FlexColumnWidth(2),
        1: FlexColumnWidth(2),
        2: FlexColumnWidth(2),
        3: FlexColumnWidth(2),
        4: FlexColumnWidth(2),
        5: FlexColumnWidth(2),
      },
      children: [
        TableRow(children: [
          _Th(l10n.thSymbol),
          _Th(l10n.thSide),
          _Th(l10n.thVolume),
          _Th(l10n.thSince),
          const _Th('Pips'),
          const _Th('PnL'),
        ]),
        for (final p in positions)
          TableRow(children: [
            _Td(p.symbol),
            _Td(
              p.side,
              color: p.side.toUpperCase() == 'LONG' || p.side.toUpperCase() == 'BUY'
                  ? NeoethosTokens.buy
                  : NeoethosTokens.sell,
            ),
            _Td(p.volume.toStringAsFixed(2)),
            _Td(sinceLabel(p.openTimestampMs)),
            _Td('${pipFmt.format(p.pnlPips)} pips'),
            _Td(
              usdFmt.format(p.pnlUsd),
              color: p.pnlUsd >= 0 ? NeoethosTokens.buy : NeoethosTokens.sell,
            ),
          ]),
      ],
    );
  }
}

/// Compact account-context strip that sits under the page header.
/// Shows the account ID, environment (Demo / Live), broker adapter,
/// account currency, and a freshness "as of" timestamp. Every value
/// is broker-sourced; we never invent labels.
class _AccountContextStrip extends StatelessWidget {
  final AsyncValue<AccountSnapshot> accountAsync;
  final AsyncValue<BrokerStatus> brokerAsync;
  const _AccountContextStrip({
    required this.accountAsync,
    required this.brokerAsync,
  });

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final acc = accountAsync.valueOrNull;
    final broker = brokerAsync.valueOrNull;
    final asOf = acc?.fetchedAtUnixMs == null
        ? null
        : DateTime.fromMillisecondsSinceEpoch(acc!.fetchedAtUnixMs!)
            .toLocal();
    final asOfLabel = asOf == null
        ? l10n.dashboardNoSnapshot
        : l10n.dashboardSnapshotAt(DateFormat('HH:mm:ss').format(asOf));
    final environment = broker?.environment ?? '—';
    final envColor = environment.toLowerCase() == 'live'
        ? NeoethosTokens.sell
        : environment.toLowerCase() == 'demo'
            ? NeoethosTokens.accent
            : NeoethosTokens.textFaint;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Row(
        children: [
          _ContextChip(
            label: l10n.chipAccount,
            value: broker?.accountId ?? '—',
          ),
          const SizedBox(width: 12),
          _ContextChip(
            label: 'BROKER',
            value: broker?.adapter ?? '—',
          ),
          const SizedBox(width: 12),
          _ContextChip(
            label: l10n.chipEnvironment,
            value: environment.toUpperCase(),
            color: envColor,
          ),
          const SizedBox(width: 12),
          _ContextChip(
            label: l10n.chipCurrency,
            value: acc?.currency ?? '—',
          ),
          const Spacer(),
          Text(
            asOfLabel,
            style: const TextStyle(
              fontSize: 11,
              color: NeoethosTokens.textFaint,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}

class _ContextChip extends StatelessWidget {
  final String label;
  final String value;
  final Color? color;
  const _ContextChip({
    required this.label,
    required this.value,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.baseline,
      textBaseline: TextBaseline.alphabetic,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 10,
            letterSpacing: 0.8,
            fontWeight: FontWeight.w800,
            color: NeoethosTokens.textFaint,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          value,
          style: TextStyle(
            fontSize: NeoethosTokens.fsBody,
            fontWeight: FontWeight.w800,
            color: color ?? NeoethosTokens.textPrimary,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
      ],
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final Object error;
  const _ErrorBanner({required this.error});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final isBrokerNotReady = error is BrokerNotReadyException;
    final message = isBrokerNotReady
        ? l10n.dashboardConnecting
        : l10n.dashboardDataUnavailable(describeError(error));
    final colour = isBrokerNotReady
        ? NeoethosTokens.textMuted
        : NeoethosTokens.sell;

    return Container(
      margin: const EdgeInsets.only(top: 4, bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: colour.withValues(alpha: 0.08),
        border: Border.all(color: colour.withValues(alpha: 0.35)),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        children: [
          Icon(
            isBrokerNotReady ? Icons.hourglass_empty : Icons.error_outline,
            size: 16,
            color: colour,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: TextStyle(color: colour, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

class _EngineHealthRow extends ConsumerWidget {
  const _EngineHealthRow();
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(enginesProvider);
    return async.when(
      data: (e) => Row(
        children: [
          Expanded(
            child: StatCard(
              label: l10n.engineDiscovery,
              value: e.discovery,
              valueColor: _colorFor(e.discovery),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: StatCard(
              label: l10n.engineTraining,
              value: e.training,
              valueColor: _colorFor(e.training),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: StatCard(
              label: l10n.statAutonomousTrader,
              value: e.autoTrader,
              valueColor: _colorFor(e.autoTrader),
            ),
          ),
        ],
      ),
      loading: () => const _Skel(),
      error: (_, __) => const _Skel(),
    );
  }

  Color? _colorFor(String value) {
    switch (value.toLowerCase()) {
      case 'running':
        return NeoethosTokens.buy;
      case 'error':
      case 'failed':
        return NeoethosTokens.sell;
      case 'idle':
        return NeoethosTokens.textFaint;
      default:
        return NeoethosTokens.textMuted;
    }
  }
}

class _Skel extends StatelessWidget {
  const _Skel();
  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Row(
      children: [
        Expanded(child: StatCard(label: l10n.engineDiscovery, value: '—')),
        const SizedBox(width: 8),
        Expanded(child: StatCard(label: l10n.engineTraining, value: '—')),
        const SizedBox(width: 8),
        Expanded(child: StatCard(label: l10n.statAutonomousTrader, value: '—')),
      ],
    );
  }
}

class _Th extends StatelessWidget {
  final String text;
  const _Th(this.text);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Text(
          text.toUpperCase(),
          style: const TextStyle(
            fontSize: 10,
            letterSpacing: 0.4,
            color: NeoethosTokens.textMuted,
            fontWeight: FontWeight.w700,
          ),
        ),
      );
}

class _Td extends StatelessWidget {
  final String text;
  final Color? color;
  const _Td(this.text, {this.color});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Text(
          text,
          style: TextStyle(
            fontSize: 12,
            color: color ?? NeoethosTokens.textPrimary,
          ),
        ),
      );
}
