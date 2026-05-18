import 'package:flutter/material.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const ViewHeader(
          title: 'Operator Overview',
          subtitle: 'Equity · open positions · engine status',
        ),
        // 4-column stat grid like the mockup
        const _StatRow(),
        const SectionCard(
          title: 'Open Positions',
          child: _PositionsTable(),
        ),
        const SectionCard(
          title: 'Engine Health',
          child: _EngineHealthRow(),
        ),
      ],
    );
  }
}

class _StatRow extends StatelessWidget {
  const _StatRow();
  @override
  Widget build(BuildContext context) {
    return const GridView.count(
      crossAxisCount: 4,
      crossAxisSpacing: 8,
      mainAxisSpacing: 8,
      childAspectRatio: 3.2,
      shrinkWrap: true,
      physics: NeverScrollableScrollPhysics(),
      children: [
        StatCard(label: 'Balance', value: '\$10,000.00'),
        StatCard(
          label: 'Equity',
          value: '\$10,243.55',
          valueColor: ForexAiTokens.buy,
        ),
        StatCard(label: 'Free Margin', value: '\$9,762.40'),
        StatCard(label: 'Open Positions', value: '2'),
      ],
    );
  }
}

class _PositionsTable extends StatelessWidget {
  const _PositionsTable();
  @override
  Widget build(BuildContext context) {
    final positions = [
      ('EURUSD', 'LONG', '0.10', '+24.5 pips', '+\$23.65'),
      ('XAUUSD', 'SHORT', '0.02', '-3.2 pips', '-\$6.40'),
    ];
    return Table(
      defaultVerticalAlignment: TableCellVerticalAlignment.middle,
      columnWidths: const {
        0: FlexColumnWidth(2),
        1: FlexColumnWidth(2),
        2: FlexColumnWidth(2),
        3: FlexColumnWidth(2),
        4: FlexColumnWidth(2),
      },
      children: [
        const TableRow(children: [
          _Th('Symbol'),
          _Th('Side'),
          _Th('Volume'),
          _Th('Pips'),
          _Th('PnL'),
        ]),
        for (final p in positions)
          TableRow(children: [
            _Td(p.$1),
            _Td(p.$2,
                color: p.$2 == 'LONG' ? ForexAiTokens.buy : ForexAiTokens.sell),
            _Td(p.$3),
            _Td(p.$4),
            _Td(p.$5,
                color: p.$5.startsWith('+')
                    ? ForexAiTokens.buy
                    : ForexAiTokens.sell),
          ]),
      ],
    );
  }
}

class _EngineHealthRow extends StatelessWidget {
  const _EngineHealthRow();
  @override
  Widget build(BuildContext context) {
    return const Row(
      children: [
        Expanded(child: StatCard(label: 'Discovery', value: 'Idle')),
        SizedBox(width: 8),
        Expanded(child: StatCard(label: 'Training', value: 'Idle')),
        SizedBox(width: 8),
        Expanded(child: StatCard(label: 'Autonomous Trader', value: 'Running',
            valueColor: ForexAiTokens.buy)),
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
            color: ForexAiTokens.textMuted,
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
            color: color ?? ForexAiTokens.textPrimary,
          ),
        ),
      );
}
