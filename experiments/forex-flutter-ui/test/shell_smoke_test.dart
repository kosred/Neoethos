// Smoke tests that prove the shell renders without errors and
// that nav-tab clicks swap the dock view.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:forex_flutter_ui/state/nav.dart';
import 'package:forex_flutter_ui/theme/theme.dart';

import 'test_harness.dart';

void main() {
  testWidgets('shell renders all four grid areas', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(shellHarness());
    // TopBar brand
    expect(find.text('NeoEthos'), findsOneWidget);
    // Sidebar section headers (letter-spaced uppercase)
    expect(find.textContaining('T R A D I N G'), findsOneWidget);
    expect(find.textContaining('A I   E N G I N E'), findsOneWidget);
    expect(find.textContaining('S Y S T E M'), findsOneWidget);
    // Default screen is Dashboard
    expect(find.text('Operator Overview'), findsOneWidget);
  });

  testWidgets('sidebar lists all 15 panels', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(shellHarness());
    for (final tab in kNavTabs) {
      expect(
        find.text(tab.title),
        findsWidgets,
        reason: 'sidebar should list ${tab.title}',
      );
    }
  });

  testWidgets('clicking a sidebar item swaps the dock view', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(shellHarness());
    // Click "Broker Setup" in the sidebar.
    await tester.tap(find.text('Broker Setup').first);
    await tester.pumpAndSettle();
    // The dock should show the BrokerSetupScreen placeholder.
    expect(find.text('Broker Setup'), findsWidgets);
    expect(
      find.textContaining('cTrader / DXtrade'),
      findsOneWidget,
    );
  });

  testWidgets('all 15 screens load without throwing', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(shellHarness());
    for (final tab in kNavTabs) {
      await tester.tap(find.text(tab.title).first);
      await tester.pumpAndSettle();
    }
    // No exception escapes pumpAndSettle.
  });

  test('nav tab catalog has 15 entries grouped 6/4/5', () {
    final trading = kNavTabs.where((t) => t.group == NavGroup.trading).length;
    final ai = kNavTabs.where((t) => t.group == NavGroup.aiEngine).length;
    final system = kNavTabs.where((t) => t.group == NavGroup.system).length;
    expect(trading, 6);
    expect(ai, 4);
    expect(system, 5);
    expect(kNavTabs.length, 15);
  });

  test('design tokens pin TradingView dark scheme', () {
    // Sanity: the hex values match the mockup CSS variables.
    expect(ForexAiTokens.appBg, const Color(0xFF0E1116));
    expect(ForexAiTokens.accent, const Color(0xFF2962FF));
    expect(ForexAiTokens.buy, const Color(0xFF26A69A));
    expect(ForexAiTokens.sell, const Color(0xFFEF5350));
  });
}
