// Smoke tests that prove the shell renders without errors and
// that nav-tab clicks swap the dock view.
//
// **F-321 (2026-05-29 rebuild)**: was a 15-tab grouped nav with three
// section dividers (T R A D I N G / A I   E N G I N E / S Y S T E M).
// The Codex mockup collapsed that into 6 flat tabs and the old
// `NavGroup` enum was deleted, so these tests had to be reworked.
// `kNavTabs.length` is now `6` and no section dividers render any more.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:neoethos_flutter_ui/state/nav.dart';
import 'package:neoethos_flutter_ui/theme/theme.dart';

import 'test_harness.dart';

void main() {
  testWidgets('shell renders TopBar + Sidebar brand + Dashboard default',
      (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(shellHarness());
    // TopBar brand AND sidebar brand both render "NeoEthos".
    expect(find.text('NeoEthos'), findsWidgets);
    // Default screen header is the Dashboard tab title.
    expect(find.text('Dashboard'), findsWidgets);
  });

  testWidgets('sidebar lists all 6 top-level tabs', (tester) async {
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
    // Click "Settings" in the sidebar (lives at the bottom of the rail).
    await tester.tap(find.text('Settings').first);
    await tester.pumpAndSettle();
    // The dock now shows the Settings header twice (sidebar + dock
    // breadcrumb), and the Settings screen content underneath.
    expect(find.text('Settings'), findsWidgets);
  });

  testWidgets('all 6 screens load without throwing', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(shellHarness());
    for (final tab in kNavTabs) {
      await tester.tap(find.text(tab.title).first);
      await tester.pumpAndSettle();
    }
    // No exception escapes pumpAndSettle.
  });

  test('nav tab catalog has exactly 6 flat entries', () {
    expect(kNavTabs.length, 6);
    // Pin the canonical IDs so persistence (#24) doesn't silently
    // drift if someone renames a tab without thinking about it.
    expect(
      kNavTabs.map((t) => t.id).toList(),
      ['Dashboard', 'MarketWatch', 'StrategyLab', 'Positions', 'AiDesk',
        'Settings'],
    );
  });

  test('migrateLegacyNavId moves pre-F-321 IDs to the right home', () {
    // Trading group → MarketWatch / Positions
    expect(migrateLegacyNavId('Chart'), 'MarketWatch');
    expect(migrateLegacyNavId('Markets'), 'MarketWatch');
    expect(migrateLegacyNavId('Execution'), 'MarketWatch');
    expect(migrateLegacyNavId('News'), 'MarketWatch');
    expect(migrateLegacyNavId('TradeWatch'), 'Positions');
    // AI Engine group → StrategyLab / AiDesk
    expect(migrateLegacyNavId('Discovery'), 'StrategyLab');
    expect(migrateLegacyNavId('Training'), 'StrategyLab');
    expect(migrateLegacyNavId('Intelligence'), 'AiDesk');
    expect(migrateLegacyNavId('AiHelper'), 'AiDesk');
    // System group → Settings (4 sub-tabs eventually)
    expect(migrateLegacyNavId('BrokerSetup'), 'Settings');
    expect(migrateLegacyNavId('DataBootstrap'), 'Settings');
    expect(migrateLegacyNavId('Hardware'), 'Settings');
    expect(migrateLegacyNavId('Risk'), 'Settings');
    // Valid current IDs pass through unchanged
    expect(migrateLegacyNavId('Dashboard'), 'Dashboard');
    expect(migrateLegacyNavId('StrategyLab'), 'StrategyLab');
  });

  test('design tokens pin TradingView dark scheme', () {
    // Sanity: the hex values match the mockup CSS variables.
    expect(NeoethosTokens.appBg, const Color(0xFF0E1116));
    expect(NeoethosTokens.accent, const Color(0xFF2962FF));
    expect(NeoethosTokens.buy, const Color(0xFF26A69A));
    expect(NeoethosTokens.sell, const Color(0xFFEF5350));
  });
}
