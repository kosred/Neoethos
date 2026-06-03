// #167: per-screen smoke tests. The test harness in `test_harness.dart`
// overrides every Riverpod provider with deterministic fixtures so each
// screen can be pumped in isolation and we can assert it renders a
// recognisable widget without crashing.
//
// The point here isn't golden-image coverage — that needs a screenshot
// pipeline we don't have. The point is that an accidental breaking
// change to a provider shape or widget contract surfaces as a failing
// test on the PR, instead of as "all buttons fail" at runtime (the
// failure mode that motivated the whole stability sweep).
//
// One smoke test per screen + a minimal assertion that something
// screen-specific renders. Add more focused tests as bugs surface.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:neoethos_flutter_ui/l10n/app_localizations.dart';
import 'package:neoethos_flutter_ui/screens/chart_screen.dart';
import 'package:neoethos_flutter_ui/screens/dashboard_screen.dart';
import 'package:neoethos_flutter_ui/screens/data_bootstrap_screen.dart';
import 'package:neoethos_flutter_ui/screens/discovery_screen.dart';
import 'package:neoethos_flutter_ui/screens/markets_screen.dart';
import 'package:neoethos_flutter_ui/screens/settings_screen.dart';
import 'package:neoethos_flutter_ui/theme/theme.dart';

import 'test_harness.dart';

Widget _wrap(Widget screen) => ProviderScope(
      overrides: testProviderOverrides(),
      child: MaterialApp(
        theme: buildNeoethosTheme(),
        // i18n (2026-06-03): screens read AppLocalizations.of(context)!, so the
        // wrapper must supply the localization delegates (else every screen
        // build null-checks to death). Pin `en` for deterministic assertions.
        locale: const Locale('en'),
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: Scaffold(body: screen),
      ),
    );

void main() {
  testWidgets('DashboardScreen renders without crashing',
      (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(_wrap(const DashboardScreen()));
    await tester.pumpAndSettle(const Duration(seconds: 1));
    expect(find.byType(DashboardScreen), findsOneWidget);
  });

  testWidgets('ChartScreen renders without crashing', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(_wrap(const ChartScreen()));
    await tester.pumpAndSettle(const Duration(seconds: 1));
    expect(find.byType(ChartScreen), findsOneWidget);
    // #198 invariant: the contextual-AI FAB is mounted.
    expect(find.text('Ask AI'), findsOneWidget);
  });

  testWidgets('MarketsScreen renders without crashing', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(_wrap(const MarketsScreen()));
    await tester.pumpAndSettle(const Duration(seconds: 1));
    expect(find.byType(MarketsScreen), findsOneWidget);
    // #185 invariant: the Forex-only filter chip is present.
    expect(find.text('Forex only'), findsOneWidget);
  });

  testWidgets('SettingsScreen renders the consolidated tab strip',
      (tester) async {
    // **F-327 invariant**: SettingsScreen is now a TabBar wrapper with
    // 6 sub-tabs (Account, App, Risk, Advanced, Hardware, Data) + a
    // Help link in the top-right. Pinning the labels here so the tab
    // strip doesn't silently lose entries.
    await useDesktopSurface(tester);
    await tester.pumpWidget(_wrap(const SettingsScreen()));
    await tester.pumpAndSettle(const Duration(seconds: 1));
    expect(find.byType(SettingsScreen), findsOneWidget);
    for (final label in const [
      'Account',
      'App',
      'Risk',
      'Advanced',
      'Hardware',
      'Data',
    ]) {
      expect(
        find.textContaining(label),
        findsWidgets,
        reason: 'Settings tab strip should list "$label"',
      );
    }
    // The Help (F1) jump-out link sits in the top-right corner.
    expect(find.textContaining('Help (F1)'), findsOneWidget);
  });

  testWidgets('AppSettingsScreen keeps the raw YAML editor (#193)',
      (tester) async {
    // **F-327 invariant**: the original SettingsScreen content lives
    // on as AppSettingsScreen and renders inside the consolidated
    // "App" tab. The #193 raw-YAML editor must still mount here.
    await useDesktopSurface(tester);
    await tester.pumpWidget(_wrap(const AppSettingsScreen()));
    await tester.pumpAndSettle(const Duration(seconds: 1));
    expect(find.byType(AppSettingsScreen), findsOneWidget);
    expect(
      find.textContaining('Advanced: full config.yaml'),
      findsOneWidget,
    );
  });

  testWidgets('DataBootstrapScreen renders without crashing',
      (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(_wrap(const DataBootstrapScreen()));
    await tester.pumpAndSettle(const Duration(seconds: 1));
    expect(find.byType(DataBootstrapScreen), findsOneWidget);
    // #192 invariant: the local-file import section exists.
    expect(
      find.textContaining('Import a local OHLCV file'),
      findsOneWidget,
    );
  });

  testWidgets('DiscoveryScreen renders without crashing', (tester) async {
    await useDesktopSurface(tester);
    await tester.pumpWidget(_wrap(const DiscoveryScreen()));
    await tester.pumpAndSettle(const Duration(seconds: 1));
    expect(find.byType(DiscoveryScreen), findsOneWidget);
    // #194 invariant: the GA hyperparams expander is mounted.
    expect(
      find.textContaining('Advanced: GA hyperparameters'),
      findsOneWidget,
    );
  });
}
