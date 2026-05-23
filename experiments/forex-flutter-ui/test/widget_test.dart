// Smoke test for the forex-ai Flutter front-end.
//
// Replaces the default `flutter create` counter-app test that referenced
// `MyApp` (which never existed in this scaffold) with a real boot test
// against `NeoethosApp` + `ProviderScope`. The actual UI-tree assertions
// live in `shell_smoke_test.dart`; this one just guarantees the entry
// point pumps a frame without panicking.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_harness.dart';

void main() {
  testWidgets('NeoethosApp boots without panicking',
      (WidgetTester tester) async {
    // The shell is designed for desktop viewport; pump the default
    // surface up before pumping the widget tree to keep the TopBar
    // Row from overflowing the 800x600 test default.
    await useDesktopSurface(tester);
    await tester.pumpWidget(appHarness());
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
