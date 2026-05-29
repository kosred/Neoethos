import 'package:flutter_test/flutter_test.dart';

import 'package:forex_flutter_ui/charts/chart_viewport.dart';

void main() {
  test('ChartViewport.live anchors the last visible bars', () {
    final viewport =
        ChartViewport.live(totalCount: 200, preferredVisibleCount: 80);

    expect(viewport.firstIndex, 120);
    expect(viewport.visibleCount, 80);
    expect(viewport.visibleEndExclusive, 200);
    expect(viewport.isAtLiveEnd, isTrue);
  });

  test('ChartViewport pans and clamps inside loaded history', () {
    final viewport =
        ChartViewport.live(totalCount: 200, preferredVisibleCount: 80);

    final older = viewport.pan(-30);
    expect(older.firstIndex, 90);
    expect(older.isAtLiveEnd, isFalse);

    expect(older.pan(-500).firstIndex, 0);
    expect(older.pan(500).firstIndex, 120);
  });

  test('ChartViewport zooms around the anchor and clamps visible count', () {
    final viewport = ChartViewport.live(
      totalCount: 200,
      preferredVisibleCount: 100,
      minVisibleCount: 20,
    );

    final zoomedIn = viewport.zoom(2.0, anchorFraction: 0.5);
    expect(zoomedIn.visibleCount, 50);
    expect(zoomedIn.firstIndex, 125);

    final zoomedOut = zoomedIn.zoom(0.25, anchorFraction: 0.5);
    expect(zoomedOut.visibleCount, 200);
    expect(zoomedOut.firstIndex, 0);
  });
}
