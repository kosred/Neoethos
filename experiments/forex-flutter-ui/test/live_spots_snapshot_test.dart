import 'package:flutter_test/flutter_test.dart';

import 'package:neoethos_flutter_ui/api/backend_client.dart';

void main() {
  test('mergeTick preserves the previous quote side on partial updates', () {
    const initial = LiveSpotsSnapshot(
      spots: [
        LiveSpotTick(
          symbolId: 1,
          symbolName: 'EURUSD',
          bid: 1.0850,
          ask: 1.0852,
          midPrice: 1.0851,
          receivedAtUnixMs: 1000,
          brokerTimestampMs: 900,
          freshnessSeconds: 0.1,
        ),
      ],
      snapshotAtUnixMs: 1000,
      symbolCount: 1,
    );

    const bidOnly = LiveSpotTick(
      symbolId: 1,
      symbolName: 'EURUSD',
      bid: 1.0860,
      ask: null,
      midPrice: null,
      receivedAtUnixMs: 1100,
      brokerTimestampMs: 1000,
      freshnessSeconds: 0.0,
    );

    final merged = initial.mergeTick(bidOnly);

    expect(merged.symbolCount, 1);
    expect(merged.snapshotAtUnixMs, 1100);
    expect(merged.spots.single.bid, 1.0860);
    expect(merged.spots.single.ask, 1.0852);
    expect(merged.spots.single.midPrice, closeTo(1.0856, 0.0000001));
    expect(merged.spots.single.brokerTimestampMs, 1000);
  });

  test('mergeTick appends a new symbol tick', () {
    const initial = LiveSpotsSnapshot(
      spots: [],
      snapshotAtUnixMs: 0,
      symbolCount: 0,
    );
    const tick = LiveSpotTick(
      symbolId: 2,
      symbolName: 'GBPUSD',
      bid: 1.2700,
      ask: 1.2702,
      midPrice: 1.2701,
      receivedAtUnixMs: 1200,
      brokerTimestampMs: null,
      freshnessSeconds: 0.0,
    );

    final merged = initial.mergeTick(tick);

    expect(merged.symbolCount, 1);
    expect(merged.spots.single.symbolName, 'GBPUSD');
    expect(merged.snapshotAtUnixMs, 1200);
  });
}
