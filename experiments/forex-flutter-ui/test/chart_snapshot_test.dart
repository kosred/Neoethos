import 'package:flutter_test/flutter_test.dart';

import 'package:neoethos_flutter_ui/api/backend_client.dart';

void main() {
  test('ChartSnapshot parses backend source provenance', () {
    final snapshot = ChartSnapshot.fromJson({
      'symbol': 'EURUSD',
      'timeframe': 'M1',
      'availableTimeframes': ['M1'],
      'candleCount': 0,
      'candles': [],
      'priceMin': 0.0,
      'priceMax': 0.0,
      'latestClose': 0.0,
      'priceChangePct': 0.0,
      'headline': 'No candles loaded',
      'source': 'disk-cache',
    });

    expect(snapshot.source, 'disk-cache');
    expect(snapshot.isDiskCache, isTrue);
    expect(snapshot.isBrokerSource, isFalse);
  });
}
