import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:forex_flutter_ui/api/backend_client.dart';
import 'package:forex_flutter_ui/main.dart';
import 'package:forex_flutter_ui/startup/backend_watchdog.dart';
import 'package:forex_flutter_ui/state/account_provider.dart';
import 'package:forex_flutter_ui/state/live_spots_provider.dart';
import 'package:forex_flutter_ui/state/pending_actions_provider.dart';
import 'package:forex_flutter_ui/state/system_providers.dart';
import 'package:forex_flutter_ui/theme/theme.dart';
import 'package:forex_flutter_ui/widgets/app_shell.dart';

class TestAccountSnapshotNotifier extends AccountSnapshotNotifier {
  @override
  Future<AccountSnapshot> build() async => const AccountSnapshot(
        balance: 10000,
        equity: 10025,
        freeMargin: 9000,
        usedMargin: 1000,
        currency: 'USD',
        fetchedAtUnixMs: 1716422400000,
        positions: [
          Position(
            positionId: 1,
            volumeUnits: 10000,
            symbol: 'EURUSD',
            side: 'LONG',
            volume: 0.1,
            openTimestampMs: 1716422400000,
            pnlPips: 12.5,
            pnlUsd: 11.3,
          ),
        ],
      );
}

class TestLiveSpotsNotifier extends LiveSpotsNotifier {
  @override
  Future<LiveSpotsSnapshot> build() async => const LiveSpotsSnapshot(
        spots: [
          LiveSpotTick(
            symbolId: 1,
            symbolName: 'EURUSD',
            bid: 1.0814,
            ask: 1.0816,
            midPrice: 1.0815,
            receivedAtUnixMs: 1716422460000,
            brokerTimestampMs: 1716422460000,
            freshnessSeconds: 0,
          ),
        ],
        snapshotAtUnixMs: 1716422460000,
        symbolCount: 1,
      );
}

class TestPendingActionsNotifier extends PendingActionsNotifier {
  @override
  Future<List<PendingAction>> build() async => const [];

  @override
  Future<void> refreshNow() async {
    state = const AsyncData([]);
  }
}

class TestBackendWatchdog extends BackendWatchdog {
  @override
  BackendHealthState build() => BackendHealthState(
        status: BackendHealthStatus.online,
        lastSeenAt: DateTime.fromMillisecondsSinceEpoch(1716422460000),
        respawnAttempts: 0,
        consecutiveFailures: 0,
      );

  @override
  Future<void> manualRestart() async {}
}

const _engineSnapshot = EnginesSnapshot(
  discovery: 'idle',
  training: 'idle',
  autoTrader: 'idle',
  discoverySummary: '',
  trainingSummary: '',
);

const _brokerStatus = BrokerStatus(
  adapter: 'cTrader',
  environment: 'Demo',
  accountId: 'demo-1',
  connected: false,
  clientIdPrefix: 'demo',
);

const _symbols = BrokerSymbolsSnapshot(
  accountId: 1,
  environment: 'Demo',
  symbolCount: 2,
  symbols: [
    BrokerSymbol(
      symbolId: 1,
      symbolName: 'EURUSD',
      enabled: true,
      description: 'Euro / US Dollar',
    ),
    BrokerSymbol(
      symbolId: 2,
      symbolName: 'GBPUSD',
      enabled: true,
      description: 'British Pound / US Dollar',
    ),
  ],
  archivedSymbols: [],
);

const _chart = ChartSnapshot(
  symbol: 'EURUSD',
  timeframe: 'M1',
  availableTimeframes: ['M1', 'M5', 'H1'],
  candleCount: 2,
  candles: [
    ChartCandle(
      tsMs: 1716422400000,
      open: 1.0800,
      high: 1.0810,
      low: 1.0795,
      close: 1.0805,
      volume: 1000,
    ),
    ChartCandle(
      tsMs: 1716422460000,
      open: 1.0805,
      high: 1.0820,
      low: 1.0800,
      close: 1.0815,
      volume: 1200,
    ),
  ],
  priceMin: 1.0795,
  priceMax: 1.0820,
  latestClose: 1.0815,
  priceChangePct: 0.13,
  headline: 'EURUSD M1 test feed',
  source: 'disk-cache',
);

List<Override> testProviderOverrides() => [
      accountSnapshotProvider.overrideWith(TestAccountSnapshotNotifier.new),
      liveSpotsProvider.overrideWith(TestLiveSpotsNotifier.new),
      pendingActionsProvider.overrideWith(TestPendingActionsNotifier.new),
      backendHealthProvider.overrideWith(TestBackendWatchdog.new),
      hardwareProvider.overrideWith(
        (ref) async => const HardwareSnapshot(
          cpuModel: 'Test CPU',
          cpuCoresLogical: 8,
          cpuCoresPhysical: 4,
          cpuLoadAvg: 0.25,
          ramTotalMb: 32768,
          ramUsedMb: 8192,
          ramAvailableMb: 24576,
          gpuName: 'Test GPU',
          gpuAvailable: true,
        ),
      ),
      riskProvider.overrideWith(
        (ref) async => const RiskSnapshot(
          riskPerTrade: 0.5,
          minRiskPerTrade: 0.1,
          maxRiskPerTrade: 1.0,
          dailyDrawdownLimit: 5.0,
          totalDrawdownLimit: 10.0,
          maxLotSize: 1.0,
          requireStopLoss: true,
          preset: 'ftmo',
          presetDisplayName: 'FTMO',
          availablePresets: [],
          propFirmRulesEnabled: true,
        ),
      ),
      settingsProvider.overrideWith(
        (ref) async => const SettingsSnapshot(
          dataDir: 'test-data',
          newsCalendarEnabled: true,
          newsCalendarSource: 'test',
          newsTradingMode: 'block_on_news',
          newsTradingModeDisplayName: 'Pause during news (safe default)',
        ),
      ),
      enginesProvider.overrideWith((ref) async => _engineSnapshot),
      brokerStatusProvider.overrideWith((ref) async => _brokerStatus),
      intelligenceProvider.overrideWith(
        (ref) async => const IntelligenceSnapshot(
          modelsDir: 'test-models',
          modelsDirExists: true,
          artifactCount: 1,
          artifacts: ['model.json'],
          lastTouchedUnixMs: 1716422400000,
          discoveryTargets: [
            DiscoveryTarget(
              symbol: 'EURUSD',
              baseTf: 'M1',
              strategyId: 'test-strategy',
              sharpe: 1.2,
              winRate: 0.56,
            ),
          ],
          walkforwardSplits: 3,
          walkforwardAvgAccuracy: 0.58,
        ),
      ),
      brokerSymbolsProvider.overrideWith((ref) async => _symbols),
      brokerAccountsProvider.overrideWith(
        (ref) async => const BrokerAccountsSnapshot(
          environment: 'Demo',
          permissionScope: 'accounts',
          accountCount: 1,
          accounts: [
            BrokerAccount(
              accountId: 'demo-1',
              brokerTitle: 'Spotware',
              accountName: 'Demo',
              traderLogin: 123,
              isLive: false,
              enabledForExecution: true,
            ),
          ],
        ),
      ),
      codexStatusProvider.overrideWith(
        (ref) async => CodexStatusSnapshot(
          authenticated: false,
          email: null,
          loginInProgress: false,
          lastError: null,
          authPath: 'test-auth-path',
        ),
      ),
      brokerTimeframesProvider.overrideWith(
        (ref) async => const ['M1', 'M5', 'H1'],
      ),
      chartProvider.overrideWith((ref) async => _chart),
      dataBootstrapProvider.overrideWith(
        (ref) async => const DataBootstrapSnapshot(
          dataDir: 'test-data',
          dataDirExists: true,
          symbols: ['EURUSD', 'GBPUSD'],
          fileCount: 2,
          lastTouchedUnixMs: 1716422400000,
        ),
      ),
    ];

Widget appHarness() => ProviderScope(
      overrides: testProviderOverrides(),
      child: const NeoethosApp(),
    );

Widget shellHarness() => ProviderScope(
      overrides: testProviderOverrides(),
      child: MaterialApp(
        theme: buildNeoethosTheme(),
        home: const AppShell(),
      ),
    );

Future<void> useDesktopSurface(WidgetTester tester) async {
  await tester.binding.setSurfaceSize(const Size(1440, 900));
  addTearDown(() => tester.binding.setSurfaceSize(null));
}
