// Unit tests for BackendSupervisor.mergeMissingScalarFields — the
// additive config schema-upgrade merger that runs on every launch
// (_seedUserDataDir Phase B). It walks the bundle config.yaml and
// splices any NEW inline-scalar fields into the operator's existing
// config.yaml without disturbing what they already have.
//
// This is the real coverage that the (now-deleted) test/merger_smoke.dart
// scratch script falsely claimed already existed.

import 'package:flutter_test/flutter_test.dart';

import 'package:neoethos_flutter_ui/startup/backend_supervisor.dart';

void main() {
  group('BackendSupervisor.mergeMissingScalarFields', () {
    test('returns null when the user already has every bundle scalar', () {
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  symbol: EURUSD\n  data_dir: data\n',
        bundleYaml: 'system:\n  symbol: EURUSD\n  data_dir: data\n',
      );
      expect(result, isNull);
    });

    test('adds a missing inline scalar inside its section', () {
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  symbol: EURUSD\nrisk:\n  preset: FTMO\n',
        bundleYaml: 'system:\n  symbol: EURUSD\n  account_currency: USD\n'
            'risk:\n  preset: FTMO\n',
      );
      expect(result, isNotNull);
      expect(result!.added, ['system.account_currency']);
      expect(result.yaml, contains('account_currency: USD'));
      // Spliced into `system`, BEFORE the next top-level `risk:` section.
      final lines = result.yaml.split('\n');
      final fieldIdx = lines.indexWhere((l) => l.contains('account_currency'));
      final riskIdx = lines.indexWhere((l) => l.trim() == 'risk:');
      expect(fieldIdx, greaterThanOrEqualTo(0));
      expect(riskIdx, greaterThan(fieldIdx),
          reason: 'new field must land inside system, not after risk:');
    });

    test('does not touch a scalar the user already declares (even if the '
        'value differs)', () {
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  symbol: EURUSD\n  data_dir: mydata\n',
        bundleYaml: 'system:\n  symbol: EURUSD\n  data_dir: data\n',
      );
      expect(result, isNull,
          reason: 'the user keeps their own value; nothing is missing');
    });

    test('carries the bundle doc-comment along with the added field', () {
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  a: 1\n',
        bundleYaml: 'system:\n  a: 1\n  # explains b\n  b: 2\n',
      );
      expect(result, isNotNull);
      expect(result!.added, ['system.b']);
      final lines = result.yaml.split('\n');
      final commentIdx = lines.indexWhere((l) => l.contains('# explains b'));
      final bIdx = lines.indexWhere((l) => l.contains('b: 2'));
      expect(commentIdx, greaterThanOrEqualTo(0));
      expect(bIdx, commentIdx + 1,
          reason: 'the doc comment must immediately precede its field');
    });

    test('skips a block-format bundle key with no inline value (F-310 guard)',
        () {
      // `symbols:` introduced as a block (list on following lines). Emitting
      // a bare `symbols:` header would corrupt the YAML, so it is skipped.
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  symbol: EURUSD\n',
        bundleYaml: 'system:\n  symbol: EURUSD\n'
            '  symbols:\n    - EURUSD\n    - GBPUSD\n',
      );
      expect(result, isNull,
          reason: 'block-format keys are not inline scalars; nothing to add');
    });

    test('does not duplicate a block-format user key with an inline bundle '
        'key (F-310 regression)', () {
      // User has `symbols:` as a block; bundle has it inline plus a new
      // scalar. Only the new scalar is added; `symbols:` stays single.
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  symbols:\n    - EURUSD\n',
        bundleYaml: 'system:\n  symbols: [EURUSD]\n  base_timeframe: M1\n',
      );
      expect(result, isNotNull);
      expect(result!.added, ['system.base_timeframe']);
      final symbolHeaders =
          '\n${result.yaml}'.split(RegExp(r'\n\s*symbols:')).length - 1;
      expect(symbolHeaders, 1,
          reason: 'must not add a duplicate inline symbols: line');
    });

    test('appends a brand-new section that exists only in the bundle', () {
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  a: 1\n',
        bundleYaml: 'system:\n  a: 1\ntelemetry:\n  enabled: true\n',
      );
      expect(result, isNotNull);
      expect(result!.added, ['telemetry.enabled']);
      expect(result.yaml, contains('telemetry:'));
      expect(result.yaml, contains('enabled: true'));
    });

    test('collects multiple missing fields across sections', () {
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: 'system:\n  a: 1\nrisk:\n  preset: FTMO\n',
        bundleYaml: 'system:\n  a: 1\n  b: 2\nrisk:\n  preset: FTMO\n'
            '  max_lots: 10\n',
      );
      expect(result, isNotNull);
      expect(result!.added, containsAll(['system.b', 'risk.max_lots']));
      expect(result.added.length, 2);
    });

    test('does NOT splice a 4-space nested field after a list — the bug that '
        'crash-looped the backend (mapping-values-not-allowed)', () {
      // Repro of the production break: the bundle has a nested block
      // `discovery_runtime:` whose child `prefilter_min_per_timeframe` sits at
      // 4-space indent, and the section also ends with a YAML list
      // (`phase5_core_models:` / `- kan`). The old merger mis-attributed the
      // 4-space field to the top-level `models` section and appended it
      // verbatim after the list → `    prefilter_min_per_timeframe: 6` landing
      // under `- kan` → invalid YAML → backend exit(1) → supervisor restart
      // spiral. The field has a backend serde default, so the correct
      // behaviour is to leave it alone (no splice).
      final userYaml = 'models:\n'
          '  phase5_core_models:\n'
          '  - transformer\n'
          '  - kan\n';
      final bundleYaml = 'models:\n'
          '  phase5_core_models:\n'
          '  - transformer\n'
          '  - kan\n'
          '  discovery_runtime:\n'
          '    prefilter_top_k: 50\n'
          '    prefilter_min_per_timeframe: 6\n';
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: userYaml,
        bundleYaml: bundleYaml,
      );
      // Nothing splice-safe (the only "missing" bundle keys are 4-space
      // children of discovery_runtime, which the merger must skip).
      expect(result, isNull,
          reason: 'nested-block fields must never be spliced at section level');
    });

    test('still adds a genuine 2-space field even when the section also '
        'contains a nested block + a list', () {
      // Guard the inverse: a real top-level-section scalar IS still added,
      // and it lands correctly (2-space), not corrupting the list above it.
      final userYaml = 'models:\n'
          '  phase5_core_models:\n'
          '  - kan\n';
      final bundleYaml = 'models:\n'
          '  phase5_core_models:\n'
          '  - kan\n'
          '  ml_threshold: 0.5\n'
          '  discovery_runtime:\n'
          '    prefilter_top_k: 50\n';
      final result = BackendSupervisor.mergeMissingScalarFields(
        userYaml: userYaml,
        bundleYaml: bundleYaml,
      );
      expect(result, isNotNull);
      expect(result!.added, ['models.ml_threshold'],
          reason: 'the 2-space scalar is added; the 4-space nested one is not');
      expect(result.yaml, contains('  ml_threshold: 0.5'));
      expect(result.yaml, isNot(contains('    prefilter_top_k')));
    });
  });
}
