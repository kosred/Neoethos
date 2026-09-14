#!/usr/bin/env python3
"""Fixture-runner negative controls only: these never constitute GPU evidence."""
import copy
import json
import unittest

import hipify_sources as audit
import native_backend_parity as parity


def records(backend="cpu"):
    return [dict(type="metadata", schema=parity.SUITES["exact-log"][0], backend=backend, cases=48,
                 role="production_cpu" if backend == "cpu" else "production_device",
                 device_name="SYNTHETIC_UNIT_TEST_NOT_A_DEVICE", device_ordinal=0,
                 architecture="synthetic", warp_size=32),
            *[dict(type="case", id=f"synthetic_{i}", input_bits="3ff0000000000000", accepted=True,
                   b_bits="0000000000000000", c_bits="0000000000000000", operation="log",
                   output_bits="0000000000000000", passed=True) for i in range(48)],
            dict(type="summary", cases=48, failures=0, device_executed=backend != "cpu")]


def encoded(rows):
    return "\n".join(json.dumps(row) for row in rows)


def receipts():
    return [dict(schema=parity.SCHEMA, phase="run", backend=backend, suite="exact-log",
                 success=True, source_identity={"test-only": "not-device-evidence"},
                 metadata=records(backend)[0], cases=records(backend)[1:-1])
            for backend in ("cpu", "cuda", "hip")]


class ResultAdmissionTests(unittest.TestCase):
    def parse(self, rows, backend="cpu"):
        return parity.parse_results(encoded(rows), "exact-log", backend)

    def test_valid_cpu_result_is_parseable_not_device_proof(self):
        self.assertEqual(len(self.parse(records())), 48)

    def test_backend_substitution_refused(self):
        with self.assertRaises(audit.StageError):
            self.parse(records("cpu"), "hip")

    def test_absent_device_refused(self):
        rows = records("hip")
        rows[-1]["device_executed"] = False
        with self.assertRaises(audit.StageError):
            self.parse(rows, "hip")

    def test_failure_refused(self):
        rows = records()
        rows[-1]["failures"] = 1
        with self.assertRaises(audit.StageError):
            self.parse(rows)

    def test_checkpoint_failure_cannot_hide_in_passing_summary(self):
        rows = records()
        rows[1]["passed"] = False
        with self.assertRaises(audit.StageError):
            self.parse(rows)

    def test_zero_or_partial_cases_refused(self):
        for rows in ([records()[0], records()[-1]], records()[:-1]):
            with self.assertRaises(audit.StageError):
                self.parse(rows)

    def test_count_drift_refused(self):
        for key in (0, -1):
            rows = records()
            rows[key]["cases"] = 2
            with self.assertRaises(audit.StageError):
                self.parse(rows)

    def test_duplicate_ids_refused(self):
        rows = records()
        rows[2] = copy.deepcopy(rows[1])
        with self.assertRaises(audit.StageError):
            self.parse(rows)

    def test_self_consistent_partial_coverage_refused(self):
        rows = records()
        rows = [rows[0], rows[1], rows[-1]]
        rows[0]["cases"] = rows[-1]["cases"] = 1
        with self.assertRaises(audit.StageError):
            self.parse(rows)

    def test_missing_actual_device_identity_refused(self):
        rows = records("hip")
        del rows[0]["device_name"]
        with self.assertRaises(audit.StageError):
            self.parse(rows, "hip")

    def test_missing_mathematical_inputs_refused(self):
        rows = records()
        del rows[1]["input_bits"]
        with self.assertRaises(audit.StageError):
            self.parse(rows)

    def test_empty_first_hit_expectations_refused(self):
        rows = [dict(type="metadata", schema=parity.SUITES["first-hit"][0], backend="cpu",
                     cases=212, fixtures=106, scope="first_hit_discrete_decisions_only",
                     role="independent_reference", input_identity=dict(
                         algorithm="fnv1a64_le_v1_noncryptographic", value="0000000000000000")),
                *[dict(type="first_hit", id=f"synthetic_{i}", expected={}, result={}) for i in range(212)],
                dict(type="summary", cases=212, failures=0, device_executed=False)]
        with self.assertRaises(audit.StageError):
            parity.parse_results(encoded(rows), "first-hit", "cpu")

    def test_unexpected_output_is_not_silently_filtered(self):
        with self.assertRaises(ValueError):
            parity.parse_results("INFO: skipped\n" + encoded(records()), "exact-log", "cpu")

    def test_non_boolean_device_evidence_refused(self):
        rows = records("cuda")
        rows[-1]["device_executed"] = 1
        with self.assertRaises(audit.StageError):
            self.parse(rows, "cuda")


class ExactComparisonTests(unittest.TestCase):
    def test_three_matching_synthetic_receipts_exercise_comparator_only(self):
        self.assertEqual(parity.compare_receipts(receipts()), 48)

    def test_missing_or_duplicate_backend_refused(self):
        for data in (receipts()[:2], [receipts()[0]] * 3):
            with self.assertRaises(audit.StageError):
                parity.compare_receipts(data)

    def test_failure_refused(self):
        data = receipts()
        data[2]["success"] = False
        with self.assertRaises(audit.StageError):
            parity.compare_receipts(data)

    def test_source_and_input_drift_refused(self):
        for field, value in (("source_identity", {"different": "source"}),
                             ("metadata", {"input_identity": "different input"})):
            data = receipts()
            data[2][field] = value
            with self.assertRaises(audit.StageError):
                parity.compare_receipts(data)

    def test_one_bit_signed_zero_and_decision_differences_refused(self):
        for field, value in (("output_bits", "0000000000000001"),
                             ("output_bits", "8000000000000000"), ("accepted", False)):
            data = receipts()
            data[2]["cases"][0][field] = value
            with self.assertRaises(audit.StageError):
                parity.compare_receipts(data)

    def test_different_suite_refused(self):
        data = receipts()
        data[2]["suite"] = "first-hit"
        with self.assertRaises(audit.StageError):
            parity.compare_receipts(data)


if __name__ == "__main__":
    unittest.main()
