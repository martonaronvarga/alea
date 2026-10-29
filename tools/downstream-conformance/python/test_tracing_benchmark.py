"""Tracing comparisons require matching work and complete planned denominators."""
import copy
import unittest
from benchmark_tracing import summarize


class TracingBenchmarkTests(unittest.TestCase):
    def rows(self, case="stream", outcome="Completed"):
        return [{"round": 0, "case": case, "mode": mode, "status": 0,
                 "measurements": [{"fingerprint": "abc", "nanoseconds": 100, "events": 0, "outcome": outcome} for _ in range(3)]}
                for mode in ("off", "none", "filtered", "info")]

    def test_matched_work_has_unit_ratios(self):
        result = summarize(self.rows(), 1)["stream"]
        self.assertTrue(result["identity_passed"])
        for mode in result["modes"].values():
            self.assertEqual(mode["median_paired_ratio"], 1)

    def test_failures_mismatches_missing_and_duplicate_jobs_disable_comparison(self):
        changes = []
        failed = self.rows(); failed[1]["status"] = 1; changes.append(failed)
        mismatched = self.rows(); mismatched[1]["measurements"][1]["fingerprint"] = "wrong"; changes.append(mismatched)
        changes.append(self.rows()[:-1])
        duplicate = self.rows(); duplicate[2] = copy.deepcopy(duplicate[1]); changes.append(duplicate)
        for rows in changes:
            result = summarize(rows, 1)["stream"]
            self.assertFalse(result["identity_passed"])
            self.assertEqual(result["modes"], {})

    def test_matching_failed_fit_is_never_labelled_a_successful_fit(self):
        result = summarize(self.rows("bernoulli", "LineSearchFailed"), 1)["bernoulli"]
        self.assertTrue(result["identity_passed"])
        self.assertFalse(result["all_fits_score_tolerance"])
        self.assertEqual(result["workload_outcomes"], ["LineSearchFailed"])
