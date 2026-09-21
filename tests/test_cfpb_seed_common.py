import unittest

import pandas as pd

from src.findisputeeval.curation.cfpb_privacy_qa import finalize_privacy_review

from src.findisputeeval.curation.cfpb_seed_common import (
    audited_fuzzy_group_cap,
    deterministic_group_cap,
    proportional_allocation,
    proportional_stratified_sample_with_accounting,
    select_coverage_supplement,
)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "Complaint ID": [str(value) for value in range(1, 11)],
                "Product": ["A"] * 6 + ["B"] * 3 + ["A|B"],
                "Issue": ["x"] * 6 + ["y"] * 3 + ["x"],
                "month": ["2026-01"] * 9 + ["2026-02"],
            }
        )

    def test_allocation_reconciles_and_is_deterministic(self):
        first = proportional_allocation(self.frame, 5, ["Product", "Issue", "month"])
        second = proportional_allocation(self.frame, 5, ["Product", "Issue", "month"])
        self.assertEqual(int(first["allocated"].sum()), 5)
        pd.testing.assert_frame_equal(first, second)

    def test_multicolumn_keys_do_not_collide_on_pipe(self):
        frame = pd.DataFrame(
            {
                "Complaint ID": ["1", "2"],
                "a": ["x|y", "x"],
                "b": ["z", "y|z"],
            }
        )
        allocation = proportional_allocation(frame, 2, ["a", "b"])
        self.assertEqual(len(allocation), 2)

    def test_sample_exact_size_and_repeatability(self):
        first, accounting = proportional_stratified_sample_with_accounting(
            self.frame, 5, ["Product", "Issue", "month"], 42
        )
        second, _ = proportional_stratified_sample_with_accounting(
            self.frame, 5, ["Product", "Issue", "month"], 42
        )
        self.assertEqual(len(first), 5)
        self.assertEqual(first["Complaint ID"].tolist(), second["Complaint ID"].tolist())
        self.assertEqual(int(accounting["allocated"].sum()), 5)

    def test_coverage_supplement_adds_uncovered_cells_only(self):
        core = self.frame.iloc[[0, 6]].copy()
        supplement, accounting = select_coverage_supplement(
            self.frame, core, ["Product", "Issue", "month"], 42
        )
        self.assertEqual(len(supplement), 1)
        self.assertEqual(supplement.iloc[0]["Complaint ID"], "10")
        self.assertFalse(bool(accounting.iloc[0]["prevalence_eligible"]))


class DuplicateTests(unittest.TestCase):
    def test_group_cap_is_deterministic(self):
        frame = pd.DataFrame(
            {
                "Complaint ID": ["1", "2", "3", "4"],
                "hash": ["a", "a", "b", "b"],
            }
        )
        first, accounting = deterministic_group_cap(frame, "hash", 1, 7, "exact")
        second, _ = deterministic_group_cap(frame, "hash", 1, 7, "exact")
        self.assertEqual(len(first), 2)
        self.assertEqual(accounting["removed"], 2)
        self.assertEqual(first["Complaint ID"].tolist(), second["Complaint ID"].tolist())

    def test_rejected_bridge_splits_fuzzy_component(self):
        frame = pd.DataFrame({"Complaint ID": ["1", "2", "3"]})
        pairs = pd.DataFrame(
            {
                "complaint_id_a": ["1", "2"],
                "complaint_id_b": ["2", "3"],
                "manual_same_template": ["yes", "yes"],
            }
        )
        # Component ID is derived by the production utility; this synthetic
        # cluster is small and therefore does not require a cluster label.
        clusters = pd.DataFrame(
            {"component_id": [], "manual_cluster_valid": []}
        )
        bridges = pd.DataFrame(
            {
                "complaint_id_a": ["2"],
                "complaint_id_b": ["3"],
                "manual_same_template": ["no"],
            }
        )
        retained, accounting = audited_fuzzy_group_cap(
            frame,
            pairs,
            clusters,
            bridges,
            cap=1,
            random_seed=7,
            maximum_unapproved_component_size=50,
        )
        self.assertEqual(len(retained), 2)
        self.assertEqual(accounting["reviewed_edges_removed"], 1)


class PrivacyReviewTests(unittest.TestCase):
    def test_pending_review_fails_closed(self):
        review = pd.DataFrame(
            {"manual_pii_present": ["pending"], "release_action": ["pending"], "reviewer": [""]}
        )
        with self.assertRaises(ValueError):
            finalize_privacy_review(review, scope="smoke20", source_sha256="abc")

    def test_negative_review_can_clear(self):
        review = pd.DataFrame(
            {"manual_pii_present": ["no"], "release_action": ["allow"], "reviewer": ["chang"]}
        )
        result = finalize_privacy_review(review, scope="smoke20", source_sha256="abc")
        self.assertTrue(result["release_clearance_passed"])


if __name__ == "__main__":
    unittest.main()
