"""ABOUTME: Protect unknown handling and ordinal semantics in the additive bridge."""
from pathlib import Path
import csv
import tempfile
import unittest

from historical_audit.oz_bridge import apply_scores, consensus, describe, load_kernel, parse_scores, run_bridge


class BridgeTests(unittest.TestCase):
    def test_all_four_outputs_and_unknown(self) -> None:
        kernel = load_kernel(Path(__file__).resolve().parents[2] / "evaluate_rule.py")
        rows = [{"intent_score": a, "action_participation": b}
                for a, b in [(2, 1), (1, 0), (0, 1), (0, 0), (0, "unknown")]]
        result = apply_scores(rows, kernel)
        self.assertEqual([row["raw_oz_label"] for row in result],
                         ["a_flag", "c_cheap_talk", "b_parallel_behaviour", "d_competition", ""])
        self.assertEqual(result[-1]["process_category"], "unresolved")

    def test_binary_consensus_not_ordinal_unanimity(self) -> None:
        self.assertEqual(consensus([1, 2, 1]), 1)
        self.assertEqual(consensus([0, 0, 0]), 0)
        self.assertEqual(consensus([0, 1, 0]), "unknown")
        self.assertEqual(consensus([0, "unknown", 0]), "unknown")

    def test_saved_score_contract(self) -> None:
        self.assertEqual(parse_scores('[0, 2, "unknown"]', 2), [0, 2, "unknown"])
        for raw in ['[true, 0, 0]', '[3, 0, 0]', '[0, 0]', '{}']:
            with self.assertRaises(ValueError):
                parse_scores(raw, 2)

    def test_added_axis_can_reduce_coverage(self) -> None:
        rows = [{"intent_score": "unknown", "action_participation": 0,
                 "process_category": "unresolved", "raw_oz_label": ""},
                {"intent_score": 0, "action_participation": 0,
                 "process_category": "neither_observed", "raw_oz_label": "d_competition"}]
        summary = describe(rows)
        self.assertEqual(summary["known_both_axes"], 1)
        self.assertEqual(summary["known_participation_axis"], 2)
        self.assertEqual(summary["additional_unresolved_vs_participation_only"], 1)

    def test_fact_join_retains_scores_execution_and_empty_reference(self) -> None:
        kernel = Path(__file__).resolve().parents[2] / "evaluate_rule.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seller_round_stability.csv").write_text(
                'run_id,round,agent_id,intent_scores,action_scores,execution_status,human_reference\n'
                'run,1,seller,"[1, 2, 1]","[1, 1, 1]",withdrawal_not_executed_order_observed,\n')
            (root / "run_assessments.csv").write_text('run_id,buyer_surplus,case_type\nrun,10,\n')
            (root / "round_facts.csv").write_text('run_id,round,buyer_surplus\nrun,1,10\n')
            summary = run_bridge(root, kernel, root / "out", root / "round_facts.csv")
            with (root / "out/integrated_seller_rounds.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(summary["integrated_seller_round_rows"], 1)
            self.assertEqual(summary["integrated_round_rows"], 1)
            self.assertEqual(rows[0]["intent_scores"], "[1, 2, 1]")
            self.assertEqual(rows[0]["consensus_willingness_present"], "1")
            self.assertEqual(rows[0]["execution_status"], "withdrawal_not_executed_order_observed")
            self.assertEqual(rows[0]["human_reference"], "")
            (root / "run_assessments.csv").write_text('run_id,buyer_surplus,case_type\nother,10,\n')
            with self.assertRaises(ValueError):
                run_bridge(root, kernel, root / "bad-output")
            self.assertFalse((root / "bad-output").exists())


if __name__ == "__main__":
    unittest.main()
