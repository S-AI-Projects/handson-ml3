"""
Unit tests for Course Lab R33-T044 RL chain environment and metric calculations.
"""
import unittest
import json
import os
from pipeline import ThreeStateChainEnv, calculate_precision_recall, calculate_mae

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestRLChainAndEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)

    def test_right_right_trajectory_reaches_terminal_reward_one(self):
        env = ThreeStateChainEnv(step_cap=5)
        res = env.run_trajectory(["right", "right"])
        self.assertEqual(res["final_state"], 2)
        self.assertEqual(res["total_reward"], 1.0)
        self.assertEqual(res["steps"], 2)
        self.assertTrue(res["done"])
        self.assertEqual(res["trajectory"], [0, 1, 2])

    def test_five_left_actions_hit_step_cap_with_reward_zero(self):
        env = ThreeStateChainEnv(step_cap=5)
        res = env.run_trajectory(["left", "left", "left", "left", "left"])
        self.assertEqual(res["final_state"], 0)
        self.assertEqual(res["total_reward"], 0.0)
        self.assertEqual(res["steps"], 5)
        self.assertTrue(res["done"])
        self.assertEqual(res["trajectory"], [0, 0, 0, 0, 0, 0])

    def test_classification_metric_fixture(self):
        # truth [1, 1, 0, 0] / prediction [1, 0, 1, 0] -> precision=recall=0.5
        c_spec = self.fixture["metric_test_cases"]["classification"]
        prec, rec = calculate_precision_recall(c_spec["y_true"], c_spec["y_pred"])
        self.assertAlmostEqual(prec, c_spec["expected_precision"], places=6)
        self.assertAlmostEqual(rec, c_spec["expected_recall"], places=6)

    def test_regression_mae_metric_fixture(self):
        # truth [1, 2, 3] / prediction [1, 3, 2] -> MAE=2/3
        r_spec = self.fixture["metric_test_cases"]["regression"]
        mae = calculate_mae(r_spec["y_true"], r_spec["y_pred"])
        self.assertAlmostEqual(mae, r_spec["expected_mae"], places=6)

    def test_zero_denominator_safe_handling(self):
        # No positive predictions -> denominator = 0
        prec, rec = calculate_precision_recall([1, 1], [0, 0], zero_division=0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)

    def test_deliberate_metric_failure_detected(self):
        # If true is inverted, precision/recall must change
        prec, rec = calculate_precision_recall([0, 0, 1, 1], [1, 0, 1, 0])
        self.assertNotEqual((prec, rec), (1.0, 1.0))

if __name__ == "__main__":
    unittest.main()
