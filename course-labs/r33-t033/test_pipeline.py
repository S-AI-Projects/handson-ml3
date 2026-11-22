"""
Unit test suite for Course Lab R33-T033: Reinforcement Learning Minimal Baseline.

Validates:
- Three-state chain dynamics: start 0, 'right' +1, 'left' -1 floor 0, terminal state 2 (reward 1.0).
- [right, right] ends with total reward 1.0 in 2 steps.
- Five left actions hit the step cap with total reward 0.0.
- AlwaysRightPolicy baseline evaluation on held-out environment episodes.
- Transition schema validation and rejection of invalid states, actions, or inconsistent rewards.
- Deliberate mutation failure detection.
"""

from pathlib import Path
import copy
import unittest

from pipeline import (
    AlwaysLeftPolicy,
    AlwaysRightPolicy,
    RandomPolicy,
    SchemaValidator,
    ThreeStateChainEnv,
    evaluate_policy,
    load_dataset,
)


class TestR33T033ReinforcementBaseline(unittest.TestCase):
    """Offline, CPU-only unit tests for RL three-state chain and minimal baseline."""

    def setUp(self) -> None:
        self.env = ThreeStateChainEnv(step_cap=5)
        self.validator = SchemaValidator()
        self.fixture_dir = Path(__file__).parent
        self.json_fixture_path = self.fixture_dir / "fixture.json"
        self.csv_fixture_path = self.fixture_dir / "fixture.csv"
        self.fixture_records = load_dataset(self.json_fixture_path)

    # 1. Right, Right Trajectory Verification
    def test_right_right_trajectory_reaches_terminal_with_reward_1(self) -> None:
        """[right, right] trajectory must reach terminal state 2 with total reward 1.0."""
        res = self.env.run_trajectory(["right", "right"])
        self.assertEqual(res["final_state"], 2, "Final state must be terminal state 2.")
        self.assertEqual(res["total_reward"], 1.0, "Total reward for [right, right] must be 1.0.")
        self.assertEqual(res["steps"], 2, "Must terminate in exactly 2 steps.")
        self.assertTrue(res["terminated"], "Episode must be terminated.")
        self.assertEqual(res["trajectory"], [0, 1, 2], "State trajectory must be 0 -> 1 -> 2.")

    # 2. Five Left Actions Trajectory Verification
    def test_five_left_actions_hit_step_cap_with_reward_0(self) -> None:
        """Five left actions must stay clamped at 0, hit step cap 5, and yield total reward 0.0."""
        res = self.env.run_trajectory(["left", "left", "left", "left", "left"])
        self.assertEqual(res["final_state"], 0, "Final state must remain at floor 0.")
        self.assertEqual(res["total_reward"], 0.0, "Total reward for five lefts must be 0.0.")
        self.assertEqual(res["steps"], 5, "Must reach step cap of 5 steps.")
        self.assertTrue(res["terminated"], "Episode must terminate upon hitting step cap.")
        self.assertEqual(res["trajectory"], [0, 0, 0, 0, 0, 0], "State must stay at 0 across all steps.")

    # 3. Environment Step Mechanics
    def test_single_step_mechanics(self) -> None:
        """Verify individual step advances, floor boundary, and terminal clamping."""
        self.env.reset()
        # Left from 0 stays 0
        s, r, done, _ = self.env.step("left")
        self.assertEqual(s, 0)
        self.assertEqual(r, 0.0)
        self.assertFalse(done)

        # Right from 0 goes to 1
        s, r, done, _ = self.env.step("right")
        self.assertEqual(s, 1)
        self.assertEqual(r, 0.0)
        self.assertFalse(done)

        # Right from 1 goes to 2 (terminal with reward 1.0)
        s, r, done, _ = self.env.step("right")
        self.assertEqual(s, 2)
        self.assertEqual(r, 1.0)
        self.assertTrue(done)

    # 4. AlwaysRightPolicy Baseline Evaluation
    def test_always_right_policy_baseline(self) -> None:
        """AlwaysRightPolicy achieves 100% success rate and mean reward 1.0 in 2 steps."""
        policy = AlwaysRightPolicy()
        metrics = evaluate_policy(policy, self.env, n_episodes=10)

        self.assertEqual(metrics["policy"], "AlwaysRightPolicy")
        self.assertEqual(metrics["mean_reward"], 1.0, "AlwaysRightPolicy must achieve reward 1.0.")
        self.assertEqual(metrics["mean_steps"], 2.0, "AlwaysRightPolicy must complete in 2 steps.")
        self.assertEqual(metrics["success_rate"], 1.0, "AlwaysRightPolicy success rate must be 1.0.")

    # 5. Baseline Comparison: AlwaysLeftPolicy & RandomPolicy
    def test_policy_comparisons(self) -> None:
        """AlwaysLeftPolicy fails to reach terminal state; AlwaysRightPolicy strictly outperforms it."""
        left_policy = AlwaysLeftPolicy()
        metrics_left = evaluate_policy(left_policy, self.env, n_episodes=5)
        self.assertEqual(metrics_left["mean_reward"], 0.0)
        self.assertEqual(metrics_left["success_rate"], 0.0)
        self.assertEqual(metrics_left["mean_steps"], 5.0)

        right_policy = AlwaysRightPolicy()
        metrics_right = evaluate_policy(right_policy, self.env, n_episodes=5)
        self.assertGreater(metrics_right["mean_reward"], metrics_left["mean_reward"])

    # 6. Synthetic Fixture Dataset Validation
    def test_fixture_records_schema_validation(self) -> None:
        """The synthetic transition fixture must pass schema validation with 0 errors."""
        valid_recs, errors = self.validator.validate_records(self.fixture_records)
        self.assertEqual(len(valid_recs), len(self.fixture_records))
        self.assertEqual(len(errors), 0, f"Fixture produced schema errors: {errors}")

    def test_csv_matches_json_fixture(self) -> None:
        """CSV and JSON fixtures must represent the identical transition records."""
        csv_records = load_dataset(self.csv_fixture_path)
        self.assertEqual(len(csv_records), len(self.fixture_records))
        for r_csv, r_json in zip(csv_records, self.fixture_records):
            self.assertEqual(r_csv["episode_id"], r_json["episode_id"])
            self.assertEqual(r_csv["state"], r_json["state"])
            self.assertEqual(r_csv["action"], r_json["action"])
            self.assertEqual(r_csv["next_state"], r_json["next_state"])
            self.assertEqual(r_csv["reward"], r_json["reward"])
            self.assertEqual(r_csv["terminated"], r_json["terminated"])

    # 7. Invalid Transitions Rejection
    def test_schema_validator_rejects_inconsistent_physics(self) -> None:
        """Inconsistent next_state or invalid actions must be rejected with descriptive error."""
        # Invalid action
        bad_action = copy.deepcopy(self.fixture_records[0])
        bad_action["action"] = "jump"
        errs = self.validator.validate_transition(bad_action)
        self.assertTrue(any(e.field == "action" and "Invalid action" in e.reason for e in errs))

        # Inconsistent next state (state 0 + right should be 1, not 2)
        bad_next = copy.deepcopy(self.fixture_records[0])
        bad_next["next_state"] = 2
        errs = self.validator.validate_transition(bad_next)
        self.assertTrue(any(e.field == "next_state" and "Inconsistent transition" in e.reason for e in errs))

        # Inconsistent reward (state 2 terminal transition without reward 1.0)
        bad_reward = copy.deepcopy(self.fixture_records[1])
        bad_reward["reward"] = 0.0
        errs = self.validator.validate_transition(bad_reward)
        self.assertTrue(any(e.field == "reward" and "Inconsistent reward" in e.reason for e in errs))

    # 8. Deliberate Mutation Failure Detection
    def test_deliberate_mutation_detection(self) -> None:
        """Deliberately mutating the terminal reward from 1.0 to 0.0 triggers an assertion failure."""
        res = self.env.run_trajectory(["right", "right"])
        with self.assertRaises(AssertionError):
            self.assertEqual(res["total_reward"], 0.0, "Deliberate assertion mutation must fail.")


if __name__ == "__main__":
    unittest.main()
