import json
import unittest
from pathlib import Path

from pipeline import analyse, validate_fixture


FIXTURE_PATH = Path(__file__).with_name("fixture.json")


class TestReinforcementLearningAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with FIXTURE_PATH.open(encoding="utf-8") as file:
            cls.data = json.load(file)

    def test_fixture_is_valid(self):
        validate_fixture(self.data)

    def test_data_is_synthetic(self):
        self.assertEqual(
            self.data["dataset_type"], "synthetic_simulated"
        )

    def test_all_splits_exist(self):
        splits = {episode["split"] for episode in self.data["episodes"]}
        self.assertEqual(splits, {"train", "validation", "test"})

    def test_right_actions_reach_terminal_state(self):
        episode = self.data["episodes"][0]
        transitions = episode["transitions"]

        self.assertEqual(
            [item["action"] for item in transitions],
            ["right", "right"]
        )
        self.assertEqual(transitions[-1]["next_state"], 2)
        self.assertEqual(transitions[-1]["reward"], 1)
        self.assertTrue(transitions[-1]["terminated"])

    def test_five_left_actions_hit_step_cap(self):
        episode = self.data["episodes"][1]
        transitions = episode["transitions"]

        self.assertEqual(len(transitions), 5)
        self.assertTrue(all(item["action"] == "left" for item in transitions))
        self.assertTrue(all(item["reward"] == 0 for item in transitions))
        self.assertTrue(all(not item["terminated"] for item in transitions))

    def test_report_finds_injected_wrong_prediction(self):
        result = analyse(self.data)

        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["errors"][0]["record_id"], "val-03")

    def test_expected_and_predicted_values_are_reported(self):
        result = analyse(self.data)
        error = result["errors"][0]

        self.assertEqual(error["expected_value"], 0)
        self.assertEqual(error["predicted_value"], 0.75)

    def test_error_reason_is_reported(self):
        result = analyse(self.data)

        self.assertEqual(
            result["errors"][0]["reason_category"],
            "overestimated_zero_return"
        )

    def test_record_counts(self):
        result = analyse(self.data)

        self.assertEqual(result["total_records"], 9)
        self.assertEqual(result["correct_count"], 8)
        self.assertEqual(result["error_count"], 1)

    def test_correct_predictions_are_not_reported(self):
        result = analyse(self.data)

        reported_ids = {item["record_id"] for item in result["errors"]}
        self.assertEqual(reported_ids, {"val-03"})

    def test_duplicate_record_ids_are_rejected(self):
        data = json.loads(json.dumps(self.data))
        data["episodes"][1]["transitions"][0]["record_id"] = "train-01"

        with self.assertRaises(ValueError):
            validate_fixture(data)

    def test_invalid_action_is_rejected(self):
        data = json.loads(json.dumps(self.data))
        data["episodes"][0]["transitions"][0]["action"] = "jump"

        with self.assertRaises(ValueError):
            validate_fixture(data)

    def test_non_boolean_terminated_is_rejected(self):
        data = json.loads(json.dumps(self.data))
        data["episodes"][0]["transitions"][0]["terminated"] = "false"

        with self.assertRaises(ValueError):
            validate_fixture(data)

    def test_missing_split_is_rejected(self):
        data = json.loads(json.dumps(self.data))
        data["episodes"][2]["split"] = "validation"

        with self.assertRaises(ValueError):
            validate_fixture(data)


if __name__ == "__main__":
    unittest.main()
