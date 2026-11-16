import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import (
    load_json,
    run_pipeline,
    validate_config,
    validate_fixture,
)


HERE = Path(__file__).parent
FIXTURE = load_json(HERE / "fixture.json")
CONFIG = load_json(HERE / "config.json")

EXPECTED = {
    "train": {"img-01": 0, "img-04": 1},
    "validation": {"img-02": 0, "img-05": 1},
    "test": {"img-03": 0, "img-06": 1},
}


class PipelineTests(unittest.TestCase):
    def test_valid_configuration(self):
        validate_config(copy.deepcopy(CONFIG))

    def test_fixture_has_six_images(self):
        self.assertEqual(len(FIXTURE["records"]), 6)

    def test_all_images_are_four_by_four(self):
        for record in FIXTURE["records"]:
            self.assertEqual(len(record["pixels"]), 4)
            self.assertTrue(all(len(row) == 4 for row in record["pixels"]))

    def test_exact_expected_predictions(self):
        self.assertEqual(run_pipeline(FIXTURE, CONFIG), EXPECTED)

    def test_reloaded_configuration_reproduces_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            fixture_path = Path(directory) / "fixture.json"
            config_path.write_text(json.dumps(CONFIG), encoding="utf-8")
            fixture_path.write_text(json.dumps(FIXTURE), encoding="utf-8")

            from pipeline import run_from_files
            first = run_from_files(fixture_path, config_path)
            reloaded = run_from_files(fixture_path, config_path)
            self.assertEqual(first, reloaded)
            self.assertEqual(reloaded, EXPECTED)

    def test_unknown_configuration_field_rejected(self):
        config = copy.deepcopy(CONFIG)
        config["mystery_setting"] = True
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_config(config)

    def test_missing_configuration_field_rejected(self):
        config = copy.deepcopy(CONFIG)
        del config["seed"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_config(config)

    def test_incompatible_feature_order_rejected(self):
        config = copy.deepcopy(CONFIG)
        config["feature_order"] = ["label", "pixels"]
        with self.assertRaisesRegex(ValueError, "feature_order"):
            validate_config(config)

    def test_contradictory_algorithm_setting_rejected(self):
        config = copy.deepcopy(CONFIG)
        config["algorithm"]["distance"] = "euclidean"
        with self.assertRaisesRegex(ValueError, "algorithm"):
            validate_config(config)

    def test_incompatible_image_shape_rejected(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture["records"][0]["pixels"] = [[0, 0], [0, 0]]
        with self.assertRaisesRegex(ValueError, "4 rows"):
            validate_fixture(fixture, CONFIG)

    def test_invalid_pixel_value_rejected(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture["records"][0]["pixels"][0][0] = 128
        with self.assertRaisesRegex(ValueError, "grayscale"):
            validate_fixture(fixture, CONFIG)

    def test_duplicate_record_id_rejected(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture["records"][1]["record_id"] = "img-01"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_fixture(fixture, CONFIG)

    def test_split_must_cover_all_records(self):
        config = copy.deepcopy(CONFIG)
        config["splits"]["test"].remove("img-06")
        with self.assertRaisesRegex(ValueError, "cover every"):
            run_pipeline(FIXTURE, config)

    def test_group_leakage_rejected(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture["records"][1]["group_id"] = "group-01"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            run_pipeline(fixture, CONFIG)

    def test_fixture_must_be_labelled_synthetic(self):
        fixture = copy.deepcopy(FIXTURE)
        fixture["data_type"] = "real_dataset"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            validate_fixture(fixture, CONFIG)


if __name__ == "__main__":
    unittest.main()
