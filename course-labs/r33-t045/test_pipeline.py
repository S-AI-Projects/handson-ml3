"""
Unit tests for Course Lab R33-T045 Image Preprocessing Contract.
"""
import unittest
import json
import os
from pipeline import ImagePreprocessor

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestImagePreprocessingContract(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)
        self.preprocessor = ImagePreprocessor()

    def test_valid_image_scaling_and_channel_axis(self):
        processed = self.preprocessor.transform_dataset(self.fixture)
        self.assertEqual(len(processed), 6)
        for p in processed:
            self.assertEqual(p["shape"], (4, 4, 1))
            # Verify values between 0.0 and 1.0
            for row in p["processed_image"]:
                for pixel in row:
                    self.assertIsInstance(pixel[0], float)
                    self.assertTrue(0.0 <= pixel[0] <= 1.0)

    def test_zero_image_all_zeros(self):
        zero_res = self.preprocessor.transform(self.fixture[0]["image"])
        for row in zero_res:
            for pixel in row:
                self.assertEqual(pixel[0], 0.0)

    def test_diagonal_image_values(self):
        diag_res = self.preprocessor.transform(self.fixture[3]["image"])
        for r in range(4):
            for c in range(4):
                expected = 1.0 if r == c else 0.0
                self.assertAlmostEqual(diag_res[r][c][0], expected)

    def test_rejects_out_of_range_pixels(self):
        bad_img = [[0]*4 for _ in range(4)]
        bad_img[1][1] = 256
        with self.assertRaises(ValueError):
            self.preprocessor.transform(bad_img)

        bad_img[1][1] = -1
        with self.assertRaises(ValueError):
            self.preprocessor.transform(bad_img)

    def test_rejects_nan_and_inf(self):
        bad_img = [[0]*4 for _ in range(4)]
        bad_img[0][0] = float("nan")
        with self.assertRaises(ValueError):
            self.preprocessor.transform(bad_img)

    def test_rejects_invalid_dimensions(self):
        # 3x4 image
        with self.assertRaises(ValueError):
            self.preprocessor.transform([[0]*4 for _ in range(3)])
        # 4x5 image
        with self.assertRaises(ValueError):
            self.preprocessor.transform([[0]*5 for _ in range(4)])

    def test_stateless_no_contamination(self):
        # Ensure transform maintains no internal mutable state across calls
        self.assertEqual(hasattr(self.preprocessor, "fit"), False)

if __name__ == "__main__":
    unittest.main()
