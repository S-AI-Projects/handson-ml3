import copy
import json
import os
import unittest
from array import array

import fixture
from fixture import split_of, to_grid
from preprocess import (ContaminationError, Float32Image, PreprocessError,
                        Scaler, serving_request)

# Expected output of the diagonal image: 1.0 on the main diagonal, else 0.0.
EXPECTED_DIAG = [1.0 if r == c else 0.0 for r in range(4) for c in range(4)]
if os.environ.get("R33_T060_MUTATE_EXPECTED"):
    EXPECTED_DIAG = [0.5 if r == c else 0.0 for r in range(4) for c in range(4)]


def fitted():
    s = Scaler()
    s.fit([r for r in fixture.FIXTURE if split_of(r) == "train"])
    return s


class FixtureContract(unittest.TestCase):
    def test_fixture_and_split_separation(self):
        recs = fixture.FIXTURE
        self.assertEqual(len(recs), 6)
        self.assertEqual(len({r["record_id"] for r in recs}), 6)
        self.assertEqual([r["label"] for r in recs], [0, 1, 0, 1, 0, 1])
        by_split = {}
        for r in recs:
            by_split.setdefault(split_of(r), set()).add(r["group_id"])
        self.assertEqual(by_split, {"train": {"g0"}, "validation": {"g1"}, "test": {"g2"}})


class OutputValues(unittest.TestCase):
    def test_diagonal_image_values_dtype_shape(self):
        out = fitted().transform(to_grid(fixture.FIXTURE[1]["pixels"]))
        self.assertEqual(out.flat, EXPECTED_DIAG)
        self.assertEqual(out.dtype, "float32")
        self.assertEqual(out.shape, (4, 4, 1))
        self.assertEqual(out._data.typecode, "f")
        self.assertEqual(out._data.itemsize, 4)

    def test_zero_image_all_zero(self):
        out = fitted().transform(to_grid(fixture.FIXTURE[0]["pixels"]))
        self.assertEqual(out.flat, [0.0] * 16)
        self.assertEqual(out.shape, (4, 4, 1))

    def test_explicit_channel_axis_in_nested_output(self):
        nested = fitted().transform(to_grid(fixture.FIXTURE[1]["pixels"])).tolist()
        self.assertEqual((len(nested), len(nested[0]), len(nested[0][0])), (4, 4, 1))
        self.assertEqual(nested[2][2], [1.0])
        self.assertEqual(nested[0][1], [0.0])

    def test_range_is_zero_to_one_and_midvalue(self):
        img = [[0, 51, 102, 153], [204, 255, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        flat = fitted().transform(img).flat
        self.assertEqual(min(flat), 0.0)
        self.assertEqual(max(flat), 1.0)
        self.assertEqual(flat[1], array("f", [0.2])[0])  # 51/255 in float32


class InvalidInputs(unittest.TestCase):
    def setUp(self):
        self.s = fitted()
        self.ok = to_grid(fixture.FIXTURE[1]["pixels"])

    def bad(self, image, pattern):
        with self.assertRaisesRegex(PreprocessError, pattern):
            self.s.transform(image)

    def test_invalid_dimensions_rejected(self):
        self.bad([row[:3] for row in self.ok[:3]], "dimensions")        # 3x3
        self.bad([row + [0] for row in self.ok], "dimensions")          # 4x5
        self.bad(self.ok + [[0, 0, 0, 0]], "dimensions")                # 5x4
        self.bad(fixture.FIXTURE[1]["pixels"], "dimensions")            # flat
        self.bad([[[v] for v in row] for row in self.ok], "channel axis")  # 4x4x1
        self.bad([self.ok[0], self.ok[1][:3], self.ok[2], self.ok[3]], "dimensions")  # ragged
        self.bad([], "dimensions")

    def test_out_of_range_pixels_rejected(self):
        for bad_value in (-1, 256, 1000):
            img = copy.deepcopy(self.ok)
            img[1][2] = bad_value
            self.bad(img, "out of range")

    def test_non_integer_pixels_rejected(self):
        for bad_value in (12.5, True, "7", None):
            img = copy.deepcopy(self.ok)
            img[0][0] = bad_value
            self.bad(img, "must be an int")


class LeakageControl(unittest.TestCase):
    def test_transform_before_fit_rejected(self):
        with self.assertRaisesRegex(PreprocessError, "not fitted"):
            Scaler().transform(to_grid([0] * 16))

    def test_heldout_transform_does_not_change_fitted_state(self):
        s = fitted()
        before = s.state()
        heldout = [to_grid(r["pixels"]) for r in fixture.FIXTURE if split_of(r) != "train"]
        self.assertEqual(len(heldout), 4)
        s.transform_batch(heldout)
        self.assertEqual(s.state(), before)
        self.assertEqual(before[0], frozenset({"r0", "r1"}))

    def test_public_interface_has_no_fit_on_transform(self):
        public = {n for n in dir(Scaler) if not n.startswith("_") and n != "PUBLIC"}
        self.assertEqual(public, set(Scaler.PUBLIC))
        self.assertFalse(hasattr(Scaler, "fit_transform"))

    def test_contamination_fixture_rejected(self):
        for split in ("validation", "test"):
            held = [r for r in fixture.FIXTURE if split_of(r) == split]
            with self.assertRaises(ContaminationError):
                Scaler().fit(held)

    def test_mixed_train_and_test_rejected_and_state_untouched(self):
        s = Scaler()
        with self.assertRaises(ContaminationError):
            s.fit(fixture.FIXTURE)  # includes validation and test records
        self.assertIsNone(s.state()[0])  # nothing was fitted
        with self.assertRaisesRegex(PreprocessError, "not fitted"):
            s.transform(to_grid([0] * 16))

    def test_refit_with_contamination_keeps_previous_state(self):
        s = fitted()
        before = s.state()
        with self.assertRaises(ContaminationError):
            s.fit([fixture.FIXTURE[4]])
        self.assertEqual(s.state(), before)


class ServingPayload(unittest.TestCase):
    def test_rest_request_body_offline(self):
        s = fitted()
        imgs = s.transform_batch([to_grid(r["pixels"]) for r in fixture.FIXTURE])
        body = json.loads(serving_request(imgs))
        self.assertEqual(body["signature_name"], "serving_default")
        self.assertEqual(len(body["instances"]), 6)
        inst = body["instances"][1]
        self.assertEqual((len(inst), len(inst[0]), len(inst[0][0])), (4, 4, 1))
        self.assertEqual(inst[3][3], [1.0])


if __name__ == "__main__":
    unittest.main()
