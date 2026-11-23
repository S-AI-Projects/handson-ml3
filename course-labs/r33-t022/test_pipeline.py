"""
Unit tests for Course Lab R33-T022 Majority-Class Baseline.
Offline, CPU-only bounded verification.
"""

import json, os, unittest, importlib.util

spec = importlib.util.spec_from_file_location("pipeline", os.path.join(os.path.dirname(__file__), "pipeline.py"))
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)
MajorityClassBaseline = pipeline.MajorityClassBaseline
stratified_split = pipeline.stratified_split

class TestMajorityClassBaseline(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), "fixture.json")) as f:
            self.fixture = json.load(f)

    def test_fixture_integrity(self):
        """Asserts 8 synthetic rows fixture with x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(self.fixture), 8)
        self.assertEqual([r["x1"] for r in self.fixture], list(range(8)))
        self.assertTrue(all(r["x2"] == 1 for r in self.fixture))
        self.assertEqual([r["label"] for r in self.fixture], [0,0,0,0,1,1,1,1])

    def test_stratified_split_balance(self):
        """Asserts stratified split maintains class balance in train and held-out test splits."""
        train, test = stratified_split(self.fixture, train_ratio=0.5)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 4)
        self.assertEqual([r["label"] for r in train].count(0), 2)
        self.assertEqual([r["label"] for r in train].count(1), 2)
        self.assertEqual([r["label"] for r in test].count(0), 2)
        self.assertEqual([r["label"] for r in test].count(1), 2)

    def test_baseline_learned_only_from_train_data(self):
        """Asserts baseline parameters use only training data."""
        # Biased train data: 3 zeros, 1 one
        biased_train = [self.fixture[0], self.fixture[1], self.fixture[2], self.fixture[4]]
        model = MajorityClassBaseline().fit(biased_train)
        self.assertEqual(model.majority_class, 0)

        # Held-out test evaluation
        held_out_test = [self.fixture[3], self.fixture[5], self.fixture[6], self.fixture[7]]
        preds = model.predict(held_out_test)
        self.assertEqual(preds, [0, 0, 0, 0])
        metrics = model.evaluate(held_out_test, preds)
        # held_out_test has labels [0, 1, 1, 1], so accuracy with all 0s is 1/4 = 0.25
        self.assertEqual(metrics["accuracy"], 0.25)

    def test_predict_without_fit_fails(self):
        """Asserts calling predict before fit raises RuntimeError."""
        model = MajorityClassBaseline()
        with self.assertRaises(RuntimeError):
            model.predict(self.fixture)

    def test_deliberate_mutation_failure(self):
        """Asserts that deliberately changing expected baseline evaluation fails."""
        train, test = stratified_split(self.fixture, train_ratio=0.5)
        model = MajorityClassBaseline().fit(train)
        preds = model.predict(test)
        metrics = model.evaluate(test, preds)
        with self.assertRaises(AssertionError):
            self.assertEqual(metrics["accuracy"], 0.99)

if __name__ == "__main__":
    unittest.main()
