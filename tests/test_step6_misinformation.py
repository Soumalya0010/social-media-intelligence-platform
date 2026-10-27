"""
Unit and integration tests for Step 6: Misinformation / Fake-News Classifier on LIAR Dataset.
"""

import unittest
from misinformation_detection.misinformation_classifier import MisinformationClassifier


class TestStep6MisinformationClassifier(unittest.TestCase):
    """Verifies LIAR training, metrics calculation, and inference scoring."""

    def setUp(self):
        self.classifier = MisinformationClassifier()

    def test_train_and_evaluate_liar(self):
        metrics = self.classifier.train()
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.70)

    def test_predict_fake_news(self):
        fake_claim = "Drinking bleach cures respiratory viral infections and gives 5G magnetic powers."
        pred = self.classifier.predict(fake_claim)
        self.assertIn("misinformation_score", pred)
        self.assertIn("misinformation_label", pred)
        self.assertGreater(pred["misinformation_score"], 0.40)

    def test_predict_credible_news(self):
        real_claim = "The federal minimum wage has remained at 7.25 dollars per hour since 2009."
        pred = self.classifier.predict(real_claim)
        self.assertIn("misinformation_score", pred)
        self.assertLess(pred["misinformation_score"], 0.50)


if __name__ == "__main__":
    unittest.main()
