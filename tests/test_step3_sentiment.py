"""
Unit and integration tests for Step 3: Spark NLP Multilingual Sentiment Analysis & Temporal Tracking.
"""

import unittest
from sentiment_analysis.sentiment_analyzer import MultilingualSentimentAnalyzer


class TestStep3SentimentAnalysis(unittest.TestCase):
    """Verifies multilingual sentiment scoring and temporal trend tracking."""

    def setUp(self):
        self.analyzer = MultilingualSentimentAnalyzer()

    def test_positive_sentiment_english(self):
        text = "This breakthrough in artificial intelligence is truly astonishing and wonderful! 😊"
        res = self.analyzer.analyze_record(text, lang="en")
        self.assertEqual(res["sentiment_label"], "positive")
        self.assertGreater(res["sentiment_score"], 0)

    def test_negative_sentiment_english(self):
        text = "Beware of dangerous misinformation and devastating security breaches. Terrible crisis! 😱"
        res = self.analyzer.analyze_record(text, lang="en")
        self.assertEqual(res["sentiment_label"], "negative")
        self.assertLess(res["sentiment_score"], 0)

    def test_multilingual_sentiment_spanish(self):
        text = "Los avances son una maravilla increíble y tienen un éxito fantástico."
        res = self.analyzer.analyze_record(text, lang="es")
        self.assertEqual(res["sentiment_label"], "positive")
        self.assertGreater(res["sentiment_score"], 0)

    def test_tracking_over_time(self):
        sample_posts = [
            {"created_utc": 1700000000, "sentiment_score": 0.8, "sentiment_label": "positive"},
            {"created_utc": 1700000000 + 3600, "sentiment_score": -0.6, "sentiment_label": "negative"},
            {"created_utc": 1700000000 + 7200, "sentiment_score": 0.5, "sentiment_label": "positive"},
            {"created_utc": 1700000000 + 86400, "sentiment_score": -0.7, "sentiment_label": "negative"},
        ]
        trends = self.analyzer.track_sentiment_over_time(sample_posts, interval_hours=12)
        self.assertGreaterEqual(len(trends), 2)
        for t in trends:
            self.assertIn("avg_sentiment_score", t)
            self.assertIn("positive_ratio", t)
            self.assertIn("net_sentiment_index", t)


if __name__ == "__main__":
    unittest.main()
