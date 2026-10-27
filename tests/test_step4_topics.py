"""
Unit and integration tests for Step 4: BERTopic Topic Modeling & Dynamic Trajectories.
"""

from datetime import datetime, timezone
import unittest
from topic_modeling.topic_modeler import BERTopicModeler


class TestStep4TopicModeling(unittest.TestCase):
    """Verifies topic fitting, keywords extraction, and topics_over_time dynamic modeling."""

    def setUp(self):
        self.modeler = BERTopicModeler()
        self.docs = [
            "Deep neural networks and transformer architecture advancements in natural language processing.",
            "State of generative AI models and automated reasoning systems.",
            "Global financial market fluctuations, stock index changes, and economic inflation policies.",
            "Interest rates, central bank updates, and currency market analysis.",
            "Cybersecurity threat intelligence, database vulnerability patches, and encryption protocols.",
            "Zero day malware exploits, phishing attacks, and endpoint security detection.",
        ]
        self.timestamps = [
            datetime(2026, 10, 20 + i, 12, 0, tzinfo=timezone.utc) for i in range(len(self.docs))
        ]

    def test_fit_transform(self):
        topics, probs, topic_info = self.modeler.fit_transform(self.docs, self.timestamps)
        self.assertEqual(len(topics), len(self.docs))
        self.assertEqual(len(probs), len(self.docs))
        self.assertGreater(len(topic_info), 0)

    def test_topics_over_time(self):
        topics, probs, _ = self.modeler.fit_transform(self.docs, self.timestamps)
        tot = self.modeler.calculate_topics_over_time(self.docs, topics, self.timestamps, nr_bins=2)
        self.assertGreater(len(tot), 0)
        for item in tot:
            self.assertIn("Topic", item)
            self.assertIn("Frequency", item)
            self.assertIn("trend_status", item)


if __name__ == "__main__":
    unittest.main()
