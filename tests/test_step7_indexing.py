"""
Unit and integration tests for Step 7: Elasticsearch Indexing & Kibana Dashboard Objects.
"""

import json
from pathlib import Path
import unittest
from indexing_and_dashboard.elasticsearch_indexer import ElasticsearchIndexer


class TestStep7Indexing(unittest.TestCase):
    """Verifies document formatting, Elasticsearch mapping, and NDJSON bulk export."""

    def setUp(self):
        self.indexer = ElasticsearchIndexer()
        self.sample_posts = [
            {
                "id": "post_1",
                "source": "reddit",
                "author": "tech_guru",
                "clean_text": "Groundbreaking discoveries in large-scale transformer computing.",
                "created_utc": 1700000000,
                "sentiment_label": "positive",
                "sentiment_score": 0.85,
                "topic_id": 1,
                "topic_name": "AI Advances",
                "author_influencer_rank": 1,
                "author_pagerank": 0.42,
                "author_community_id": 0,
                "misinformation_score": 0.05,
                "misinformation_label": "credible",
            }
        ]

    def test_format_document(self):
        doc = self.indexer.format_document(self.sample_posts[0])
        self.assertEqual(doc["id"], "post_1")
        self.assertIn("timestamp", doc)
        self.assertEqual(doc["sentiment_label"], "positive")
        self.assertEqual(doc["author_influencer_rank"], 1)

    def test_bulk_index_and_ndjson_generation(self):
        res = self.indexer.bulk_index(self.sample_posts)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_documents"], 1)

        ndjson_file = Path(res["ndjson_file"])
        self.assertTrue(ndjson_file.exists())

        with open(ndjson_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), 2)  # 1 action line + 1 data line

    def test_kibana_dashboard_json_validity(self):
        kibana_file = Path(__file__).resolve().parent.parent / "indexing_and_dashboard" / "kibana_dashboard.json"
        self.assertTrue(kibana_file.exists())
        with open(kibana_file, "r", encoding="utf-8") as f:
            kibana_data = json.load(f)
        self.assertIn("objects", kibana_data)
        self.assertGreater(len(kibana_data["objects"]), 3)


if __name__ == "__main__":
    unittest.main()
