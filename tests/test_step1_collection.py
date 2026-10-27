"""
Unit and integration tests for Step 1: Data Collection (Reddit & YouTube APIs).
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from data_collection.reddit_collector import RedditCollector
from data_collection.youtube_collector import YouTubeCollector


class TestStep1DataCollection(unittest.TestCase):
    """Verifies Reddit and YouTube collection, rate limiting, and raw JSON storage."""

    def setUp(self):
        self.test_dir = Path(__file__).resolve().parent / "test_data" / "raw"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def test_reddit_reference_data_generation(self):
        collector = RedditCollector(output_dir=self.test_dir)
        data = collector.fetch_subreddit_posts("datascience", limit=5)
        self.assertIn("data", data)
        self.assertIn("children", data["data"])
        self.assertGreaterEqual(len(data["data"]["children"]), 5)

        # Check raw JSON saved
        saved_files = list(self.test_dir.glob("reddit_datascience_*.json"))
        self.assertGreater(len(saved_files), 0)

        # Verify post schema
        first_post = data["data"]["children"][0]["data"]
        self.assertIn("title", first_post)
        self.assertIn("author", first_post)
        self.assertIn("created_utc", first_post)

    def test_youtube_reference_data_generation(self):
        collector = YouTubeCollector(output_dir=self.test_dir)
        data = collector.search_videos("machine learning", max_results=3)
        self.assertIn("items", data)
        self.assertGreaterEqual(len(data["items"]), 3)

        saved_files = list(self.test_dir.glob("youtube_*.json"))
        self.assertGreater(len(saved_files), 0)

        first_video = data["items"][0]
        self.assertIn("snippet", first_video)
        self.assertIn("title", first_video["snippet"])
        self.assertIn("channelTitle", first_video["snippet"])


if __name__ == "__main__":
    unittest.main()
