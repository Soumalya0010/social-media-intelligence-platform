"""
Collector Runner for Step 1.
Orchestrates collection from Reddit and YouTube APIs and stores raw JSON.
"""

import argparse
import logging
from typing import Dict, Any, List

from config import DEFAULT_SUBREDDITS, DEFAULT_YOUTUBE_QUERIES
from data_collection.reddit_collector import RedditCollector
from data_collection.youtube_collector import YouTubeCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_collection(
    sources: List[str] = ["reddit", "youtube"],
    subreddits: List[str] = DEFAULT_SUBREDDITS,
    youtube_queries: List[str] = DEFAULT_YOUTUBE_QUERIES,
    limit: int = 15,
) -> Dict[str, Any]:
    """
    Executes Step 1 of the platform:
    Collect data with Reddit and YouTube APIs, storing raw JSON.
    """
    results: Dict[str, Any] = {"reddit": [], "youtube": []}

    if "reddit" in sources:
        collector = RedditCollector()
        for sub in subreddits:
            try:
                data = collector.fetch_subreddit_posts(sub.strip(), limit=limit)
                results["reddit"].append({"subreddit": sub.strip(), "count": len(data.get("data", {}).get("children", []))})
            except Exception as e:
                logger.error(f"Error collecting from Reddit r/{sub}: {e}")

    if "youtube" in sources:
        yt_collector = YouTubeCollector()
        for q in youtube_queries:
            try:
                data = yt_collector.search_videos(q.strip(), max_results=limit)
                results["youtube"].append({"query": q.strip(), "count": len(data.get("items", []))})
            except Exception as e:
                logger.error(f"Error collecting from YouTube for '{q}': {e}")

    logger.info("Step 1 data collection completed successfully.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data from Reddit and YouTube APIs (Step 1).")
    parser.add_argument("--sources", nargs="+", default=["reddit", "youtube"], help="Sources to collect from")
    parser.add_argument("--limit", type=int, default=15, help="Limit items per source")
    args = parser.parse_args()

    run_collection(sources=args.sources, limit=args.limit)
