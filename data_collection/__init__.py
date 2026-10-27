"""
Data Collection Package for Reddit and YouTube APIs.
Step 1: Collect data with Reddit and YouTube APIs (respect rate limits and terms of use). Store raw JSON.
"""

from .reddit_collector import RedditCollector
from .youtube_collector import YouTubeCollector

__all__ = ["RedditCollector", "YouTubeCollector"]
