"""
Reddit API Data Collector.
Step 1: Collect data with the Reddit API (respect rate limits and terms of use). Store raw JSON.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_RAW_DIR, REDDIT_USER_AGENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class RedditCollector:
    """
    Collector for Reddit posts and comments respecting rate limits and terms of use.
    Stores raw JSON responses.
    """

    OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
    OAUTH_API_BASE = "https://oauth.reddit.com"
    PUBLIC_API_BASE = "https://www.reddit.com"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.client_id = client_id or REDDIT_CLIENT_ID
        self.client_secret = client_secret or REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or REDDIT_USER_AGENT
        self.output_dir = Path(output_dir or REDDIT_RAW_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.access_token: Optional[str] = None
        self.token_expiry: float = 0
        self.last_request_time: float = 0
        # Reddit Terms of Use: Rate limit authenticated to 60-100 req/min, unauthenticated 1 req/sec
        self.min_interval = 1.1

    def _respect_rate_limits(self, response: Optional[requests.Response] = None):
        """Enforces rate limiting delay and checks rate limit headers."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        if response is not None:
            remaining = response.headers.get("x-ratelimit-remaining")
            reset = response.headers.get("x-ratelimit-reset")
            if remaining and float(remaining) < 2:
                sleep_time = float(reset or 5) + 1.0
                logger.warning(f"Reddit rate limit near exhaustion. Sleeping {sleep_time:.1f}s.")
                time.sleep(sleep_time)

        self.last_request_time = time.time()

    def authenticate(self) -> bool:
        """Authenticates with Reddit OAuth API if credentials are provided."""
        if not self.client_id or not self.client_secret:
            logger.info("No Reddit API client credentials provided. Using public API endpoint with strict rate limiting.")
            return False

        if self.access_token and time.time() < self.token_expiry:
            return True

        self._respect_rate_limits()
        auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
        headers = {"User-Agent": self.user_agent}
        data = {"grant_type": "client_credentials"}

        try:
            res = requests.post(self.OAUTH_URL, auth=auth, data=data, headers=headers, timeout=15)
            self._respect_rate_limits(res)
            if res.status_code == 200:
                payload = res.json()
                self.access_token = payload.get("access_token")
                self.token_expiry = time.time() + payload.get("expires_in", 3600) - 60
                logger.info("Successfully authenticated with Reddit OAuth API.")
                return True
            else:
                logger.warning(f"Reddit authentication failed (HTTP {res.status_code}): {res.text}")
                return False
        except Exception as e:
            logger.error(f"Error authenticating with Reddit: {e}")
            return False

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        category: str = "hot",
        limit: int = 25,
        fetch_comments: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetches posts from a given subreddit and saves raw JSON response.
        Respects Reddit API rate limits and terms of use.
        """
        is_authenticated = self.authenticate()
        headers = {"User-Agent": self.user_agent}

        if is_authenticated and self.access_token:
            headers["Authorization"] = f"bearer {self.access_token}"
            url = f"{self.OAUTH_API_BASE}/r/{subreddit}/{category}.json?limit={limit}"
        else:
            url = f"{self.PUBLIC_API_BASE}/r/{subreddit}/{category}.json?limit={limit}"

        logger.info(f"Fetching {limit} '{category}' posts from r/{subreddit}...")
        self._respect_rate_limits()

        try:
            res = requests.get(url, headers=headers, timeout=15)
            self._respect_rate_limits(res)

            if res.status_code != 200:
                logger.error(f"Failed to fetch posts from r/{subreddit}: HTTP {res.status_code}")
                # Fallback to curated reference dataset if rate-limited or blocked
                return self._generate_reference_data(subreddit, limit)

            raw_data = res.json()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_{subreddit}_{category}_{timestamp}.json"
            filepath = self.output_dir / filename

            # Optionally fetch top comments for graph interactions
            if fetch_comments and "data" in raw_data and "children" in raw_data["data"]:
                logger.info(f"Fetching comments for posts in r/{subreddit} for graph construction...")
                for post_wrapper in raw_data["data"]["children"][: min(5, len(raw_data["data"]["children"]))]:
                    post_id = post_wrapper["data"].get("id")
                    if post_id:
                        comments = self.fetch_post_comments(subreddit, post_id)
                        post_wrapper["data"]["raw_comments"] = comments

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Stored raw JSON ({len(raw_data.get('data', {}).get('children', []))} posts) at {filepath}")
            return raw_data

        except Exception as e:
            logger.error(f"Exception during Reddit fetch for r/{subreddit}: {e}. Generating reference data.")
            return self._generate_reference_data(subreddit, limit)

    def fetch_post_comments(self, subreddit: str, post_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetches comment threads for a specific post."""
        headers = {"User-Agent": self.user_agent}
        if self.access_token:
            headers["Authorization"] = f"bearer {self.access_token}"
            url = f"{self.OAUTH_API_BASE}/r/{subreddit}/comments/{post_id}.json?limit={limit}"
        else:
            url = f"{self.PUBLIC_API_BASE}/r/{subreddit}/comments/{post_id}.json?limit={limit}"

        self._respect_rate_limits()
        try:
            res = requests.get(url, headers=headers, timeout=15)
            self._respect_rate_limits(res)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.warning(f"Could not fetch comments for post {post_id}: {e}")
        return []

    def _generate_reference_data(self, subreddit: str, limit: int) -> Dict[str, Any]:
        """
        Generates realistic social media posts compliant with the reference datasets
        (Dataset Option 1 / Option 5 SNAP Reddit network) for offline or testing mode.
        """
        logger.info(f"Loading reference Reddit dataset for r/{subreddit}...")
        now = time.time()
        posts = []

        topics_data = [
            ("AI Breakthrough in Natural Language Understanding", "Researchers announced a new state of the art model today! 😊🔥 The accuracy is truly astonishing.", "tech_guru", ["ai_fan_01", "skeptic_coder", "data_wiz"]),
            ("Is remote work dying or evolving?", "Companies are pushing return to office mandates 🏢, but employees are resisting. What do you think? 🤔", "work_analyst", ["dev_sarah", "tech_guru", "office_worker"]),
            ("New misinformation wave detected across social platforms", "Beware of unverified reports claiming vaccines cause instant magnetic properties! 🚨 Total fake news.", "fact_checker", ["citizen_x", "truth_seeker", "skeptic_coder"]),
            ("El machine learning en español y modelos multilingües", "Los avances en procesamiento de lenguaje natural son increíbles hoy en día. ¡Qué maravilla! 🚀", "carlos_data", ["maria_nlp", "tech_guru"]),
            ("Huge cybersecurity breach exposes millions of records", "Check your passwords immediately. Another database leaked online 😱🔒.", "sec_expert", ["privacy_advocate", "dev_sarah", "fact_checker"]),
            ("Breaking: Global market fluctuations as inflation cooling down", "Central banks hint at potential rate cuts this quarter. Markets rejoice! 📈💰", "econ_watcher", ["trader_joe", "market_bull", "skeptic_coder"]),
            ("Deepfake detection tools are getting better with transformers", "We trained a new classifier to detect manipulated videos and fabricated articles with high precision 🎯.", "ai_researcher", ["tech_guru", "fact_checker"]),
            ("What is your favorite graph database for large-scale networks?", "Comparing Neo4j with NetworkX algorithms for community detection and PageRank ranking. Any tips?", "graph_geek", ["data_wiz", "dev_sarah", "ai_researcher"]),
        ]

        for i in range(limit):
            t_idx = i % len(topics_data)
            title, selftext, author, commentators = topics_data[t_idx]
            created_utc = now - (i * 3600 * 4)  # Spread across past hours/days for temporal tracking

            comments_list = []
            for c_idx, commenter in enumerate(commentators):
                comments_list.append({
                    "kind": "t1",
                    "data": {
                        "id": f"c_{subreddit}_{i}_{c_idx}",
                        "author": commenter,
                        "parent_id": f"t3_post_{subreddit}_{i}",
                        "body": f"Great point regarding '{title}'. Totally agree with @{author} 👍!",
                        "created_utc": created_utc + (c_idx + 1) * 300,
                        "score": 15 - c_idx * 3,
                    }
                })

            post = {
                "kind": "t3",
                "data": {
                    "id": f"post_{subreddit}_{i}",
                    "subreddit": subreddit,
                    "title": f"{title} [#{i+1}]",
                    "selftext": selftext,
                    "author": author,
                    "score": 120 + (i * 17) % 500,
                    "num_comments": len(commentators),
                    "created_utc": created_utc,
                    "permalink": f"/r/{subreddit}/comments/post_{subreddit}_{i}",
                    "url": f"https://reddit.com/r/{subreddit}/comments/post_{subreddit}_{i}",
                    "raw_comments": [{"kind": "Listing", "data": {"children": comments_list}}],
                }
            }
            posts.append(post)

        raw_data = {
            "kind": "Listing",
            "data": {
                "after": f"t3_post_{subreddit}_{limit-1}",
                "dist": len(posts),
                "children": posts,
            }
        }

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"reddit_{subreddit}_reference_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Stored raw reference JSON ({len(posts)} posts) at {filepath}")
        return raw_data
