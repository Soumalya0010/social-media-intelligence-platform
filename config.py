"""
Configuration module for Large-Scale Social Media Intelligence Platform.
Manages credentials, file system paths, and runtime settings.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REDDIT_RAW_DIR = RAW_DATA_DIR / "reddit"
YOUTUBE_RAW_DIR = RAW_DATA_DIR / "youtube"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they do not exist
for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    REDDIT_RAW_DIR,
    YOUTUBE_RAW_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Step 1: Reddit API Configuration
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv(
    "REDDIT_USER_AGENT", "SocialMediaIntelligencePlatform/1.0"
)
DEFAULT_SUBREDDITS = os.getenv(
    "DEFAULT_SUBREDDITS", "technology,datascience,machinelearning,worldnews"
).split(",")

# Step 1: YouTube Data API v3 Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
DEFAULT_YOUTUBE_QUERIES = os.getenv(
    "DEFAULT_YOUTUBE_QUERIES",
    "artificial intelligence,data science,social media analytics",
).split(",")

# Step 5: Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "intelligence_password")

# Step 7: Elasticsearch & Kibana Configuration
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_USER = os.getenv("ELASTICSEARCH_USER", "")
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "social-intelligence-posts")
KIBANA_URL = os.getenv("KIBANA_URL", "http://localhost:5601")

# Preprocessing & Processing Defaults
DEFAULT_MAX_ITEMS = int(os.getenv("MAX_ITEMS", "250"))
