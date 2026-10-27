# API & Data Schema Documentation

## 1. Environment & API Credentials

### Reddit API Configuration
- **Endpoint**: `https://oauth.reddit.com/r/{subreddit}/{category}.json`
- **Rate Limit**: Max 60-100 requests per minute with active OAuth token.
- **Headers**:
  ```http
  User-Agent: SocialMediaIntelligencePlatform/1.0 (by /u/YourUsername)
  Authorization: bearer <access_token>
  ```
- **Environment Variables**:
  - `REDDIT_CLIENT_ID`: OAuth application client ID
  - `REDDIT_CLIENT_SECRET`: OAuth application secret key
  - `REDDIT_USER_AGENT`: Custom identification string required by Reddit terms of use

### YouTube Data API v3 Configuration
- **Endpoints**:
  - Search: `https://www.googleapis.com/youtube/v3/search`
  - Video Statistics: `https://www.googleapis.com/youtube/v3/videos`
  - Comments: `https://www.googleapis.com/youtube/v3/commentThreads`
- **Quota**: 10,000 units/day default allocation (search = 100 units, videos = 1 unit).
- **Environment Variables**:
  - `YOUTUBE_API_KEY`: Google Cloud developer API key

---

## 2. Processed Post Schema

Every processed record in `data/processed/` adheres to the following unified JSON schema:

```json
{
  "id": "reddit_post_technology_0",
  "source": "reddit",
  "type": "post",
  "parent_id": null,
  "author": "tech_guru",
  "created_utc": 1700000000.0,
  "iso_timestamp": "2026-10-20T12:00:00+00:00",
  "raw_text": "AI Breakthrough in Natural Language Understanding. Researchers announced...",
  "clean_text": "AI Breakthrough in Natural Language Understanding. Researchers announced...",
  "emojis_detected": ["😊", "🔥"],
  "emoji_count": 2,
  "detected_language": "en",
  "language_confidence": 0.999,
  "score": 340,
  "views": 0,
  "num_comments": 14,
  "url": "https://reddit.com/r/technology/comments/...",
  "sentiment_label": "positive",
  "sentiment_score": 0.825,
  "sentiment_confidence": 0.912,
  "topic_id": 1,
  "topic_name": "Topic_1_ai_models_breakthrough",
  "topic_probability": 0.941,
  "author_influencer_rank": 1,
  "author_pagerank": 0.38421,
  "author_community_id": 0,
  "misinformation_score": 0.082,
  "misinformation_label": "credible"
}
```

---

## 3. Graph Database Schema (Neo4j)

### Node Labels
- `(:User)`
  - `username` (String, Unique Primary Key)
  - `source` (String: "reddit", "youtube")
  - `pagerank` (Float: NetworkX PageRank score)
  - `influencer_rank` (Integer: Global leaderboard position)
  - `community_id` (Integer: Partition cluster identifier)
- `(:Post)`
  - `id` (String, Unique Primary Key)
  - `title` (String: Snippet of cleaned text)
  - `score` (Integer: Engagement score)
  - `source` (String)

### Relationship Types
- `(:User)-[:AUTHORED]->(:Post)`
- `(:User)-[:REPLIED_TO]->(:User)`
- `(:User)-[:MENTIONED]->(:User)`
- `(:User)-[:INTERACTED_WITH {weight: Float}]->(:User)`

---

## 4. Elasticsearch Index Mapping

- **Index Name**: `social-intelligence-posts`
- **Primary Timestamp Field**: `timestamp` (ISO 8601 `strict_date_optional_time`)
- **Key Aggregations**:
  - `terms` aggregation on `topic_name` for topic distributions
  - `date_histogram` on `timestamp` for temporal trends
  - `avg` on `sentiment_score` for dynamic sentiment shifts
  - `range` filter on `misinformation_score` for automated alert triggers
