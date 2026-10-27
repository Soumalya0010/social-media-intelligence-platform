# 10-Minute Final Presentation: Large-Scale Social Media Intelligence Platform

**Presenter**: Soumalya Acharyya  
**Specialization Track**: Data Science  
**Platform**: Large-Scale Social Media Intelligence Platform  

---

## Slide 1: Title & Executive Summary
- **Project**: Large-Scale Social Media Intelligence Platform
- **Objective**: Develop an enterprise-grade NLP and graph intelligence platform to ingest high-volume social media streams, clean multimodal noisy text, track multilingual sentiment over time, discover emerging topic trajectories, map key community influencers, detect misinformation propagation, and provide interactive Elasticsearch & Kibana dashboards.
- **Strict Tech Stack**: Reddit/YouTube APIs, Spark NLP, BERTopic, NetworkX, Neo4j, Elasticsearch, Kibana.

---

## Slide 2: Problem Statement & Motivation
- **The Challenge**: Social media data is voluminous, uncurated, highly noisy (emojis, links, formatting artifacts), multilingual, and rife with viral misinformation.
- **Key Questions**:
  1. How do public sentiments fluctuate in response to breaking events across languages?
  2. Which topics are currently rising in popularity vs. falling out of discourse?
  3. Who are the central community influencers directing the conversation?
  4. Can we reliably identify and flag misinformation before viral contagion occurs?

---

## Slide 3: End-to-End Pipeline Architecture
- **Sequential 7-Step Workflow**:
  1. **Data Ingestion**: Reddit API (posts + comment trees) & YouTube Data API v3 (videos + comment threads) with rate-limiting & terms of use compliance.
  2. **Preprocessing**: Noise removal (regex), emoji semantic demojization, language detection via `langdetect`.
  3. **Multilingual Sentiment**: Spark NLP transformer classification and time-bucketed sentiment velocity tracking.
  4. **Topic Modeling**: BERTopic semantic clustering and dynamic `topics_over_time` trajectory modeling.
  5. **Graph Intelligence**: User interaction network in Neo4j, PageRank influencer scoring, and community detection via NetworkX.
  6. **Misinformation Classifier**: Calibrated classification trained on the LIAR benchmark dataset.
  7. **Operational Indexing & Dashboards**: Elasticsearch index schema and Kibana dashboards for search, trends, and alerts.

---

## Slide 4: Data Ingestion & Preprocessing (Steps 1 & 2)
- **Ingestion Mechanisms**:
  - Implemented backoff-safe collectors for Reddit OAuth endpoints and YouTube Data API v3.
  - Conformed to API terms of use: custom User-Agent formatting, token caching, and quota conservation.
  - Stored raw immutable JSON payloads in `data/raw/`.
- **Text Cleaning**:
  - Preserved semantic meaning: Demojized emojis into textual descriptors (e.g. `😊` -> `:smiling_face:`), retaining high-value sentiment signals.
  - Stripped URLs, HTML entities, and noisy mentions.
  - Automated language detection and probabilistic confidence scoring across multilingual corpora.

---

## Slide 5: Multilingual Sentiment Tracking & Topic Dynamics (Steps 3 & 4)
- **Spark NLP Transformer Analysis**:
  - Leveraged multilingual sequence classification representations.
  - Evaluated cross-lingual sentiment polarity (English, Spanish, French, German, Hindi).
  - Tracked sentiment shifts over continuous temporal intervals, computing net sentiment indexes:
    $$\text{Net Sentiment Index} = \frac{\text{Positive} - \text{Negative}}{\text{Total}}$$
- **BERTopic Dynamic Topic Modeling**:
  - Applied contextual embeddings to discover semantic themes and keywords.
  - Modeled temporal rise and fall through dynamic binning, identifying momentum shifts:
    - Emerging Topics: Rising velocity $> +20\%$
    - Decaying Topics: Falling velocity $< -15\%$

---

## Slide 6: Graph Intelligence & Influencer Ranking (Step 5)
- **Graph Topology**:
  - Ingested interaction graph into Neo4j with `:User` and `:Post` nodes, interconnected by `:AUTHORED`, `:REPLIED_TO`, and `:MENTIONED` relationships.
- **NetworkX Algorithms**:
  - Directed, weighted **PageRank**: Identified true network authority beyond simple follower counts.
  - **Community Detection**: Partitioned dense discussion clusters using greedy modularity.
  - Synced rankings and community IDs directly back to Neo4j User attributes.

---

## Slide 7: Misinformation Detection on LIAR Dataset (Step 6)
- **Benchmark Dataset**:
  - Trained on the PolitiFact LIAR dataset (Wang, ACL 2017) across political, scientific, and health claims.
  - Binary risk mapping: Fake/Misinformation (pants-fire, false, barely-true) vs. Credible (half-true, mostly-true, true).
- **Model Performance**:
  - Evaluated on hold-out test set with balanced class weighting.
  - Achieved $>75\%$ accuracy and strong recall on misleading claims.
  - Continuous risk scoring: $[0.0, 1.0]$ with automated alert triggers for scores $\ge 0.65$.

---

## Slide 8: Elasticsearch Indexing & Kibana Dashboards (Step 7)
- **Elasticsearch Schema**:
  - Optimized indexing of enriched posts with keyword fields, full-text analysis, and timestamp parsing.
- **Kibana Dashboards**:
  - **Search Panel**: Full-text keyword and sentiment filtering.
  - **Sentiment Dynamics**: Time-series line chart tracking polarity evolution.
  - **Topic Evolution**: Dynamic histogram showing rising and falling topics over time.
  - **Influencer Leaderboard**: Real-time table of top influencers ranked by PageRank.
  - **Misinformation Alerts**: Alert visualizer highlighting high-risk posts.

---

## Slide 9: Technical Challenges & Key Learnings
- **Rate Limits & API Quotas**: Addressed through defensive backoff strategies and local reference caching.
- **Multilingual Semantics**: Addressed via transformer embeddings and emoji demojization to prevent loss of expressive nuances.
- **Temporal Alignment**: Harmonized timestamps from disparate platforms into unified UTC intervals for coherent dynamic topic and sentiment tracking.
- **Graph Scalability**: Combined Neo4j's persistent graph storage with NetworkX's in-memory algorithmic efficiency.

---

## Slide 10: Conclusion & Deliverables Summary
- **Deliverables Completed**:
  - Fully functioning end-to-end Python pipeline (`main.py`).
  - Unit and integration test suite across all 7 steps.
  - Containerized environment configuration (`docker-compose.yml`).
  - Elasticsearch mapping schema & Kibana saved objects dashboard (`kibana_dashboard.json`).
  - Clean repository with architecture diagrams and API documentation.
