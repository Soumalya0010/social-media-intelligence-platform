"""
Elasticsearch Indexer.
Step 7: Index everything in Elasticsearch.
Required Tech Stack: Elasticsearch
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    DATA_DIR,
    ELASTICSEARCH_INDEX,
    ELASTICSEARCH_PASSWORD,
    ELASTICSEARCH_URL,
    ELASTICSEARCH_USER,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ElasticsearchIndexer:
    """
    Manages index creation, mapping definitions, and bulk indexing of enriched records
    into Elasticsearch. Also generates NDJSON bulk ingestion payloads for offline environments.
    """

    INDEX_MAPPING = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "type": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "author": {"type": "keyword"},
                "created_utc": {"type": "double"},
                "timestamp": {"type": "date"},
                "raw_text": {"type": "text"},
                "clean_text": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "detected_language": {"type": "keyword"},
                "language_confidence": {"type": "float"},
                "emojis_detected": {"type": "keyword"},
                "emoji_count": {"type": "integer"},
                "score": {"type": "integer"},
                "views": {"type": "long"},
                "num_comments": {"type": "integer"},
                "url": {"type": "keyword"},
                # Step 3: Sentiment fields
                "sentiment_label": {"type": "keyword"},
                "sentiment_score": {"type": "float"},
                "sentiment_confidence": {"type": "float"},
                # Step 4: Topic fields
                "topic_id": {"type": "integer"},
                "topic_name": {"type": "keyword"},
                "topic_probability": {"type": "float"},
                # Step 5: Graph / Influencer fields
                "author_influencer_rank": {"type": "integer"},
                "author_pagerank": {"type": "float"},
                "author_community_id": {"type": "integer"},
                # Step 6: Misinformation fields
                "misinformation_score": {"type": "float"},
                "misinformation_label": {"type": "keyword"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
    }

    def __init__(
        self,
        hosts: str = ELASTICSEARCH_URL,
        user: str = ELASTICSEARCH_USER,
        password: str = ELASTICSEARCH_PASSWORD,
        index_name: str = ELASTICSEARCH_INDEX,
    ):
        self.hosts = hosts
        self.user = user
        self.password = password
        self.index_name = index_name
        self.es = None
        self.is_connected = False
        self._connect()

    def _connect(self):
        """Attempts connection to Elasticsearch cluster."""
        try:
            from elasticsearch import Elasticsearch

            auth = (self.user, self.password) if self.user and self.password else None
            self.es = Elasticsearch(
                hosts=[self.hosts],
                basic_auth=auth,
                request_timeout=1,
                max_retries=0,
                retry_on_timeout=False,
            )
            if self.es.ping():
                self.is_connected = True
                logger.info(f"Connected to Elasticsearch cluster at {self.hosts}")
            else:
                logger.warning(f"Elasticsearch ping failed at {self.hosts}.")
                self.is_connected = False
        except Exception as e:
            logger.warning(
                f"Elasticsearch cluster unreachable at {self.hosts} ({e}). "
                "Enabling NDJSON Bulk Export & Simulation Mode."
            )
            self.es = None
            self.is_connected = False

    def create_index_if_not_exists(self) -> bool:
        """Creates target index with schema mapping if connected."""
        if not self.is_connected or not self.es:
            return False

        try:
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name, body=self.INDEX_MAPPING)
                logger.info(f"Created Elasticsearch index: '{self.index_name}' with custom mapping.")
            else:
                logger.info(f"Elasticsearch index '{self.index_name}' already exists.")
            return True
        except Exception as e:
            logger.error(f"Error creating Elasticsearch index: {e}")
            return False

    def format_document(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes and prepares a record for Elasticsearch document ingestion."""
        # Ensure timestamp is formatted for ES date parsing
        ts = post.get("iso_timestamp")
        if not ts:
            created_utc = post.get("created_utc", datetime.now(timezone.utc).timestamp())
            ts = datetime.fromtimestamp(created_utc, timezone.utc).isoformat()

        doc = {
            "id": post.get("id"),
            "source": post.get("source"),
            "type": post.get("type", "post"),
            "parent_id": post.get("parent_id"),
            "author": post.get("author", "unknown"),
            "created_utc": float(post.get("created_utc", 0)),
            "timestamp": ts,
            "raw_text": post.get("raw_text", ""),
            "clean_text": post.get("clean_text", ""),
            "detected_language": post.get("detected_language", "en"),
            "language_confidence": float(post.get("language_confidence", 0.0)),
            "emojis_detected": post.get("emojis_detected", []),
            "emoji_count": int(post.get("emoji_count", 0)),
            "score": int(post.get("score", 0)),
            "views": int(post.get("views", 0)),
            "num_comments": int(post.get("num_comments", 0)),
            "url": post.get("url", ""),
            "sentiment_label": post.get("sentiment_label", "neutral"),
            "sentiment_score": float(post.get("sentiment_score", 0.0)),
            "sentiment_confidence": float(post.get("confidence", 0.5)),
            "topic_id": int(post.get("topic_id", -1)),
            "topic_name": post.get("topic_name", "General"),
            "topic_probability": float(post.get("topic_probability", 0.5)),
            "author_influencer_rank": int(post.get("author_influencer_rank", 999)),
            "author_pagerank": float(post.get("author_pagerank", 0.0)),
            "author_community_id": int(post.get("author_community_id", -1)),
            "misinformation_score": float(post.get("misinformation_score", 0.0)),
            "misinformation_label": post.get("misinformation_label", "credible"),
        }
        return doc

    def bulk_index(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Indexes all enriched documents into Elasticsearch.
        Also exports an NDJSON bulk ingestion file for offline / batch ingestion.
        """
        logger.info(f"Step 7: Indexing {len(posts)} enriched intelligence records into Elasticsearch...")

        docs = [self.format_document(p) for p in posts]

        # 1. Live Elasticsearch Bulk Ingestion if connected
        indexed_count = 0
        if self.is_connected and self.es:
            try:
                from elasticsearch.helpers import bulk

                actions = [
                    {"_index": self.index_name, "_id": d["id"], "_source": d}
                    for d in docs
                ]
                success, errors = bulk(self.es, actions, refresh=True)
                indexed_count = success
                logger.info(f"Successfully bulk indexed {success} documents into Elasticsearch.")
            except Exception as e:
                logger.error(f"Error performing Elasticsearch bulk index: {e}")

        # 2. Always generate NDJSON bulk file for reproducibility
        ndjson_path = PROCESSED_DATA_DIR / "elasticsearch_bulk.ndjson"
        with open(ndjson_path, "w", encoding="utf-8") as f:
            for d in docs:
                action_line = json.dumps({"index": {"_index": self.index_name, "_id": d["id"]}})
                data_line = json.dumps(d)
                f.write(action_line + "\n" + data_line + "\n")

        # Save mapping schema
        mapping_path = PROCESSED_DATA_DIR / "elasticsearch_schema.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.INDEX_MAPPING, f, indent=2)

        logger.info(f"Generated Elasticsearch NDJSON bulk payload at {ndjson_path}")
        return {
            "status": "success",
            "total_documents": len(docs),
            "indexed_count": indexed_count,
            "ndjson_file": str(ndjson_path),
            "is_connected": self.is_connected,
        }

    def search_query(
        self,
        query_text: Optional[str] = None,
        sentiment: Optional[str] = None,
        min_misinformation: Optional[float] = None,
        size: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search method across indexed intelligence data.
        """
        must_clauses = []
        if query_text:
            must_clauses.append({"match": {"clean_text": query_text}})
        if sentiment:
            must_clauses.append({"term": {"sentiment_label": sentiment}})
        if min_misinformation is not None:
            must_clauses.append({"range": {"misinformation_score": {"gte": min_misinformation}}})

        query_body = {"query": {"bool": {"must": must_clauses if must_clauses else [{"match_all": {}}]}}, "size": size}

        if self.is_connected and self.es:
            res = self.es.search(index=self.index_name, body=query_body)
            return [hit["_source"] for hit in res.get("hits", {}).get("hits", [])]

        # Offline query simulation
        bulk_path = PROCESSED_DATA_DIR / "posts_with_misinformation.json"
        if bulk_path.exists():
            with open(bulk_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            filtered = data
            if query_text:
                filtered = [d for d in filtered if query_text.lower() in (d.get("clean_text", "")).lower()]
            if sentiment:
                filtered = [d for d in filtered if d.get("sentiment_label") == sentiment]
            if min_misinformation is not None:
                filtered = [d for d in filtered if d.get("misinformation_score", 0) >= min_misinformation]
            return filtered[:size]

        return []


def run_indexing(
    input_posts_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes complete Step 7 indexing workflow:
    Reads fully enriched posts -> establishes ES index mapping ->
    bulk indexes documents -> generates NDJSON file.
    """
    in_file = Path(input_posts_path or (PROCESSED_DATA_DIR / "posts_with_misinformation.json"))
    if not in_file.exists():
        in_file = Path(PROCESSED_DATA_DIR / "posts_with_graph_metrics.json")
    if not in_file.exists():
        in_file = Path(PROCESSED_DATA_DIR / "clean_posts.json")

    with open(in_file, "r", encoding="utf-8") as f:
        posts = json.load(f)

    indexer = ElasticsearchIndexer()
    indexer.create_index_if_not_exists()
    results = indexer.bulk_index(posts)

    return results


if __name__ == "__main__":
    run_indexing()
