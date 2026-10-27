"""
Indexing and Dashboard Package for Elasticsearch and Kibana.
Step 7: Index everything in Elasticsearch and build Kibana dashboards for search, trends, and alerts.
"""

from .elasticsearch_indexer import ElasticsearchIndexer, run_indexing

__all__ = ["ElasticsearchIndexer", "run_indexing"]
