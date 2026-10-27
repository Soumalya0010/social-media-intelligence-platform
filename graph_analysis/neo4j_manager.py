"""
Neo4j Graph Database Manager.
Step 5: Build a user-interaction graph in Neo4j.
Required Tech Stack: Neo4j
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Manages Neo4j graph database operations for user interactions.
    Provides schema definitions, node/relationship ingestion, and metric synchronization.
    Includes offline cypher export when Neo4j daemon is not actively reachable.
    """

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.is_connected = False
        self._connect()

    def _connect(self):
        """Attempts connection to Neo4j instance."""
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.is_connected = True
            logger.info(f"Successfully connected to Neo4j instance at {self.uri}")
        except Exception as e:
            logger.warning(
                f"Neo4j instance unreachable at {self.uri} ({e}). "
                "Enabling Neo4j Cypher Script Export & Simulation mode."
            )
            self.driver = None
            self.is_connected = False

    def close(self):
        """Closes driver session."""
        if self.driver:
            self.driver.close()

    def build_user_interaction_graph(
        self,
        users: List[Dict[str, Any]],
        posts: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Builds the user-interaction graph in Neo4j.
        Creates:
        - User nodes: (:User {username, source})
        - Post nodes: (:Post {id, title, score, source})
        - Relationships:
          - (:User)-[:AUTHORED]->(:Post)
          - (:User)-[:INTERACTED_WITH {weight: ...}]->(:User)
          - (:User)-[:REPLIED_TO]->(:User)
        """
        logger.info(f"Building Neo4j user-interaction graph ({len(users)} users, {len(interactions)} interactions)...")

        cypher_queries = [
            # Schema Constraints
            "CREATE CONSTRAINT user_username IF NOT EXISTS FOR (u:User) REQUIRE u.username IS UNIQUE;",
            "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE;",
        ]

        # User node ingestion
        for u in users:
            cypher_queries.append(
                f"MERGE (u:User {{username: '{u['username']}'}}) "
                f"ON CREATE SET u.source = '{u.get('source', 'social_media')}', u.created_at = datetime();"
            )

        # Post node ingestion & Authored edge
        for p in posts:
            author = p.get("author", "unknown").replace("'", "")
            pid = p.get("id", "").replace("'", "")
            title = (p.get("clean_text") or p.get("raw_text", ""))[:100].replace("'", "\\'")
            cypher_queries.append(
                f"MERGE (p:Post {{id: '{pid}'}}) "
                f"ON CREATE SET p.title = '{title}', p.score = {p.get('score', 0)}, p.source = '{p.get('source', '')}';\n"
                f"MERGE (u:User {{username: '{author}'}}) "
                f"MERGE (u)-[:AUTHORED]->(p);"
            )

        # Interacted / Replied relationships
        for edge in interactions:
            u_src = edge["source_user"].replace("'", "")
            u_tgt = edge["target_user"].replace("'", "")
            w = edge.get("weight", 1)
            rel_type = edge.get("type", "INTERACTED_WITH")
            cypher_queries.append(
                f"MERGE (u1:User {{username: '{u_src}'}}) "
                f"MERGE (u2:User {{username: '{u_tgt}'}}) "
                f"MERGE (u1)-[r:{rel_type}]->(u2) "
                f"ON CREATE SET r.weight = {w} "
                f"ON MATCH SET r.weight = r.weight + {w};"
            )

        # Execute in Neo4j if connected
        executed_count = 0
        if self.is_connected and self.driver:
            try:
                with self.driver.session() as session:
                    for query in cypher_queries:
                        session.run(query)
                        executed_count += 1
                logger.info(f"Successfully executed {executed_count} Cypher operations in Neo4j.")
            except Exception as e:
                logger.error(f"Error executing Cypher queries in Neo4j: {e}")

        # Always export Cypher script for reproducible deployment and audit
        export_path = Path(DATA_DIR / "processed" / "neo4j_graph_export.cypher")
        with open(export_path, "w", encoding="utf-8") as f:
            f.write("// Neo4j User-Interaction Graph Ingestion Script\n")
            f.write("\n".join(cypher_queries))

        logger.info(f"Exported Neo4j Cypher ingestion script to {export_path}")
        return {"status": "success", "cypher_queries_count": len(cypher_queries), "is_connected": self.is_connected}

    def sync_influencer_rankings(self, rankings: List[Dict[str, Any]]):
        """
        Updates Neo4j User nodes with PageRank scores and Community Detection IDs.
        """
        if not self.is_connected or not self.driver:
            return

        logger.info(f"Syncing PageRank and Community IDs to Neo4j for {len(rankings)} users...")
        try:
            with self.driver.session() as session:
                for r in rankings:
                    query = (
                        "MATCH (u:User {username: $username}) "
                        "SET u.pagerank = $pagerank, u.influencer_rank = $rank, u.community_id = $comm_id"
                    )
                    session.run(
                        query,
                        username=r["username"],
                        pagerank=r["pagerank"],
                        rank=r["rank"],
                        comm_id=r["community_id"],
                    )
            logger.info("Successfully synced influencer metrics to Neo4j.")
        except Exception as e:
            logger.error(f"Error syncing rankings to Neo4j: {e}")
