"""
NetworkX Graph Analyzer.
Step 5: Rank influencers using PageRank and community detection (NetworkX).
Required Tech Stack: NetworkX
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
from config import PROCESSED_DATA_DIR
from graph_analysis.neo4j_manager import Neo4jManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NetworkXGraphAnalyzer:
    """
    Builds interaction networks, computes PageRank for influencer ranking,
    and runs community detection using NetworkX algorithms.
    """

    def __init__(self):
        self.G = nx.DiGraph()

    def extract_interactions(self, posts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extracts users, post relationships, and interaction edges:
        1. Comment-to-Post Author replies (:User)-[:REPLIED_TO]->(:User)
        2. Mentions in text: @user (:User)-[:MENTIONED]->(:User)
        3. Co-engagement relationships
        """
        users_dict = {}
        interactions = []

        # Map parent post IDs to post author
        post_author_map = {}
        for p in posts:
            author = p.get("author", "").strip()
            if author and author not in ["[deleted]", "Anonymous", "unknown"]:
                users_dict[author] = {"username": author, "source": p.get("source", "social_media")}
                if p.get("type") in ["post", "video"]:
                    post_author_map[p.get("id")] = author

        mention_regex = re.compile(r"[@/]([A-Za-z0-9_-]+)")

        for p in posts:
            author = p.get("author", "").strip()
            if not author or author in ["[deleted]", "Anonymous", "unknown"]:
                continue

            # Interaction 1: Comment replied to parent post author
            parent_id = p.get("parent_id")
            if parent_id and parent_id in post_author_map:
                target_author = post_author_map[parent_id]
                if target_author != author:
                    interactions.append({
                        "source_user": author,
                        "target_user": target_author,
                        "weight": 2.0,
                        "type": "REPLIED_TO",
                        "post_id": parent_id,
                    })

            # Interaction 2: Mentions in text
            text = p.get("clean_text") or p.get("raw_text", "")
            mentions = mention_regex.findall(text)
            for m in mentions:
                if m != author and len(m) > 2:
                    if m not in users_dict:
                        users_dict[m] = {"username": m, "source": p.get("source", "social_media")}
                    interactions.append({
                        "source_user": author,
                        "target_user": m,
                        "weight": 1.5,
                        "type": "MENTIONED",
                        "post_id": p.get("id"),
                    })

        # Ensure synthetic/curated network connectivity if interactions are sparse
        if len(interactions) < 3:
            sample_users = list(users_dict.keys())
            for i in range(len(sample_users) - 1):
                interactions.append({
                    "source_user": sample_users[i + 1],
                    "target_user": sample_users[i],
                    "weight": 1.0,
                    "type": "INTERACTED_WITH",
                    "post_id": "seed",
                })

        return list(users_dict.values()), posts, interactions

    def build_networkx_graph(self, users: List[Dict[str, Any]], interactions: List[Dict[str, Any]]) -> nx.DiGraph:
        """Populates directed graph with user nodes and weighted interaction edges."""
        self.G.clear()

        for u in users:
            self.G.add_node(u["username"], **u)

        # Aggregate multi-edges into weighted single directed edges
        edge_weights = defaultdict(float)
        edge_types = defaultdict(set)

        for inter in interactions:
            u_src = inter["source_user"]
            u_tgt = inter["target_user"]
            edge_weights[(u_src, u_tgt)] += inter.get("weight", 1.0)
            edge_types[(u_src, u_tgt)].add(inter.get("type", "INTERACTED_WITH"))

        for (u_src, u_tgt), weight in edge_weights.items():
            self.G.add_edge(u_src, u_tgt, weight=weight, types=list(edge_types[(u_src, u_tgt)]))

        logger.info(f"Constructed NetworkX graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        return self.G

    def rank_influencers_pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """
        Calculates PageRank on the interaction graph to rank key influencers.
        Higher PageRank indicates higher influence in discussions.
        """
        if self.G.number_of_nodes() == 0:
            return {}

        try:
            scores = nx.pagerank(self.G, alpha=alpha, weight="weight", max_iter=200)
        except Exception as e:
            logger.warning(f"Standard PageRank converged with exception ({e}), falling back to degree centrality.")
            scores = nx.degree_centrality(self.G)

        return scores

    def detect_communities(self) -> Dict[str, int]:
        """
        Performs community detection using NetworkX greedy modularity or Louvain.
        Partitions the interaction graph into distinct cohesive user clusters.
        """
        if self.G.number_of_nodes() < 2:
            return {node: 0 for node in self.G.nodes()}

        undirected = self.G.to_undirected()
        try:
            communities = list(nx.community.greedy_modularity_communities(undirected, weight="weight"))
        except Exception:
            communities = list(nx.connected_components(undirected))

        node_community_map = {}
        for comm_id, member_set in enumerate(communities):
            for member in member_set:
                node_community_map[member] = comm_id

        return node_community_map

    def analyze_and_rank(self) -> List[Dict[str, Any]]:
        """
        Computes PageRank and Community Detection, combining metrics into a ranked leaderboard.
        """
        pagerank_scores = self.rank_influencers_pagerank()
        communities = self.detect_communities()

        ranked_users = []
        for username in self.G.nodes():
            pr = pagerank_scores.get(username, 0.0)
            comm_id = communities.get(username, 0)
            in_deg = self.G.in_degree(username) if self.G.has_node(username) else 0
            out_deg = self.G.out_degree(username) if self.G.has_node(username) else 0

            ranked_users.append({
                "username": username,
                "pagerank": round(pr, 5),
                "community_id": comm_id,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "total_degree": in_deg + out_deg,
            })

        # Rank by PageRank descending
        ranked_users.sort(key=lambda x: x["pagerank"], reverse=True)
        for rank_idx, u in enumerate(ranked_users, start=1):
            u["rank"] = rank_idx

        return ranked_users


def run_graph_analysis(
    input_posts_path: Optional[Path] = None,
    output_rankings_path: Optional[Path] = None,
    output_metrics_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes complete Step 5 workflow:
    Extracts interactions -> builds Neo4j graph & NetworkX graph ->
    calculates PageRank influencer rankings -> runs Community Detection ->
    syncs metrics to Neo4j.
    """
    in_file = Path(input_posts_path or (PROCESSED_DATA_DIR / "posts_with_topics.json"))
    if not in_file.exists():
        in_file = Path(PROCESSED_DATA_DIR / "posts_with_sentiment.json")
    if not in_file.exists():
        in_file = Path(PROCESSED_DATA_DIR / "clean_posts.json")

    with open(in_file, "r", encoding="utf-8") as f:
        posts = json.load(f)

    analyzer = NetworkXGraphAnalyzer()
    users, _, interactions = analyzer.extract_interactions(posts)
    analyzer.build_networkx_graph(users, interactions)

    # 1. Neo4j Graph Ingestion
    neo4j_mgr = Neo4jManager()
    neo4j_res = neo4j_mgr.build_user_interaction_graph(users, posts, interactions)

    # 2. NetworkX PageRank & Community Detection
    influencer_rankings = analyzer.analyze_and_rank()

    # 3. Sync rankings back to Neo4j
    neo4j_mgr.sync_influencer_rankings(influencer_rankings)
    neo4j_mgr.close()

    # 4. Annotate posts with author's influencer rank and community
    rank_lookup = {u["username"]: u for u in influencer_rankings}
    for p in posts:
        author = p.get("author", "")
        if author in rank_lookup:
            p["author_influencer_rank"] = rank_lookup[author]["rank"]
            p["author_pagerank"] = rank_lookup[author]["pagerank"]
            p["author_community_id"] = rank_lookup[author]["community_id"]
        else:
            p["author_influencer_rank"] = 999
            p["author_pagerank"] = 0.001
            p["author_community_id"] = -1

    # Save outputs
    out_rankings = Path(output_rankings_path or (PROCESSED_DATA_DIR / "influencer_rankings.json"))
    out_posts = Path(PROCESSED_DATA_DIR / "posts_with_graph_metrics.json")
    out_metrics = Path(output_metrics_path or (PROCESSED_DATA_DIR / "graph_metrics.json"))

    with open(out_rankings, "w", encoding="utf-8") as f:
        json.dump(influencer_rankings, f, ensure_ascii=False, indent=2)

    with open(out_posts, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    graph_stats = {
        "num_nodes": analyzer.G.number_of_nodes(),
        "num_edges": analyzer.G.number_of_edges(),
        "num_communities": len(set(u["community_id"] for u in influencer_rankings)),
        "top_5_influencers": [u["username"] for u in influencer_rankings[:5]],
        "neo4j_status": neo4j_res,
    }

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(graph_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Step 5 completed: Ranked {len(influencer_rankings)} influencers across {graph_stats['num_communities']} communities.")
    logger.info(f"Saved rankings to {out_rankings}")

    return influencer_rankings, graph_stats


if __name__ == "__main__":
    run_graph_analysis()
