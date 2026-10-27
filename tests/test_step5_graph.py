"""
Unit and integration tests for Step 5: Neo4j & NetworkX Graph Analysis, PageRank, Community Detection.
"""

import unittest
from graph_analysis.neo4j_manager import Neo4jManager
from graph_analysis.networkx_analyzer import NetworkXGraphAnalyzer


class TestStep5GraphAnalysis(unittest.TestCase):
    """Verifies graph creation, PageRank influencer ranking, and community detection."""

    def setUp(self):
        self.analyzer = NetworkXGraphAnalyzer()
        self.users = [
            {"username": "alpha_influencer", "source": "reddit"},
            {"username": "beta_user", "source": "reddit"},
            {"username": "gamma_user", "source": "youtube"},
            {"username": "delta_user", "source": "reddit"},
        ]
        self.interactions = [
            {"source_user": "beta_user", "target_user": "alpha_influencer", "weight": 5.0, "type": "REPLIED_TO"},
            {"source_user": "gamma_user", "target_user": "alpha_influencer", "weight": 3.0, "type": "MENTIONED"},
            {"source_user": "delta_user", "target_user": "beta_user", "weight": 2.0, "type": "REPLIED_TO"},
        ]

    def test_networkx_graph_building(self):
        G = self.analyzer.build_networkx_graph(self.users, self.interactions)
        self.assertEqual(G.number_of_nodes(), 4)
        self.assertGreaterEqual(G.number_of_edges(), 3)

    def test_pagerank_influencer_ranking(self):
        self.analyzer.build_networkx_graph(self.users, self.interactions)
        rankings = self.analyzer.analyze_and_rank()

        self.assertGreater(len(rankings), 0)
        # alpha_influencer receives the most incoming interactions, should be rank 1
        top_user = rankings[0]
        self.assertEqual(top_user["username"], "alpha_influencer")
        self.assertEqual(top_user["rank"], 1)
        self.assertIn("community_id", top_user)

    def test_neo4j_cypher_export(self):
        neo4j_mgr = Neo4jManager()
        sample_posts = [{"id": "p1", "author": "alpha_influencer", "clean_text": "Graph tutorial", "score": 100}]
        res = neo4j_mgr.build_user_interaction_graph(self.users, sample_posts, self.interactions)
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["cypher_queries_count"], 0)


if __name__ == "__main__":
    unittest.main()
