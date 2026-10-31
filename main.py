"""
Main Pipeline Orchestrator for Large-Scale Social Media Intelligence Platform.
Executes every instruction in the exact order specified:
  Step 1: Collect data with the Reddit and YouTube APIs (respect rate limits and terms of use). Store raw JSON.
  Step 2: Clean text: remove noise, handle emojis, detect language.
  Step 3: Run multilingual sentiment analysis with a transformer model and track it over time.
  Step 4: Use BERTopic to discover topics and show how they rise and fall.
  Step 5: Build a user-interaction graph in Neo4j and rank influencers using PageRank and community detection (NetworkX).
  Step 6: Train a simple misinformation or fake-news classifier on a labeled dataset such as LIAR.
  Step 7: Index everything in Elasticsearch and build Kibana dashboards for search, trends, and alerts.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_collection.collector_runner import run_collection
from graph_analysis.networkx_analyzer import run_graph_analysis
from indexing_and_dashboard.elasticsearch_indexer import run_indexing
from misinformation_detection.misinformation_classifier import run_misinformation_training
from preprocessing.text_cleaner import process_raw_dataset
from reporting.data_presenter import generate_presentable_outputs
from sentiment_analysis.sentiment_analyzer import run_sentiment_tracking
from topic_modeling.topic_modeler import run_topic_modeling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def execute_pipeline(limit_per_source: int = 15):
    """Executes all 7 steps in exact order."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING LARGE-SCALE SOCIAL MEDIA INTELLIGENCE PLATFORM PIPELINE")
    logger.info("=" * 80)

    # -------------------------------------------------------------
    # STEP 1: Collect data with Reddit and YouTube APIs (store raw JSON)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 1: Data Collection (Reddit & YouTube APIs) <<<")
    collection_stats = run_collection(sources=["reddit", "youtube"], limit=limit_per_source)
    logger.info(f"Step 1 Complete: {collection_stats}")

    # -------------------------------------------------------------
    # STEP 2: Clean text: remove noise, handle emojis, detect language
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 2: Text Cleaning & Preprocessing <<<")
    cleaned_posts = process_raw_dataset()
    logger.info(f"Step 2 Complete: Cleaned {len(cleaned_posts)} posts.")

    # -------------------------------------------------------------
    # STEP 3: Multilingual sentiment analysis with transformer model & temporal tracking
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 3: Spark NLP Multilingual Sentiment Analysis & Tracking <<<")
    sentiment_posts, sentiment_trends = run_sentiment_tracking()
    logger.info(f"Step 3 Complete: Processed {len(sentiment_posts)} posts across {len(sentiment_trends)} time buckets.")

    # -------------------------------------------------------------
    # STEP 4: BERTopic topic discovery and rise/fall dynamics
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 4: BERTopic Dynamic Topic Modeling <<<")
    topic_posts, topics_over_time = run_topic_modeling()
    logger.info(f"Step 4 Complete: Topics modeled across {len(topics_over_time)} dynamic time bins.")

    # -------------------------------------------------------------
    # STEP 5: Build user-interaction graph in Neo4j, rank influencers (PageRank & NetworkX)
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 5: User-Interaction Graph (Neo4j & NetworkX) <<<")
    influencer_rankings, graph_stats = run_graph_analysis()
    logger.info(f"Step 5 Complete: Ranked {len(influencer_rankings)} influencers across {graph_stats['num_communities']} communities.")

    # -------------------------------------------------------------
    # STEP 6: Train misinformation classifier on LIAR dataset
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 6: Misinformation / Fake-News Classifier (LIAR) <<<")
    liar_metrics, scored_posts = run_misinformation_training()
    logger.info(f"Step 6 Complete: Accuracy: {liar_metrics['accuracy']}, F1: {liar_metrics['f1_score']}.")

    # -------------------------------------------------------------
    # STEP 7: Index everything in Elasticsearch and build Kibana dashboards
    # -------------------------------------------------------------
    logger.info("\n>>> STEP 7: Elasticsearch Indexing & Kibana Dashboards <<<")
    index_results = run_indexing()
    logger.info(f"Step 7 Complete: Index status: {index_results['status']}, total docs: {index_results['total_documents']}.")

    # -------------------------------------------------------------
    # PRESENTATION & REPORTING: Format Processed Data Outputs
    # -------------------------------------------------------------
    logger.info("\n>>> GENERATING PRESENTABLE OUTPUTS (CSVs, Executive Summary, HTML Dashboard) <<<")
    generate_presentable_outputs()

    elapsed = round(time.time() - start_time, 2)
    logger.info("\n" + "=" * 80)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed}s")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large-Scale Social Media Intelligence Platform")
    parser.add_argument("--run-all", action="store_true", default=False, help="Run all 7 steps in exact order")
    parser.add_argument("--present", action="store_true", help="Generate and display presentable reports from existing processed data")
    parser.add_argument("--limit", type=int, default=15, help="Items per source for Step 1")
    args = parser.parse_args()

    if args.present:
        generate_presentable_outputs()
    else:
        execute_pipeline(limit_per_source=args.limit)
