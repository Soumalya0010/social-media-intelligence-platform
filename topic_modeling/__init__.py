"""
Topic Modeling Package using BERTopic.
Step 4: Use BERTopic to discover topics and show how they rise and fall.
"""

from .topic_modeler import BERTopicModeler, run_topic_modeling

__all__ = ["BERTopicModeler", "run_topic_modeling"]
