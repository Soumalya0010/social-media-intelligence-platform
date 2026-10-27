"""
Misinformation Detection Package.
Step 6: Train a simple misinformation or fake-news classifier on a labeled dataset such as LIAR.
"""

from .misinformation_classifier import MisinformationClassifier, run_misinformation_training

__all__ = ["MisinformationClassifier", "run_misinformation_training"]
