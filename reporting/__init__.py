"""
Reporting and Presentation Package.
Transforms raw processed JSON outputs of all 7 steps into presentable formats:
- Tabular CSVs for Excel/Sheets analysis
- Executive Markdown Summary Report
- Interactive Self-Contained HTML Intelligence Dashboard
- Formatted Terminal Presentation Tables
"""

from .data_presenter import generate_presentable_outputs, print_terminal_summary

__all__ = ["generate_presentable_outputs", "print_terminal_summary"]
