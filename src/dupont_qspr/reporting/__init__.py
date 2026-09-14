"""Tables and figures for the point-accuracy report.

Reads what the nested run persisted; refits nothing. Everything is reported per
property, because the three targets are on unrelated scales.
"""

from __future__ import annotations

from dupont_qspr.reporting.figures import DARK, LIGHT, render_all
from dupont_qspr.reporting.load import latest_nested_result, load_oof_predictions
from dupont_qspr.reporting.tables import baseline_table, ranking_summary, results_table

__all__ = [
    "DARK",
    "LIGHT",
    "baseline_table",
    "latest_nested_result",
    "load_oof_predictions",
    "ranking_summary",
    "render_all",
    "results_table",
]
