"""Evaluation metrics, reported per property and never pooled.

The three properties are on unrelated scales - log units against Kelvin - so a
pooled average would hide a weak model behind a strong one. Every function here
takes one property's values at a time.
"""

from __future__ import annotations

from dupont_qspr.metrics.point import point_metrics, rmse, rmse_over_sd, spearman

__all__ = ["point_metrics", "rmse", "rmse_over_sd", "spearman"]
