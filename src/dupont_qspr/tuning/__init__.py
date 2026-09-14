"""Hyperparameter search, run inside the inner cross-validation loop.

Optuna and cross-validation are orthogonal and neither replaces the other. Optuna
is the *search strategy* - it proposes configurations. Inner CV is the *evaluation
protocol* - it scores a proposal. ``spaces`` defines what may be proposed;
``nested_run`` drives the loop that scores it.
"""

from __future__ import annotations

from dupont_qspr.tuning.spaces import MTL_SPACE, XGB_SPACE, suggest_mtl, suggest_xgb

__all__ = ["MTL_SPACE", "XGB_SPACE", "suggest_mtl", "suggest_xgb"]
