"""Steps 7, 9 and 10: read what the runs persisted and turn it into decisions.

``comparison`` puts the model tracks side by side on identical held-out rows and
chooses the family that ships. ``coverage`` checks whether the calibrated
intervals hold up as molecules move away from the training set, and reads the
applicability-domain threshold off the error curve. ``ablation`` runs and
summarises the low-data experiment.

Nothing here imports XGBoost or PyTorch at module level. The ablation functions
import their model family inside the branch that needs it, because the two
runtimes cannot share a process.
"""

from __future__ import annotations

from dupont_qspr.analysis.comparison import (
    FamilyDecision,
    comparison_table,
    decide_family,
    load_track_predictions,
)
from dupont_qspr.analysis.coverage import (
    choose_ad_threshold,
    conditional_coverage,
    error_vs_distance,
    load_interval_frame,
)

__all__ = [
    "FamilyDecision",
    "choose_ad_threshold",
    "comparison_table",
    "conditional_coverage",
    "decide_family",
    "error_vs_distance",
    "load_interval_frame",
    "load_track_predictions",
]
