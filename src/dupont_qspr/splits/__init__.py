"""Scaffold splitting and the nested cross-validation harness.

``scaffold`` groups molecules by their Bemis-Murcko core; ``nested`` deals those
groups into scaffold-disjoint outer and inner folds, balancing per-property label
counts rather than raw molecule counts.
"""

from __future__ import annotations

from dupont_qspr.splits.nested import FoldBuild, build_nested_folds
from dupont_qspr.splits.scaffold import ScaffoldAssignment, assign_scaffolds

__all__ = [
    "FoldBuild",
    "ScaffoldAssignment",
    "assign_scaffolds",
    "build_nested_folds",
]
