"""Data acquisition, curation, and the union table.

The pipeline this package implements, in order:

``download`` fetches raw files and verifies them against published checksums;
``standardize`` reduces every structure to one canonical join key, recording each
rejection; ``curate_mp`` handles the melting-point set's exclusion flags and
disagreeing replicates; ``union`` outer-joins the three curated tables into the
sparse target matrix the rest of the project consumes.
"""

from __future__ import annotations

from dupont_qspr.data.build import DatasetBuild, build_dataset
from dupont_qspr.data.sources import SOURCES, DataSource

__all__ = ["SOURCES", "DataSource", "DatasetBuild", "build_dataset"]
