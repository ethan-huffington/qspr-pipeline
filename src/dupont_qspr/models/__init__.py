"""Model tracks. Both satisfy ``QSPRModel``, so the harness is written once.

**This module deliberately imports nothing.**

XGBoost and PyTorch each link their own OpenMP runtime, and a process that has
imported one cannot safely load the other - it hangs or segfaults on macOS. If
this file eagerly imported ``xgb``, then ``import dupont_qspr.models.mtl`` would
drag XGBoost in as a side effect of touching the package, and the neural track
would poison its own process before doing any work.

So callers import from the submodule they actually want::

    from dupont_qspr.models.xgb import XGBPropertyModel   # Track 1 process
    from dupont_qspr.models.mtl import MultiTaskModel     # Track 2 process

The two tracks are separate runs in any case, so this costs nothing beyond a
slightly longer import line - and a test asserts the neural path stays clean.
"""

from __future__ import annotations

__all__: list[str] = []
