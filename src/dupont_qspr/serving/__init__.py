"""Step 11: the deployable scoring artifact.

``scorer`` holds the runtime object that turns SMILES into scored records and the
on-disk bundle format it loads. ``final_fit`` builds that bundle from all the
training data. ``pyfunc`` wraps it as an MLflow model.

**This module deliberately imports nothing**, for the same reason as
``dupont_qspr.models``: the XGBoost and neural families link separate OpenMP
runtimes, so each loads only inside the code path for its own family.
"""

from __future__ import annotations

__all__: list[str] = []
