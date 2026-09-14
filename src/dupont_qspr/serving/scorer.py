"""SMILES in, scored records out - the deliverable's runtime.

The path follows brief §13 exactly::

    SMILES -> standardisation -> featurisation -> model -> conformal interval
           -> applicability-domain distance -> scored record

Two requirements are structural rather than bolted on. The interface takes a list,
because the consumer scores many molecules at once. And a molecule that cannot be
standardised yields a structured error record in its place instead of raising,
because one bad SMILES must not cost the rest of the batch.

The bundle is a plain directory - a manifest, the model files for one family, and
the training fingerprints for the applicability domain packed to bits (about 4 MB
rather than 120 MB). It is readable without MLflow, which keeps the scorer
testable on its own and lets the CLI use it directly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from dupont_qspr.contracts import (
    PROPERTIES,
    ApplicabilityDomain,
    ErrorRecord,
    PropertyPrediction,
    ScoredRecord,
)

__all__ = ["MANIFEST", "QSPRScorer", "save_ad_reference"]

MANIFEST = "manifest.json"


def save_ad_reference(fingerprints: npt.NDArray[np.float32], directory: Path) -> None:
    """Store training fingerprint bits packed eight to a byte."""
    bits = np.nan_to_num(fingerprints) > 0
    np.savez_compressed(
        directory / "ad_reference.npz",
        bits=np.packbits(bits, axis=1),
        n_bits=np.int64(bits.shape[1]),
    )


@dataclass(slots=True)
class QSPRScorer:
    manifest: dict[str, Any]
    ad_reference: npt.NDArray[np.float32]
    _xgb_point: dict[str, Any] = field(default_factory=dict)
    _xgb_quantile: dict[str, Any] = field(default_factory=dict)
    _ensemble: Any = None

    # -- loading ----------------------------------------------------------- #

    @classmethod
    def load(cls, directory: Path) -> QSPRScorer:
        directory = Path(directory)
        manifest = json.loads((directory / MANIFEST).read_text())
        packed = np.load(directory / "ad_reference.npz")
        bits = np.unpackbits(packed["bits"], axis=1, count=int(packed["n_bits"]))
        scorer = cls(manifest=manifest, ad_reference=bits.astype(np.float32))
        if manifest["family"] == "xgb":
            scorer._load_xgb(directory)
        elif manifest["family"] == "mtl":
            scorer._load_mtl(directory)
        else:
            raise ValueError(f"unknown family {manifest['family']!r}")
        return scorer

    def _load_xgb(self, directory: Path) -> None:
        from xgboost import XGBRegressor

        from dupont_qspr.models.xgb import XGBPropertyModel, XGBQuantileModel

        quantiles = tuple(self.manifest["quantiles"])
        for prop in PROPERTIES:
            point = XGBRegressor()
            point.load_model(str(directory / "xgb" / f"{prop}_point.json"))
            wrapped = XGBPropertyModel()
            wrapped._model = point
            self._xgb_point[prop] = wrapped

            quantile = XGBQuantileModel(quantiles=quantiles)
            for index in range(len(quantiles)):
                regressor = XGBRegressor()
                regressor.load_model(str(directory / "xgb" / f"{prop}_q{index}.json"))
                quantile._models.append(regressor)
            self._xgb_quantile[prop] = quantile

    def _load_mtl(self, directory: Path) -> None:
        import torch

        from dupont_qspr.models.ensemble import DeepEnsemble
        from dupont_qspr.models.mtl import MultiTaskHeads, MultiTaskModel

        params = self.manifest["mtl_params"]
        stats = np.load(directory / "mtl" / "standardization.npz")
        ensemble = DeepEnsemble(
            params=params, n_members=int(self.manifest["n_members"])
        )
        for member in range(ensemble.n_members):
            net = MultiTaskHeads(
                int(self.manifest["embedding_width"]),
                hidden_width=int(params["hidden_width"]),
                n_layers=int(params["n_layers"]),
                dropout=float(params["dropout"]),
                n_properties=len(PROPERTIES),
            )
            net.load_state_dict(
                torch.load(
                    directory / "mtl" / f"member_{member}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            model = MultiTaskModel(params=params)
            model._net = net.eval()
            model._centre = stats[f"centre_{member}"]
            model._scale = stats[f"scale_{member}"]
            ensemble._members.append(model)
        self._ensemble = ensemble

    # -- scoring ----------------------------------------------------------- #

    def score(self, smiles: Sequence[Any]) -> list[dict[str, Any]]:
        """One record per input, in input order."""
        from dupont_qspr.data.standardize import standardize_smiles
        from dupont_qspr.features.descriptors import compute_path_a

        settings = self.manifest["standardize"]
        records: list[dict[str, Any] | None] = [None] * len(smiles)
        valid: list[tuple[int, str]] = []
        for position, raw in enumerate(smiles):
            try:
                result = standardize_smiles(
                    raw if isinstance(raw, str) else None,
                    mixture_ratio=settings["mixture_ratio"],
                    require_carbon=settings["require_carbon"],
                    keep_stereochemistry=settings["keep_stereochemistry"],
                )
            except Exception as error:  # noqa: BLE001 - one bad input must not kill a batch
                records[position] = ErrorRecord(
                    str(raw), f"standardization failed: {error}"
                ).to_dict()
                continue
            if result.canonical is None:
                detail = f" ({result.detail})" if result.detail else ""
                records[position] = ErrorRecord(
                    str(raw), f"rejected: {result.reason}{detail}"
                ).to_dict()
            else:
                valid.append((position, result.canonical))

        if valid:
            unique = list(dict.fromkeys(c for _, c in valid))
            index = {c: k for k, c in enumerate(unique)}
            path_a = compute_path_a(unique)
            point, raw_quantiles = self._predict(unique, path_a)
            distance = self._distance(path_a)
            threshold = self.manifest.get("ad_threshold")
            coverage = float(self.manifest["nominal_coverage"])

            for position, canonical in valid:
                k = index[canonical]
                predictions = {}
                for j, prop in enumerate(PROPERTIES):
                    offset = float(self.manifest["cqr_offsets"][prop])
                    value = float(point[k, j])
                    lower = float(raw_quantiles[k, j, 0]) - offset
                    upper = float(raw_quantiles[k, j, -1]) + offset
                    # The point model and the quantile models are fitted separately,
                    # so in rare cases the point can sit just outside the interval.
                    # Widening to include it only ever increases coverage.
                    lower, upper = min(lower, upper, value), max(lower, upper, value)
                    predictions[prop] = PropertyPrediction(
                        value, lower, upper, coverage
                    )
                records[position] = ScoredRecord(
                    smiles_canonical=canonical,
                    predictions=predictions,
                    applicability_domain=ApplicabilityDomain(
                        nn_tanimoto_distance=float(distance[k]),
                        # No threshold means step 9 never ran: claim nothing is in domain.
                        in_domain=bool(
                            threshold is not None and distance[k] <= threshold
                        ),
                    ),
                    model_version=self.manifest["model_version"],
                    featurizer_version=self.manifest["featurizer_version"],
                ).to_dict()

        return [r for r in records if r is not None]

    def _predict(
        self, smiles: list[str], path_a: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Point predictions ``(n, 3)`` and raw quantiles ``(n, 3, 3)``."""
        if self.manifest["family"] == "xgb":
            point = np.column_stack(
                [self._xgb_point[p].predict(path_a) for p in PROPERTIES]
            )
            raw = np.stack(
                [self._xgb_quantile[p].predict_quantiles(path_a) for p in PROPERTIES],
                axis=1,
            )
            return point, raw

        from dupont_qspr.features.encoders import compute_path_b

        embeddings = compute_path_b(
            smiles,
            model_name=self.manifest["encoder_model"],
            device="cpu",
            batch_size=int(self.manifest["encoder_batch_size"]),
            pooling=self.manifest["encoder_pooling"],
        )
        return self._ensemble.predict(embeddings), self._ensemble.predict_quantiles(
            embeddings
        )

    def _distance(self, path_a: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        from dupont_qspr.uncertainty.applicability import ApplicabilityDomainIndex

        index = ApplicabilityDomainIndex(
            fingerprint_start=int(self.manifest["fingerprint_start"])
        )
        index._reference = self.ad_reference
        return index.distance(np.nan_to_num(path_a))
