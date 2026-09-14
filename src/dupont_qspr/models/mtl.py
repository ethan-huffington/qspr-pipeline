"""Track 2 - one network, three heads, trained on a sparse target matrix.

The structural contrast with Track 1 is the point of the whole project. Track 1
fits three independent models, each seeing only the rows where its property is
measured. Track 2 fits *one* network whose lower layers are shared across all
three properties, so a molecule labelled only for melting point still shapes the
representation that solubility and lipophilicity are read out of. That is the
mechanism by which multi-task learning can beat independent models on sparse data
- and equally the mechanism by which it can lose, if the properties fight over
the shared capacity.

The union table is 54% melting-point-only, so almost every row would be discarded
by one of Track 1's three models and is usable by Track 2. Whether that helps is
the question step 7 answers.

Three implementation details carry the design.

**The masked loss.** Each row contributes to a head only where that property is
measured. An unlabelled cell must produce no gradient at all - not a small
gradient, not a gradient toward zero, none. Getting this subtly wrong is silent:
the network trains, the loss decreases, and the model has quietly learned that
missing means zero.

**Target standardization from training rows only.** The three targets live on
wildly different scales (log units against Kelvin), so they are z-scored before
the loss, or melting point's squared error would dwarf the others by four orders
of magnitude. Those statistics must come from the training fold; taking them from
the whole dataset leaks the test distribution's location and scale.

**The trunk is frozen and precomputed.** What this module trains is small: a
couple of shared layers and three linear read-outs, on top of embeddings computed
once at step 2. Measured, a fit takes 5-8s against 4-21s for an XGBoost fit, and
this track needs a third as many studies (five per encoder rather than fifteen,
because one network serves all three properties). A full run is therefore roughly
an hour per encoder - cheaper than Track 1, but not the "minutes" that "we only
train small heads" might suggest.

Device note: MPS is about twice as fast per epoch as CPU, but floating-point
differences change the early-stopping trajectory, so runs are not bit-identical
across devices. CPU is the default for reproducibility; ``--device mps`` trades
that for speed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from dupont_qspr.contracts import PROPERTIES, PropertyName

__all__ = ["MultiTaskHeads", "MultiTaskModel", "default_mtl_params", "masked_mse"]


def default_mtl_params() -> dict[str, Any]:
    return {
        "hidden_width": 256,
        "n_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss_weights": {"logS": 1.0, "logP": 1.0, "mp_K": 1.0},
    }


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean squared error over labelled cells only.

    ``target`` carries ``nan`` wherever ``mask`` is False, and that is the trap.
    Multiplying a ``nan`` by zero gives ``nan``, not zero - so the obvious
    ``((pred - target) ** 2 * mask).mean()`` produces a ``nan`` loss the moment
    any cell is unlabelled, and if it somehow did not, ``nan`` would still
    propagate through the backward pass. The targets are therefore sanitized
    *before* they touch the arithmetic, and the mask is applied to the residual
    rather than to the squared error.

    Normalization is per property - dividing each property's summed error by *its*
    own labelled count - rather than one global division. A global mean would let
    melting point, with five times as many labels, dominate the gradient purely by
    being more numerous.
    """
    safe_target = torch.nan_to_num(target, nan=0.0)
    residual = (prediction - safe_target) * mask

    labelled = mask.sum(dim=0)
    per_property = (residual**2).sum(dim=0) / labelled.clamp(min=1)

    # A property absent from this batch contributes neither error nor weight,
    # rather than contributing a zero that would drag the mean down.
    active = (labelled > 0).to(per_property.dtype)
    weighted = (per_property * weights * active).sum()
    denominator = (weights * active).sum().clamp(min=1e-8)
    return weighted / denominator


class MultiTaskHeads(nn.Module):
    """Shared layers over a frozen embedding, then one linear read-out per property."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden_width: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_properties: int = len(PROPERTIES),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for _ in range(max(1, n_layers)):
            layers += [
                nn.Linear(width, hidden_width),
                nn.GELU(),
                # Dropout is doing real work here: a few thousand labels against a
                # 768-dimensional input overfits without it.
                nn.Dropout(dropout),
            ]
            width = hidden_width
        self.shared = nn.Sequential(*layers)
        # One head per property rather than a single Linear(width, 3), so that a
        # head can be inspected, frozen or replaced independently.
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(n_properties)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        return torch.cat([head(shared) for head in self.heads], dim=1)


@dataclass(slots=True)
class MultiTaskModel:
    """Fits :class:`MultiTaskHeads` on a sparse target matrix. Satisfies ``QSPRModel``."""

    params: dict[str, Any] = field(default_factory=default_mtl_params)
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 512
    seed: int = 0
    device: str = "cpu"
    properties: tuple[PropertyName, ...] = PROPERTIES

    best_epoch: int | None = None
    _net: MultiTaskHeads | None = None
    _centre: npt.NDArray[np.float64] | None = None
    _scale: npt.NDArray[np.float64] | None = None
    _history: list[float] = field(default_factory=list)

    # -- helpers ----------------------------------------------------------- #

    def _standardize(
        self, Y: npt.NDArray[np.float64], M: npt.NDArray[np.bool_]
    ) -> None:
        """Per-property mean and spread over labelled training rows only."""
        centre = np.zeros(len(self.properties))
        scale = np.ones(len(self.properties))
        for j in range(len(self.properties)):
            values = Y[M[:, j], j]
            if values.size > 1:
                centre[j] = float(values.mean())
                scale[j] = float(values.std()) or 1.0
        self._centre, self._scale = centre, scale

    def _to_tensors(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float64],
        M: npt.NDArray[np.bool_],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._centre is not None and self._scale is not None
        standardized = (Y - self._centre) / self._scale
        return (
            torch.as_tensor(X, dtype=torch.float32, device=self.device),
            torch.as_tensor(standardized, dtype=torch.float32, device=self.device),
            torch.as_tensor(M, dtype=torch.float32, device=self.device),
        )

    # -- the QSPRModel contract -------------------------------------------- #

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float64],
        M: npt.NDArray[np.bool_],
        *,
        eval_set: tuple[
            npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.bool_]
        ]
        | None = None,
    ) -> Self:
        torch.manual_seed(self.seed)
        self._standardize(Y, M)

        net = MultiTaskHeads(
            X.shape[1],
            hidden_width=int(self.params["hidden_width"]),
            n_layers=int(self.params["n_layers"]),
            dropout=float(self.params["dropout"]),
            n_properties=len(self.properties),
        ).to(self.device)

        weights = torch.as_tensor(
            [float(self.params["loss_weights"][p]) for p in self.properties],
            dtype=torch.float32,
            device=self.device,
        )
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        Xt, Yt, Mt = self._to_tensors(X, Y, M)
        validation = None
        if eval_set is not None:
            # Standardized with the TRAINING statistics, deliberately. Re-centring
            # the validation set on its own mean would hide exactly the shift the
            # validation set exists to detect.
            Xv, Yv, Mv = eval_set
            validation = self._to_tensors(Xv, Yv, Mv)

        best = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        since_improved = 0
        generator = torch.Generator().manual_seed(self.seed)

        for epoch in range(self.max_epochs):
            net.train()
            order = torch.randperm(Xt.shape[0], generator=generator).to(self.device)
            for start in range(0, Xt.shape[0], self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = masked_mse(net(Xt[batch]), Yt[batch], Mt[batch], weights)
                loss.backward()
                optimizer.step()

            score = (
                self._validation_loss(net, validation, weights) if validation else None
            )
            if score is None:
                continue
            self._history.append(score)
            if score < best - 1e-5:
                best, since_improved = score, 0
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
                self.best_epoch = epoch + 1
            else:
                since_improved += 1
                if since_improved >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net.eval()
        return self

    def _validation_loss(
        self,
        net: MultiTaskHeads,
        validation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
    ) -> float:
        net.eval()
        with torch.no_grad():
            Xv, Yv, Mv = validation
            return float(masked_mse(net(Xv), Yv, Mv, weights))

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Predictions in each property's native units, not standardized ones."""
        if self._net is None or self._centre is None or self._scale is None:
            raise RuntimeError("model has not been fitted")
        with torch.no_grad():
            raw = self._net(torch.as_tensor(X, dtype=torch.float32, device=self.device))
        return raw.cpu().numpy().astype(np.float64) * self._scale + self._centre

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError(
            "The neural track's uncertainty signal is a deep ensemble across seeds, "
            "which arrives with the uncertainty layer at build step 8. Track 2 at "
            "step 6 reports point accuracy only."
        )
