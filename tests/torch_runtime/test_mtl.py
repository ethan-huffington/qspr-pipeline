"""Track 2, and the silent failure the masked loss exists to prevent.

If an unlabelled cell contributes gradient, nothing breaks visibly. The network
trains, the loss falls, and the model quietly learns that "missing" means whatever
the sanitized placeholder happened to be. Every number downstream would be wrong
and every diagnostic would look healthy. So the mask is tested at the level it
actually operates: gradients, not outputs.

These tests must not import xgboost, and one of them asserts exactly that.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from dupont_qspr.contracts import PROPERTIES
from dupont_qspr.models.mtl import (
    MultiTaskHeads,
    MultiTaskModel,
    default_mtl_params,
    masked_mse,
)


@pytest.fixture
def sparse() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A sparse target matrix shaped like the real one: mostly one label per row."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(240, 16)).astype(np.float32)
    weights = rng.normal(size=(16, 3))
    Y = X @ weights + rng.normal(scale=0.1, size=(240, 3))
    M = rng.random((240, 3)) < 0.45
    M[np.arange(240), rng.integers(0, 3, 240)] = True  # every row keeps ≥1 label
    Y[~M] = np.nan
    return X, Y.astype(np.float64), M


class TestMaskedLoss:
    def test_unlabelled_property_produces_exactly_zero_gradient(self) -> None:
        """The headline guarantee, checked where it actually lives.

        With no labelled rows for logP anywhere in the batch, that head's weights
        must receive gradient of exactly zero — not merely small.
        """
        torch.manual_seed(0)
        net = MultiTaskHeads(8, hidden_width=16, n_layers=1, dropout=0.0)
        x = torch.randn(32, 8)
        target = torch.randn(32, 3)
        mask = torch.ones(32, 3)
        mask[:, 1] = 0.0  # logP entirely unlabelled
        target[:, 1] = float("nan")

        loss = masked_mse(net(x), target, mask, torch.ones(3))
        loss.backward()

        logp_head = net.heads[1]
        assert torch.count_nonzero(logp_head.weight.grad) == 0
        assert torch.count_nonzero(logp_head.bias.grad) == 0
        # The labelled heads must still be learning.
        assert torch.count_nonzero(net.heads[0].weight.grad) > 0

    def test_poisoning_unlabelled_targets_changes_nothing(self) -> None:
        """An absurd value in a masked cell must be invisible to the optimiser."""

        def gradients(fill: float) -> torch.Tensor:
            torch.manual_seed(0)
            net = MultiTaskHeads(8, hidden_width=16, n_layers=1, dropout=0.0)
            target = torch.randn(32, 3)
            mask = (torch.rand(32, 3) < 0.6).float()
            target[mask == 0] = fill
            masked_mse(net(torch.randn(32, 8)), target, mask, torch.ones(3)).backward()
            return net.shared[0].weight.grad.clone()

        torch.manual_seed(1)
        baseline = gradients(float("nan"))
        torch.manual_seed(1)
        poisoned = gradients(1e9)
        assert torch.allclose(baseline, poisoned)

    def test_nan_targets_do_not_produce_a_nan_loss(self) -> None:
        """nan × 0 is nan, not zero — the trap the sanitisation step exists for."""
        target = torch.full((16, 3), float("nan"))
        target[:, 0] = 1.0
        mask = torch.zeros(16, 3)
        mask[:, 0] = 1.0

        loss = masked_mse(torch.zeros(16, 3), target, mask, torch.ones(3))
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(1.0)

    def test_each_property_is_normalised_by_its_own_label_count(self) -> None:
        """A property with more labels must not dominate by sheer numerosity.

        Here logS has 100 labelled rows with error 1 and logP has 4 with error 1.
        Under a global mean, logS would drive the loss 25:1. Under per-property
        normalisation they contribute equally.
        """
        prediction = torch.zeros(100, 3)
        target = torch.full((100, 3), float("nan"))
        mask = torch.zeros(100, 3)
        target[:, 0], mask[:, 0] = 1.0, 1.0
        target[:4, 1], mask[:4, 1] = 1.0, 1.0

        loss = masked_mse(prediction, target, mask, torch.ones(3))
        assert loss.item() == pytest.approx(1.0)

    def test_loss_weights_shift_the_balance(self) -> None:
        prediction = torch.zeros(20, 3)
        target = torch.zeros(20, 3)
        target[:, 0] = 2.0  # logS is wrong
        mask = torch.ones(20, 3)

        light = masked_mse(prediction, target, mask, torch.tensor([0.2, 1.0, 1.0]))
        heavy = masked_mse(prediction, target, mask, torch.tensor([5.0, 1.0, 1.0]))
        assert heavy > light

    def test_a_property_absent_from_the_batch_is_excluded_not_zeroed(self) -> None:
        """Counting an absent property as zero error would flatter the loss."""
        prediction = torch.zeros(10, 3)
        target = torch.full((10, 3), float("nan"))
        mask = torch.zeros(10, 3)
        target[:, 2], mask[:, 2] = 3.0, 1.0

        loss = masked_mse(prediction, target, mask, torch.ones(3))
        assert loss.item() == pytest.approx(9.0)  # not 9/3


class TestMultiTaskModel:
    def test_standardisation_uses_only_the_rows_it_was_fitted_on(self, sparse) -> None:
        """Taking scale from the whole dataset would leak the test distribution."""
        X, Y, M = sparse
        subset = np.arange(0, 120)
        model = MultiTaskModel(max_epochs=2, patience=1).fit(
            X[subset], Y[subset], M[subset]
        )

        for j in range(len(PROPERTIES)):
            values = Y[subset][M[subset, j], j]
            assert model._centre[j] == pytest.approx(float(values.mean()))
            assert model._scale[j] == pytest.approx(float(values.std()))

    def test_predictions_come_back_in_native_units(self, sparse) -> None:
        """The heads work in z-scores; callers must never see them."""
        X, Y, M = sparse
        model = MultiTaskModel(max_epochs=25, patience=25).fit(X, Y, M)
        predicted = model.predict(X)

        assert predicted.shape == (X.shape[0], len(PROPERTIES))
        for j in range(len(PROPERTIES)):
            observed = Y[M[:, j], j]
            # Within an order of magnitude of the target's own location.
            assert abs(predicted[:, j].mean() - observed.mean()) < 3 * observed.std()

    def test_learns_something_on_a_learnable_signal(self, sparse) -> None:
        X, Y, M = sparse
        split = 180
        model = MultiTaskModel(
            params={**default_mtl_params(), "learning_rate": 5e-3},
            max_epochs=200,
            patience=30,
        ).fit(
            X[:split],
            Y[:split],
            M[:split],
            eval_set=(X[split:], Y[split:], M[split:]),
        )
        predicted = model.predict(X[split:])
        for j in range(len(PROPERTIES)):
            keep = M[split:, j]
            if keep.sum() < 5:
                continue
            truth = Y[split:][keep, j]
            error = float(np.sqrt(np.mean((truth - predicted[keep, j]) ** 2)))
            assert error < float(np.std(truth))  # beats predicting the mean

    def test_predicting_before_fitting_raises(self, sparse) -> None:
        X, _, _ = sparse
        with pytest.raises(RuntimeError, match="not been fitted"):
            MultiTaskModel().predict(X)

    def test_quantiles_point_at_the_step_that_implements_them(self, sparse) -> None:
        X, Y, M = sparse
        model = MultiTaskModel(max_epochs=2, patience=1).fit(X, Y, M)
        with pytest.raises(NotImplementedError, match="build step 8"):
            model.predict_quantiles(X, [0.05, 0.5, 0.95])

    def test_is_reproducible_for_a_fixed_seed(self, sparse) -> None:
        X, Y, M = sparse
        first = (
            MultiTaskModel(seed=7, max_epochs=20, patience=20).fit(X, Y, M).predict(X)
        )
        second = (
            MultiTaskModel(seed=7, max_epochs=20, patience=20).fit(X, Y, M).predict(X)
        )
        assert np.allclose(first, second)


def test_the_neural_path_never_imports_xgboost() -> None:
    """Importing xgboost into this process would poison it.

    XGBoost and PyTorch link separate OpenMP runtimes and cannot coexist. The
    models package therefore imports nothing eagerly; this asserts that the
    arrangement still holds, since a stray convenience import in ``__init__``
    would reintroduce the hang with no other visible symptom.
    """
    assert "xgboost" not in sys.modules
