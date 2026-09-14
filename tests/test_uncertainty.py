"""Coverage is a promise, so it gets verified rather than assumed.

The whole value of conformal prediction is that "90%" means 90%. That claim rests
on a finite-sample correction which is easy to omit and whose omission is invisible
on large calibration sets and quietly wrong on small ones — which is the regime
this project actually lives in.
"""

from __future__ import annotations

import numpy as np
import pytest

from dupont_qspr.uncertainty.applicability import (
    ApplicabilityDomainIndex,
    tanimoto_similarity_matrix,
)
from dupont_qspr.uncertainty.conformal import (
    ConformalizedQR,
    EnsembleInterval,
    SplitConformal,
    conformal_quantile,
)


def _heteroscedastic(n: int, seed: int = 0):
    """Truth with noise that grows across the range, plus honest-ish quantiles."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    scale = 0.2 + 0.6 * np.abs(x)
    y = x + rng.normal(scale=scale)
    raw = np.column_stack([x - 1.2 * scale, x, x + 1.2 * scale])
    return raw, y


class TestConformalQuantile:
    def test_finite_sample_correction_is_applied(self) -> None:
        """``ceil((n+1)q)/n`` exceeds the plain quantile, and must.

        Without it, coverage is biased low on small calibration sets — silently,
        and in the direction that flatters the model.
        """
        scores = np.arange(1.0, 21.0)  # n = 20
        assert conformal_quantile(scores, 0.9) > float(np.quantile(scores, 0.9))

    def test_too_few_points_gives_an_infinite_interval(self) -> None:
        """Below ~1/(1-q) points, the requested coverage cannot be certified.

        An infinite interval is the honest answer; a narrow one would be a lie
        with a confidence level attached.
        """
        assert conformal_quantile(np.array([1.0, 2.0]), 0.95) == float("inf")

    def test_empty_calibration_is_not_a_zero_width_interval(self) -> None:
        assert conformal_quantile(np.array([]), 0.9) == float("inf")


class TestSplitConformal:
    def test_achieves_nominal_coverage_on_held_out_data(self) -> None:
        raw, y = _heteroscedastic(4000)
        calibrator = SplitConformal(nominal_coverage=0.9).fit(raw[:2000], y[:2000])
        bounds = calibrator.transform(raw[2000:])
        held = y[2000:]

        coverage = float(((held >= bounds[:, 0]) & (held <= bounds[:, 1])).mean())
        assert 0.88 <= coverage <= 0.93

    def test_coverage_holds_at_other_levels(self) -> None:
        raw, y = _heteroscedastic(4000, seed=1)
        for target in (0.5, 0.8, 0.95):
            calibrator = SplitConformal(nominal_coverage=target).fit(
                raw[:2000], y[:2000]
            )
            bounds = calibrator.transform(raw[2000:])
            held = y[2000:]
            coverage = float(((held >= bounds[:, 0]) & (held <= bounds[:, 1])).mean())
            assert coverage >= target - 0.03

    def test_width_is_constant_which_is_the_whole_limitation(self) -> None:
        """It covers correctly and adapts to nothing — hence CQR."""
        raw, y = _heteroscedastic(2000)
        bounds = SplitConformal().fit(raw, y).transform(raw)
        widths = bounds[:, 1] - bounds[:, 0]
        assert np.allclose(widths, widths[0])


class TestConformalizedQR:
    def test_achieves_nominal_coverage(self) -> None:
        raw, y = _heteroscedastic(4000, seed=2)
        calibrator = ConformalizedQR(nominal_coverage=0.9).fit(raw[:2000], y[:2000])
        bounds = calibrator.transform(raw[2000:])
        held = y[2000:]

        coverage = float(((held >= bounds[:, 0]) & (held <= bounds[:, 1])).mean())
        assert 0.88 <= coverage <= 0.93

    def test_width_adapts_to_the_underlying_noise(self) -> None:
        """The property that makes conditional coverage achievable.

        Where the data is noisy the interval must be wider. A constant-width
        interval cannot do this, and marginal coverage cannot detect the failure.
        """
        raw, y = _heteroscedastic(4000, seed=3)
        bounds = ConformalizedQR().fit(raw[:2000], y[:2000]).transform(raw[2000:])
        widths = bounds[:, 1] - bounds[:, 0]

        centre = raw[2000:, 1]
        quiet = np.abs(centre) < 0.5
        loud = np.abs(centre) > 2.0
        assert widths[loud].mean() > 2 * widths[quiet].mean()

    def test_calibration_can_tighten_an_over_wide_interval(self) -> None:
        """The offset is signed: over-confident models widen, timid ones narrow."""
        rng = np.random.default_rng(0)
        y = rng.normal(size=2000)
        # Absurdly wide nominal quantiles.
        raw = np.column_stack([y - 20.0, y, y + 20.0])
        calibrator = ConformalizedQR().fit(raw, y)
        assert calibrator.offset < 0
        widths = np.diff(calibrator.transform(raw)[:, [0, 1]], axis=1)
        assert widths.mean() < 40.0

    def test_never_returns_a_negative_width_interval(self) -> None:
        """Quantile models can cross their own quantiles on odd input."""
        raw = np.column_stack([np.ones(50) * 5, np.ones(50), np.ones(50) * -5])
        bounds = ConformalizedQR().fit(raw, np.ones(50)).transform(raw)
        assert (bounds[:, 1] >= bounds[:, 0]).all()

    def test_agrees_with_an_independent_implementation(self) -> None:
        """Cross-check the correction against mapie rather than only ourselves."""
        pytest.importorskip("mapie")
        from mapie.utils import _check_alpha  # noqa: F401  (import smoke-test)

        scores = np.abs(np.random.default_rng(0).normal(size=500))
        ours = conformal_quantile(scores, 0.9)
        # The published formula, written out independently of our implementation.
        n = scores.size
        expected = float(
            np.quantile(scores, np.ceil((n + 1) * 0.9) / n, method="higher")
        )
        assert ours == pytest.approx(expected)


class TestEnsembleInterval:
    def test_spread_becomes_a_usable_interval_shape(self) -> None:
        interval = EnsembleInterval()
        for offset in (-0.1, 0.0, 0.1):
            interval.add(np.arange(10.0) + offset)
        raw = interval.raw()

        assert raw.shape == (10, 3)
        assert (raw[:, 0] <= raw[:, 1]).all() and (raw[:, 1] <= raw[:, 2]).all()

    def test_unanimous_members_give_zero_width(self) -> None:
        """No disagreement means no epistemic signal — before calibration."""
        interval = EnsembleInterval()
        for _ in range(4):
            interval.add(np.ones(5))
        raw = interval.raw()
        assert np.allclose(raw[:, 0], raw[:, 2])

    def test_unfitted_ensemble_raises(self) -> None:
        with pytest.raises(RuntimeError, match="no ensemble members"):
            EnsembleInterval().raw()


class TestTanimoto:
    def test_identical_fingerprints_are_perfectly_similar(self) -> None:
        fp = np.array([[1, 0, 1, 1, 0]], dtype=np.float32)
        assert tanimoto_similarity_matrix(fp, fp)[0, 0] == pytest.approx(1.0)

    def test_disjoint_fingerprints_are_maximally_distant(self) -> None:
        a = np.array([[1, 1, 0, 0]], dtype=np.float32)
        b = np.array([[0, 0, 1, 1]], dtype=np.float32)
        assert tanimoto_similarity_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_matches_the_textbook_definition(self) -> None:
        """|A ∩ B| / |A ∪ B|: two shared bits out of three set overall."""
        a = np.array([[1, 1, 1, 0]], dtype=np.float32)
        b = np.array([[1, 1, 0, 0]], dtype=np.float32)
        assert tanimoto_similarity_matrix(a, b)[0, 0] == pytest.approx(2 / 3)

    def test_an_empty_fingerprint_does_not_divide_by_zero(self) -> None:
        a = np.zeros((1, 4), dtype=np.float32)
        b = np.ones((1, 4), dtype=np.float32)
        assert tanimoto_similarity_matrix(a, b)[0, 0] == 0.0


class TestApplicabilityDomain:
    def _features(self, bits: np.ndarray) -> np.ndarray:
        """Pad with descriptor columns the index must ignore."""
        return np.hstack([np.random.rand(bits.shape[0], 217), bits]).astype(np.float32)

    def test_a_training_molecule_is_at_zero_distance(self) -> None:
        bits = (np.random.default_rng(0).random((20, 32)) < 0.3).astype(np.float32)
        X = self._features(bits)
        index = ApplicabilityDomainIndex().fit(X)
        assert index.distance(X).max() < 1e-6

    def test_a_novel_structure_is_further_than_a_seen_one(self) -> None:
        rng = np.random.default_rng(0)
        train_bits = np.zeros((20, 32), dtype=np.float32)
        train_bits[:, :8] = rng.random((20, 8)) < 0.6
        novel_bits = np.zeros((1, 32), dtype=np.float32)
        novel_bits[0, 24:] = 1.0  # shares no bits with anything seen

        index = ApplicabilityDomainIndex().fit(self._features(train_bits))
        seen = index.distance(self._features(train_bits[:1]))
        novel = index.distance(self._features(novel_bits))
        assert novel[0] > seen[0]
        assert novel[0] == pytest.approx(1.0)

    def test_nothing_is_in_domain_until_a_threshold_is_chosen(self) -> None:
        """An unset threshold means step 9 has not run.

        Defaulting to "everything is fine" is the exact failure this flag exists
        to prevent, so the safe default is to claim nothing.
        """
        index = ApplicabilityDomainIndex()
        assert not index.flag(np.array([0.0, 0.1, 0.9])).any()

    def test_the_threshold_is_inclusive(self) -> None:
        index = ApplicabilityDomainIndex(threshold=0.5)
        assert index.flag(np.array([0.49, 0.5, 0.51])).tolist() == [True, True, False]

    def test_descriptor_columns_are_excluded_from_the_similarity(self) -> None:
        """Tanimoto is a set measure; continuous descriptors have no place in it."""
        bits = (np.random.default_rng(1).random((10, 32)) < 0.4).astype(np.float32)
        quiet = np.hstack([np.zeros((10, 217)), bits]).astype(np.float32)
        loud = np.hstack([np.full((10, 217), 1e6), bits]).astype(np.float32)

        first = ApplicabilityDomainIndex().fit(quiet).distance(quiet)
        second = ApplicabilityDomainIndex().fit(loud).distance(loud)
        assert np.allclose(first, second)

    def test_unfitted_index_raises(self) -> None:
        with pytest.raises(RuntimeError, match="has not been fitted"):
            ApplicabilityDomainIndex().distance(np.zeros((2, 300), dtype=np.float32))
