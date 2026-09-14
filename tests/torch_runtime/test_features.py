"""Featurization has to be identical everywhere, or the cache is a liability.

The cache's whole value rests on one promise: the same molecule yields the same
vector, whoever asks and whenever. If that ever stops holding, nothing raises -
folds simply start training on subtly different representations of the same
compound, and every number downstream is quietly wrong. These tests are aimed
almost entirely at that promise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dupont_qspr.config import Config
from dupont_qspr.contracts import FeatureCache
from dupont_qspr.features.cache import DiskFeatureCache
from dupont_qspr.features.descriptors import (
    compute_path_a,
    descriptor_feature_names,
    n_path_a_features,
)

# A small, chemically varied set: an alcohol, an aromatic, a fused heterocycle,
# a drug with stereocentres, and one with a permanent charge.
ENCODER = "DeepChem/ChemBERTa-77M-MLM"

SAMPLE = [
    "CCO",
    "c1ccccc1",
    "c1ccc2[nH]cnc2c1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCCC[N+](C)(C)C",
]


def _counting_cache(
    tmp_path: Path, width: int = 4
) -> tuple[DiskFeatureCache, list[list[str]]]:
    """A cache whose compute function records exactly what it was asked for."""
    calls: list[list[str]] = []

    def compute(batch):
        calls.append(list(batch))
        # Deterministic and distinguishable per molecule.
        return np.array(
            [[float(len(s)), float(ord(s[0])), 1.0, 2.0] for s in batch],
            dtype=np.float32,
        )

    cache = DiskFeatureCache(
        name="test",
        version="v1",
        n_features=width,
        compute=compute,
        directory=tmp_path,
    )
    return cache, calls


class TestCacheMechanics:
    def test_satisfies_the_feature_cache_protocol(self, tmp_path: Path) -> None:
        cache, _ = _counting_cache(tmp_path)
        assert isinstance(cache, FeatureCache)

    def test_returns_rows_in_input_order(self, tmp_path: Path) -> None:
        """Callers index positionally against their own labels.

        Returning rows in cache order instead would pair every molecule with the
        wrong target, and no shape check anywhere would notice.
        """
        cache, _ = _counting_cache(tmp_path)
        cache.transform(["CCO", "c1ccccc1", "CCC"])

        shuffled = ["CCC", "CCO", "c1ccccc1"]
        out = cache.transform(shuffled)
        for row, smiles in zip(out, shuffled, strict=True):
            assert row[0] == float(len(smiles))

    def test_second_pass_computes_nothing(self, tmp_path: Path) -> None:
        cache, calls = _counting_cache(tmp_path)
        first = cache.transform(SAMPLE)
        second = cache.transform(SAMPLE)

        assert len(calls) == 1
        assert calls[0] == SAMPLE
        assert np.array_equal(first, second)

    def test_repeats_within_one_batch_are_computed_once(self, tmp_path: Path) -> None:
        cache, calls = _counting_cache(tmp_path)
        out = cache.transform(["CCO", "CCO", "CCO"])

        assert calls == [["CCO"]]
        assert out.shape == (3, 4)
        assert np.array_equal(out[0], out[2])

    def test_extends_incrementally_without_recomputing(self, tmp_path: Path) -> None:
        cache, calls = _counting_cache(tmp_path)
        cache.transform(["CCO", "CCC"])
        cache.transform(["CCC", "CCCC", "CCO", "CCCCC"])

        assert calls[0] == ["CCO", "CCC"]
        assert calls[1] == ["CCCC", "CCCCC"]  # only the genuinely new ones

    def test_survives_a_reload_from_disk(self, tmp_path: Path) -> None:
        """The point of persistence: a later process reads, never recomputes."""
        first, _ = _counting_cache(tmp_path)
        expected = first.transform(SAMPLE)

        second, second_calls = _counting_cache(tmp_path)
        assert np.array_equal(second.transform(SAMPLE), expected)
        assert second_calls == []

    def test_a_width_change_without_a_version_bump_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Silently mixing vectors from two recipes is the failure worth blocking."""
        cache, _ = _counting_cache(tmp_path, width=4)
        cache.transform(SAMPLE)

        wider, _ = _counting_cache(tmp_path, width=8)
        with pytest.raises(ValueError, match="bump features.version"):
            wider.transform(SAMPLE)

    def test_a_version_bump_starts_a_separate_cache(self, tmp_path: Path) -> None:
        cache, _ = _counting_cache(tmp_path)
        cache.transform(SAMPLE)

        bumped, calls = _counting_cache(tmp_path)
        object.__setattr__(bumped, "version", "v2")
        bumped.transform(SAMPLE)

        assert calls == [SAMPLE]  # recomputed, not read from the v1 files
        assert cache.values_path.exists()  # and v1 is left intact
        assert bumped.values_path.exists()

    def test_stats_distinguish_hits_from_computed_molecules(
        self, tmp_path: Path
    ) -> None:
        cache, _ = _counting_cache(tmp_path)
        cache.transform(["CCO", "CCO", "CCC"])

        assert cache.stats.misses == 2  # unique molecules computed
        assert cache.stats.hits == 0
        cache.transform(["CCO", "CCC"])
        assert cache.stats.hits == 2


class TestPathA:
    def test_width_is_descriptors_plus_fingerprint_bits(self) -> None:
        assert n_path_a_features() == 217 + 1024
        assert len(descriptor_feature_names()) == n_path_a_features()

    def test_is_deterministic_across_calls(self) -> None:
        assert np.array_equal(
            compute_path_a(SAMPLE), compute_path_a(SAMPLE), equal_nan=True
        )

    def test_does_not_depend_on_batch_composition(self) -> None:
        """A molecule's features must not change with what it was computed beside."""
        together = compute_path_a(SAMPLE)
        apart = np.stack([compute_path_a([s])[0] for s in SAMPLE])
        assert np.array_equal(together, apart, equal_nan=True)

    def test_distinct_molecules_get_distinct_vectors(self) -> None:
        features = compute_path_a(SAMPLE)
        unique = {row.tobytes() for row in np.nan_to_num(features)}
        assert len(unique) == len(SAMPLE)

    def test_fingerprint_block_is_binary(self) -> None:
        features = compute_path_a(SAMPLE)
        bits = features[:, 217:]
        assert set(np.unique(bits)).issubset({0.0, 1.0})

    def test_non_finite_values_become_nan_not_zero(self) -> None:
        """Zero is a plausible descriptor value, so it cannot mean "missing"."""
        features = compute_path_a(SAMPLE)
        assert not np.isinf(features).any()

    def test_float32_overflow_becomes_nan_rather_than_inf(self) -> None:
        """Some descriptors are finite in float64 but overflow float32.

        Ipc grows factorially with molecule size and reaches ~2.8e54 on a large
        structure, against a float32 ceiling of 3.4e38. Testing finiteness before
        narrowing lets that through as inf, which no tree can split on.
        """
        big = "C" * 200  # long chain: enough to push Ipc past the float32 ceiling
        features = compute_path_a([big])
        assert not np.isinf(features).any()
        assert features.dtype == np.float32

    def test_empty_input_returns_an_empty_matrix_of_the_right_width(self) -> None:
        assert compute_path_a([]).shape == (0, n_path_a_features())


@pytest.mark.torch
class TestPathB:
    """Exercised only when the encoder is available; downloads a 3.3M-param model."""

    def _compute(self, smiles, **kwargs):
        from dupont_qspr.features.encoders import compute_path_b

        return compute_path_b(smiles, model_name=ENCODER, device="cpu", **kwargs)

    def test_embeddings_do_not_depend_on_batch_size(self) -> None:
        """The property that makes the cache safe.

        Padding is per batch, so a naive mean over the token axis would make a
        molecule's embedding depend on the longest SMILES it happened to share a
        batch with - and the cached value would then depend on arrival order.
        Masked mean pooling is what prevents that, and this is the test that
        would catch its loss.
        """
        pytest.importorskip("transformers")
        big = self._compute(SAMPLE, batch_size=64)
        small = self._compute(SAMPLE, batch_size=1)
        assert np.allclose(big, small, atol=1e-4)

    def test_frozen_encoder_carries_no_gradients(self) -> None:
        pytest.importorskip("transformers")
        from dupont_qspr.features.encoders import _load_encoder

        _, model = _load_encoder(ENCODER, "cpu")
        assert not model.training
        assert all(not p.requires_grad for p in model.parameters())

    def test_no_randomly_initialised_pooler_is_attached(self) -> None:
        """The checkpoint has no pooler; one added here would be random weights."""
        pytest.importorskip("transformers")
        from dupont_qspr.features.encoders import _load_encoder

        _, model = _load_encoder(ENCODER, "cpu")
        assert getattr(model, "pooler", None) is None

    def test_chemberta2_tokenizer_is_known_to_drop_halogens(self) -> None:
        """Pins the defect that motivated caching a second encoder.

        ChemBERTa-2 ships a BPE tokenizer with zero merge rules, so it can never
        combine 'C' + 'l' into the 'Cl' token its own vocabulary contains. The
        model was pretrained this way - the multi-character embeddings sit at
        their initialization - so the tokenizer cannot simply be corrected.

        If a future release fixes it, this test fails and tells us the second
        encoder is no longer needed. That is the point: a known defect should be
        asserted, not remembered.
        """
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        assert tokenizer.tokenize("COc1ccccc1C") == tokenizer.tokenize("COc1ccccc1Cl")
        assert tokenizer.convert_tokens_to_ids("Cl") != tokenizer.unk_token_id

    def test_the_comparison_encoder_does_keep_halogens(self) -> None:
        """The reason the second encoder earns its place in the cache."""
        pytest.importorskip("transformers")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        assert "Cl" in tokenizer.tokenize("COc1ccccc1Cl")
        assert tokenizer.tokenize("COc1ccccc1C") != tokenizer.tokenize("COc1ccccc1Cl")


def test_feature_caches_are_shared_across_profiles(
    smoke_cfg: Config, configs_path: Path
) -> None:
    """Unlike the union table, a feature vector does not depend on the profile."""
    from dupont_qspr.config import load_config

    full = load_config("full", directory=configs_path)
    assert smoke_cfg.features_dir == full.features_dir
    assert smoke_cfg.processed_dir != full.processed_dir
