"""Curation has to be correct in the ways that are otherwise invisible.

Rejecting a molecule is loud - it shows up in the ledger. The dangerous failures
are the silent ones: a mixture quietly reduced to its largest fragment carrying
another component's measurement, two spellings of one molecule joining as two
rows, replicate measurements averaged so the disagreement disappears. Those are
what these tests are for.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dupont_qspr.contracts import (
    PROPERTIES,
    SMILES_COLUMN,
    UNION_TABLE_SCHEMA,
    source_column,
    validate_union_table,
)
from dupont_qspr.data.curate_mp import KELVIN_OFFSET, curate_bradley, resolve_replicates
from dupont_qspr.data.download import SourceUnavailableError, ensure_source, file_md5
from dupont_qspr.data.sources import SOURCES, DataSource
from dupont_qspr.data.standardize import (
    RejectReason,
    resolve_duplicate_values,
    screen_range,
    standardize_frame,
    standardize_smiles,
)
from dupont_qspr.data.union import build_union_table, label_availability_matrix


class TestStandardizeStructures:
    @pytest.mark.parametrize(
        ("smiles", "expected"),
        [
            ("CCO", "CCO"),
            ("OCC", "CCO"),  # same molecule, written backwards
            ("C(C)O", "CCO"),  # and again with explicit branching
            ("c1ccccc1", "c1ccccc1"),
            ("C1=CC=CC=C1", "c1ccccc1"),  # Kekulé form of benzene
        ],
    )
    def test_equivalent_spellings_collapse_to_one_key(
        self, smiles: str, expected: str
    ) -> None:
        """The property the whole join depends on."""
        assert standardize_smiles(smiles).canonical == expected

    @pytest.mark.parametrize(
        ("smiles", "reason"),
        [
            ("", RejectReason.EMPTY),
            ("   ", RejectReason.EMPTY),
            ("not_a_molecule", RejectReason.UNPARSEABLE),
            ("c1ccccc1.CCO", RejectReason.MIXTURE),
            ("[Na+].[Cl-]", RejectReason.MIXTURE),
        ],
    )
    def test_rejections(self, smiles: str, reason: RejectReason) -> None:
        result = standardize_smiles(smiles)
        assert not result.ok
        assert result.reason is reason

    def test_salt_is_stripped_but_a_mixture_is_refused(self) -> None:
        """Both are several disconnected fragments; only one has an obvious subject."""
        salt = standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        assert salt.ok
        assert salt.stripped_fragments == 1
        assert "Na" not in salt.canonical

        mixture = standardize_smiles("c1ccccc1.CCO")
        assert mixture.reason is RejectReason.MIXTURE

    def test_stoichiometric_copies_are_not_a_mixture(self) -> None:
        """A metal salt of two identical ligands has one unambiguous subject.

        Judged on fragment size alone this looks like a two-component mixture,
        because the two ligands tie for largest. Deduplicating identical fragments
        before the size test is what keeps these ~140 AqSolDB entries.
        """
        result = standardize_smiles(
            "[Zn++].CCC(C)O[P]([S-])(=S)OC(C)CC(C)C.CCC(C)O[P]([S-])(=S)OC(C)CC(C)C"
        )
        assert result.ok
        assert "Zn" not in result.canonical

    @pytest.mark.parametrize(
        "smiles", ["c1cncn1", "c1cccn1", "c1cccc2ncnc12", "Sc1nc2ccccc2n1"]
    )
    def test_aromatic_nitrogen_missing_hydrogen_is_repaired(self, smiles: str) -> None:
        """Pyrrole-type N written as pyridine-type: ~280 Bradley compounds.

        These are ordinary heterocycles, so dropping them would bias the training
        set away from a whole structural class rather than just losing noise.
        """
        result = standardize_smiles(smiles)
        assert result.ok
        assert result.repaired
        assert "[nH]" in result.canonical

    def test_repair_does_not_fire_on_a_structure_that_already_parses(self) -> None:
        assert not standardize_smiles("c1cc[nH]c1").repaired

    def test_charges_are_neutralized_where_unambiguous(self) -> None:
        result = standardize_smiles("CC(=O)[O-]")
        assert result.ok
        assert result.neutralized
        assert result.canonical == "CC(=O)O"

    def test_permanent_charge_is_preserved(self) -> None:
        """A quaternary ammonium is not a protonation state to be undone."""
        result = standardize_smiles("CCCC[N+](C)(C)C")
        assert result.ok
        assert "[N+]" in result.canonical

    @pytest.mark.parametrize(
        ("smiles", "note"),
        [
            (
                "[Mn].C[c-]1cccc1.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
                "cyclopentadienyl anion",
            ),
            ("[Na+].CC(=O)[C-]1C(=O)OC(=CC1=O)C", "sodium enolate"),
        ],
    )
    def test_structural_charges_survive_neutralization(
        self, smiles: str, note: str
    ) -> None:
        """A charge that creates aromaticity is not a protonation state.

        The cyclopentadienyl anion is aromatic *because* of its negative charge.
        Stripping it leaves an all-carbon five-membered ring that cannot be
        aromatic, and RDKit will write that SMILES out but refuse to read it back.
        The round-trip guard catches it and keeps the charged form.
        """
        result = standardize_smiles(smiles)
        assert result.ok
        assert not result.neutralized
        assert "-]" in result.canonical

    def test_every_canonical_output_can_be_read_back(self) -> None:
        """The output is the project's join key, so it must survive a round trip.

        Without this guard a molecule reaches featurization and silently becomes a
        row of NaN — no exception, no ledger entry, just a corrupt feature vector.
        """
        from rdkit import Chem

        inputs = [
            "CCO",
            "CC(=O)[O-]",
            "[Mn].C[c-]1cccc1.[C-]#[O+].[C-]#[O+].[C-]#[O+]",
            "[Na+].CC(=O)[C-]1C(=O)OC(=CC1=O)C",
            "c1cncn1",
            "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",
            "CCCC[N+](C)(C)C",
        ]
        for smiles in inputs:
            result = standardize_smiles(smiles)
            if result.ok:
                assert Chem.MolFromSmiles(result.canonical) is not None, (
                    result.canonical
                )

    def test_stereochemistry_is_kept_by_default(self) -> None:
        assert "@" in standardize_smiles("C[C@H](N)C(=O)O").canonical
        assert (
            "@"
            not in standardize_smiles(
                "C[C@H](N)C(=O)O", keep_stereochemistry=False
            ).canonical
        )


class TestStandardizeFrame:
    def test_rejects_go_to_the_ledger_not_into_the_void(self) -> None:
        frame = pl.DataFrame(
            {"smi": ["CCO", "not_a_molecule", "c1ccccc1.CCO"], "v": [1.0, 2.0, 3.0]}
        )
        kept, ledger = standardize_frame(
            frame, smiles_column="smi", source="test", value_column="v"
        )

        assert kept.height == 1
        assert ledger.height == 2
        assert set(ledger.get_column("reason")) == {"unparseable", "mixture"}
        # The rejected value travels with the reason, so the loss is auditable.
        assert set(ledger.get_column("value")) == {2.0, 3.0}

    def test_screen_range_records_both_kinds_of_loss(self) -> None:
        frame = pl.DataFrame({"smiles": ["A", "B", "C"], "mp_K": [300.0, 5000.0, None]})
        kept, ledger = screen_range(
            frame, column="mp_K", prop="mp_K", bounds=(100.0, 750.0), source="test"
        )

        assert kept.height == 1
        reasons = set(ledger.get_column("reason"))
        assert reasons == {
            str(RejectReason.OUT_OF_RANGE),
            str(RejectReason.MISSING_VALUE),
        }


class TestDuplicateResolution:
    def test_duplicates_collapse_to_median_and_are_recorded(self) -> None:
        frame = pl.DataFrame(
            {"smiles": ["CCO", "CCO", "CCO", "c1ccccc1"], "logS": [1.0, 2.0, 9.0, 5.0]}
        )
        resolved, ledger = resolve_duplicate_values(
            frame, value_column="logS", source="test", keep=[]
        )

        assert resolved.height == 2
        # Median, not mean: one bad transcription must not drag the value with it.
        assert resolved.filter(pl.col("smiles") == "CCO").get_column("logS")[0] == 2.0
        assert ledger.height == 1
        assert "3 rows shared this canonical SMILES" in ledger.get_column("detail")[0]

    def test_replicates_keep_their_spread(self) -> None:
        frame = pl.DataFrame(
            {
                "smiles": ["A", "A", "A", "B", "B"],
                "mp_K": [300.0, 302.0, 301.0, 400.0, 460.0],
                "src": ["x", "y", "z", "x", "y"],
            }
        )
        resolved, conflicts = resolve_replicates(
            frame, value_column="mp_K", tolerance=5.0, source_column="src"
        )

        a = resolved.filter(pl.col("smiles") == "A")
        assert a.get_column("median_k")[0] == 301.0
        assert a.get_column("spread_k")[0] == 2.0
        assert a.get_column("n_measurements")[0] == 3

        # Only B disagrees by more than the tolerance.
        assert conflicts.height == 1
        assert conflicts.get_column("smiles")[0] == "B"
        assert conflicts.get_column("spread_k")[0] == 60.0

    def test_disagreement_is_recorded_rather_than_averaged_away(self) -> None:
        """The spread bounds achievable RMSE, so it must survive curation."""
        frame = pl.DataFrame(
            {"smiles": ["A", "A"], "mp_K": [300.0, 400.0], "src": ["x", "y"]}
        )
        resolved, conflicts = resolve_replicates(
            frame, value_column="mp_K", tolerance=5.0, source_column="src"
        )
        assert resolved.get_column("spread_k")[0] == 100.0
        assert conflicts.height == 1
        assert conflicts.get_column("sources")[0] == "x; y"


class TestBradleyCuration:
    def _raw(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["good", "flagged", "dupe-a", "dupe-b", "bad-smiles"],
                "smiles": ["CCO", "c1ccccc1", "CC(C)O", "CC(C)O", "not_a_molecule"],
                "mpC": [-114.0, 5.5, 20.0, 24.0, 50.0],
                "source": ["s1", "s2", "s3", "s4", "s5"],
                "donotuse": [None, "x", None, None, None],
                "donotuseBecause": [None, "decomposes", None, None, None],
            }
        )

    def test_flagged_rows_are_excluded_with_the_publisher_s_reason(self) -> None:
        curated, ledger, _ = curate_bradley(self._raw())

        assert "c1ccccc1" not in set(curated.get_column("smiles"))
        flagged = ledger.filter(pl.col("stage") == "donotuse_flag")
        assert flagged.height == 1
        assert flagged.get_column("detail")[0] == "decomposes"

    def test_celsius_becomes_kelvin(self) -> None:
        curated, _, _ = curate_bradley(self._raw())
        ethanol = curated.filter(pl.col("smiles") == "CCO")
        assert ethanol.get_column("mp_K")[0] == pytest.approx(-114.0 + KELVIN_OFFSET)

    def test_replicates_collapse_and_keep_their_spread(self) -> None:
        curated, _, _ = curate_bradley(self._raw())
        propanol = curated.filter(pl.col("smiles") == "CC(C)O")

        assert propanol.height == 1
        assert propanol.get_column("mp_n_measurements")[0] == 2
        assert propanol.get_column("mp_spread_k")[0] == pytest.approx(4.0)
        assert propanol.get_column("mp_K")[0] == pytest.approx(22.0 + KELVIN_OFFSET)

    def test_high_purity_subset_drops_disagreeing_replicates(self) -> None:
        raw = self._raw().with_columns(
            pl.when(pl.col("name") == "dupe-b")
            .then(90.0)
            .otherwise(pl.col("mpC"))
            .alias("mpC")
        )
        full, _, _ = curate_bradley(raw, subset="full", tolerance_k=5.0)
        pure, ledger, _ = curate_bradley(
            raw, subset="double_plus_good", tolerance_k=5.0
        )

        assert "CC(C)O" in set(full.get_column("smiles"))
        assert "CC(C)O" not in set(pure.get_column("smiles"))
        assert (ledger.get_column("reason") == "replicates_disagree").any()

    def test_missing_columns_produce_a_readable_error(self) -> None:
        with pytest.raises(ValueError, match="Could not find the SMILES column"):
            curate_bradley(pl.DataFrame({"structure": ["CCO"], "mpC": [10.0]}))


class TestUnionTable:
    def test_sources_join_on_canonical_smiles_not_raw_strings(self) -> None:
        """The join key must survive being written differently by each source.

        Ethanol as 'CCO' in one dataset and 'OCC' in another has to become one
        row with two labels. Joining before canonicalization gives two rows, each
        half-labelled, and nothing downstream reports the mistake.
        """
        solubility = pl.DataFrame(
            {SMILES_COLUMN: [standardize_smiles("CCO").canonical], "logS": [-0.3]}
        )
        lipophilicity = pl.DataFrame(
            {SMILES_COLUMN: [standardize_smiles("OCC").canonical], "logP": [-0.31]}
        )
        table = build_union_table(
            solubility=solubility, lipophilicity=lipophilicity, melting_point=None
        )

        assert table.height == 1
        assert table.get_column("logS")[0] == pytest.approx(-0.3)
        assert table.get_column("logP")[0] == pytest.approx(-0.31)

    def test_result_satisfies_the_union_table_contract(self) -> None:
        table = build_union_table(
            solubility=pl.DataFrame(
                {SMILES_COLUMN: ["CCO", "CCC"], "logS": [-0.3, -1.1]}
            ),
            lipophilicity=pl.DataFrame({SMILES_COLUMN: ["CCO"], "logP": [-0.31]}),
            melting_point=pl.DataFrame({SMILES_COLUMN: ["c1ccccc1"], "mp_K": [278.7]}),
        )
        validate_union_table(table.select(list(UNION_TABLE_SCHEMA)))

    def test_missing_sources_still_produce_the_full_schema(self) -> None:
        """Downstream code must never branch on which datasets happened to load."""
        table = build_union_table(
            solubility=pl.DataFrame({SMILES_COLUMN: ["CCO"], "logS": [-0.3]}),
            lipophilicity=None,
            melting_point=None,
        )
        for prop in PROPERTIES:
            assert prop in table.columns
            assert source_column(prop) in table.columns
        assert table.get_column("logP").null_count() == 1

    def test_targets_are_never_imputed(self) -> None:
        table = build_union_table(
            solubility=pl.DataFrame({SMILES_COLUMN: ["CCO"], "logS": [-0.3]}),
            lipophilicity=pl.DataFrame({SMILES_COLUMN: ["CCC"], "logP": [1.8]}),
            melting_point=None,
        )
        assert table.height == 2
        # A molecule labelled by only one source keeps genuine nulls elsewhere.
        assert table.get_column("logS").null_count() == 1
        assert table.get_column("logP").null_count() == 1

    def test_availability_matrix_counts_combinations(self) -> None:
        table = build_union_table(
            solubility=pl.DataFrame(
                {SMILES_COLUMN: ["CCO", "CCC"], "logS": [-0.3, -1.1]}
            ),
            lipophilicity=pl.DataFrame({SMILES_COLUMN: ["CCO"], "logP": [-0.31]}),
            melting_point=None,
        )
        matrix = label_availability_matrix(table)

        assert matrix["n_molecules"] == 2
        assert matrix["per_property"] == {"logS": 2, "logP": 1, "mp_K": 0}
        assert matrix["pairwise_overlap"]["logS&logP"] == 1
        assert matrix["per_combination"] == {"logS": 1, "logS+logP": 1}
        assert matrix["n_complete_rows"] == 0


class TestSourceIntegrity:
    def test_every_source_declares_a_checksum_and_citation(self) -> None:
        for source in SOURCES:
            assert source.md5, f"{source.key} has no checksum to verify against"
            assert source.citation, f"{source.key} has no citation"

    def test_a_corrupted_file_is_refused_rather_than_curated(
        self, tmp_path: Path
    ) -> None:
        """A truncated download must fail here, not silently halve the dataset."""
        source = DataSource(
            key="fake",
            filename="fake.csv",
            citation="none",
            url=None,
            md5="0" * 32,
            notes="",
            manual_instructions="place it by hand",
        )
        (tmp_path / "fake.csv").write_text("smiles,value\nCCO,1.0\n")

        with pytest.raises(SourceUnavailableError, match="expected 0{32}"):
            ensure_source(source, tmp_path)

    def test_a_matching_file_is_accepted_from_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "fake.csv"
        path.write_text("smiles,value\nCCO,1.0\n")
        source = DataSource(
            key="fake",
            filename="fake.csv",
            citation="none",
            url=None,
            md5=file_md5(path),
            notes="",
        )

        acquisition = ensure_source(source, tmp_path)
        assert acquisition.action == "cached"
        assert acquisition.verified
