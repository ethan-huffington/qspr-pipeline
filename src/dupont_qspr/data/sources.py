"""Declarative registry of where the raw data comes from.

Separated from the fetching logic so that provenance is readable in one place.
Every source records a citation and a checksum: a dataset used without a pinned
version is no more reproducible than a model shipped without its featurizer.

A note on why these are not the URLs the brief names. The brief specifies
retrieval through the ``tdc`` package, which cannot be installed alongside this
project's numpy 2 / pandas 3 / Python 3.13 environment. Each dataset is therefore
taken from its own upstream publisher instead, which is a step closer to the
origin rather than further from it - AqSolDB comes from the authors' own
repository rather than a redistribution of it. The one cost is that TDC's
canonical scaffold splits are unavailable, which §10 of the brief lists only as an
optional secondary comparability check.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AQSOLDB", "BRADLEY", "LIPOPHILICITY", "SOURCES", "DataSource"]


@dataclass(frozen=True, slots=True)
class DataSource:
    """One raw file, and everything needed to obtain and trust it."""

    key: str
    filename: str
    citation: str
    #: ``None`` when the file cannot be fetched programmatically and must be
    #: placed by hand - see ``manual_instructions``.
    url: str | None
    #: Expected MD5 of the raw file. Checked on every load, not just on download,
    #: so a truncated or substituted file is caught before it reaches curation.
    md5: str | None
    notes: str
    manual_instructions: str | None = None

    @property
    def is_manual(self) -> bool:
        return self.url is None


AQSOLDB = DataSource(
    key="aqsoldb",
    filename="aqsoldb_curated_solubility.csv",
    citation=(
        "Sorkun, M.C., Khetan, A., Er, S. (2019). AqSolDB, a curated reference set "
        "of aqueous solubility and 2D descriptors for a diverse set of compounds. "
        "Scientific Data 6, 143."
    ),
    url="https://raw.githubusercontent.com/mcsorkun/AqSolDB/master/results/data_curated.csv",
    md5="3edc446837425fc1d2dca4b881313e90",
    notes=(
        "9,982 compounds, solubility as log mol/L. Taken from the authors' own "
        "repository. Carries 'SD' and 'Occurrences' columns recording spread across "
        "replicate measurements, and a 'Group' column naming which of the nine "
        "underlying source datasets each value came from - both feed the "
        "measurement-heterogeneity analysis the brief asks for in §14."
    ),
)

LIPOPHILICITY = DataSource(
    key="lipophilicity",
    filename="lipophilicity_astrazeneca.csv",
    citation=(
        "AstraZeneca experimental octanol/water distribution coefficient (logD at "
        "pH 7.4), distributed via MoleculeNet (Wu et al., 2018, Chem. Sci. 9, 513)."
    ),
    url="https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    md5="85a0e1cb8b38b0dfc3f96ff47a57f0ab",
    notes=(
        "4,200 compounds. Columns: CMPD_CHEMBLID, exp, smiles. 'exp' is logD7.4, "
        "not logP - the distinction matters for interpretation but not for the "
        "regression, and the brief treats the column as the lipophilicity target."
    ),
)

BRADLEY = DataSource(
    key="bradley",
    filename="BradleyMeltingPointDataset.xlsx",
    citation=(
        "Bradley, J.-C., Lang, A., Williams, A. (2014). Jean-Claude Bradley Open "
        "Melting Point Dataset. figshare. doi:10.6084/m9.figshare.1031637"
    ),
    # figshare's download host is not reachable from every environment; when it is,
    # the downloader uses this, and when it is not it falls back to the manually
    # placed file. Either way the MD5 below decides whether the file is trusted.
    url="https://ndownloader.figshare.com/files/1503990",
    md5="6a4690f289e8377ef333208f846bcae1",
    notes=(
        "28,645 raw entries. Carries 'donotuse' flags marking salts, mixtures, "
        "decomposition and out-of-range values, plus multiple literature "
        "measurements per compound. Both are load-bearing: the flags drive "
        "exclusion and the replicates are what make the conflict analysis possible."
    ),
    manual_instructions=(
        "Download BradleyMeltingPointDataset.xlsx (2.2 MB) from\n"
        "  https://figshare.com/articles/dataset/"
        "Jean_Claude_Bradley_Open_Melting_Point_Datset/1031637\n"
        "and save it to data/raw/BradleyMeltingPointDataset.xlsx\n"
        "It is verified against MD5 6a4690f289e8377ef333208f846bcae1, so a wrong "
        "or truncated file is rejected rather than silently curated."
    ),
)

SOURCES: tuple[DataSource, ...] = (AQSOLDB, LIPOPHILICITY, BRADLEY)
