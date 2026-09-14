"""Fetch raw files into ``data/raw/``, and refuse to proceed on an unverified one.

Downloads are idempotent: a file already present and matching its recorded MD5 is
left alone, so re-running the pipeline costs nothing and works offline.

The checksum is checked on *load*, not only on download. That is the difference
between catching a truncated or substituted file and quietly curating it - the
failure mode a silent half-download produces is a model trained on 60% of the data
with no indication anything went wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests

from dupont_qspr.data.sources import SOURCES, DataSource

__all__ = ["SourceUnavailableError", "ensure_all", "ensure_source", "file_md5"]

_CHUNK = 1 << 16
_TIMEOUT = 120


class SourceUnavailableError(RuntimeError):
    """A raw file could not be obtained, with instructions for fixing it."""


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Acquisition:
    """What happened when a source was requested."""

    source: DataSource
    path: Path
    action: str  # "cached" | "downloaded" | "placed-by-hand"
    md5: str
    verified: bool

    def describe(self) -> str:
        mark = "ok" if self.verified else "UNVERIFIED"
        return f"{self.source.key:<14} {self.action:<15} {mark:<11} {self.path.name}"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
            response.raise_for_status()
            # Some hosts answer 202 with an empty body when a request is deferred
            # or refused by an intermediary. Treating that as success would write
            # a zero-byte file and fail much later, in the parser.
            if response.status_code != 200:
                raise SourceUnavailableError(
                    f"{url} returned HTTP {response.status_code} rather than 200"
                )
            with partial.open("wb") as handle:
                for chunk in response.iter_content(_CHUNK):
                    handle.write(chunk)
        if partial.stat().st_size == 0:
            raise SourceUnavailableError(f"{url} returned an empty body")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def ensure_source(
    source: DataSource, raw_dir: Path, *, refresh: bool = False
) -> Acquisition:
    """Make ``source`` present in ``raw_dir`` and verified, or raise.

    Order of preference: an existing verified file, then a download, then a clear
    instruction to place the file by hand. A source whose host is unreachable is a
    normal outcome here, not a crash - the message says exactly what to do.
    """
    path = raw_dir / source.filename

    if path.exists() and not refresh:
        digest = file_md5(path)
        if source.md5 is None or digest == source.md5:
            return Acquisition(
                source, path, "cached", digest, verified=source.md5 is not None
            )
        raise SourceUnavailableError(
            f"{path} is present but its MD5 is {digest}, expected {source.md5}.\n"
            "The file is truncated, modified, or a different release. Delete it and "
            "re-run, or pass refresh=True."
        )

    if source.url is not None:
        try:
            _download(source.url, path)
        except (requests.RequestException, SourceUnavailableError) as error:
            raise SourceUnavailableError(
                f"Could not download {source.key} from {source.url}\n"
                f"  reason: {error}\n\n"
                + (source.manual_instructions or "No manual fallback is documented.")
            ) from error
    else:
        raise SourceUnavailableError(
            f"{source.key} has no programmatic source.\n\n"
            + (source.manual_instructions or "")
        )

    digest = file_md5(path)
    if source.md5 is not None and digest != source.md5:
        path.unlink(missing_ok=True)
        raise SourceUnavailableError(
            f"Downloaded {source.key} but its MD5 is {digest}, expected {source.md5}. "
            "The upstream file has changed; update sources.py deliberately rather "
            "than loosening this check."
        )
    return Acquisition(
        source, path, "downloaded", digest, verified=source.md5 is not None
    )


def ensure_all(
    raw_dir: Path, *, refresh: bool = False, required: bool = True
) -> tuple[list[Acquisition], list[SourceUnavailableError]]:
    """Acquire every registered source.

    With ``required=False`` a source that cannot be obtained is collected rather
    than raised, so the rest of the pipeline can still run on what is available -
    useful while one dataset is waiting on a manual download.
    """
    acquired: list[Acquisition] = []
    failures: list[SourceUnavailableError] = []
    for source in SOURCES:
        try:
            acquired.append(ensure_source(source, raw_dir, refresh=refresh))
        except SourceUnavailableError as error:
            if required:
                raise
            failures.append(error)
    return acquired, failures
