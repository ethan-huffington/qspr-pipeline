"""Baseline test.

Importing by package name (rather than by file path) is what the src layout
buys us: this resolves the same way from tests, notebooks, and scripts, with
no sys.path manipulation and no dependence on the current directory.
"""

import dupont_qspr


def test_package_imports() -> None:
    assert dupont_qspr.__name__ == "dupont_qspr"
