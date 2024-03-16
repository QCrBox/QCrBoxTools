# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""
Test module for `qcrboxtools.util.cif`.

This module contains unit tests for the various utility functions related to
crystallographic information file (CIF) manipulation found in `qcrboxtools.util.cif`.
It tests functionality like extracting estimated standard uncertainties (sus) from
formatted strings and manipulating CIF data structure.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from qcrboxtools.cif.merge import (
    InConsistentCentringError,
    NoCentringFoundError,
    check_centring_equal,
    read_cif_safe,
    replace_structure_from_cif,
    split_su_array,
)

test_file_path = Path("./tests/cif/cif_files")


@pytest.fixture(scope="module", name="cif_with_replacement")
def fixture_cif_with_replacement(
    tmp_path_factory: pytest.TempPathFactory,
) -> Tuple[dict, dict, dict]:
    """
    Pytest fixture to replace the structure block of one CIF with another.

    This fixture replaces the structure block from the '80K_P_out.cif' CIF file
    with the structure block from the '105K_P_out.cif' CIF file.
    """
    from_cif_path = test_file_path / "80K_P_out.cif"
    to_cif_path = test_file_path / "105K_P_out.cif"
    from_cif_dataset = "80K_P"
    to_cif_dataset = "105K_P"
    combined_cif_path = tmp_path_factory.mktemp("output") / "output.cif"

    replace_structure_from_cif(
        to_cif_path, to_cif_dataset, from_cif_path, from_cif_dataset, combined_cif_path
    )

    from_cif = read_cif_safe(from_cif_path)
    to_cif = read_cif_safe(to_cif_path)
    combined_cif = read_cif_safe(combined_cif_path)
    return from_cif["80K_P"], to_cif["105K_P"], combined_cif["105K_P"]


def test_cif_atom_site_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the atom site block from the original CIF is copied correctly
    to the combined CIF.
    """
    from_cif, to_cif, combined_cif = cif_with_replacement
    to_keys = [key for key in to_cif.keys() if key.startswith("_atom_site")]
    from_keys = [key for key in from_cif.keys() if key.startswith("_atom_site")]

    ommitted_keys = [val for val in to_keys if val not in from_keys]
    for key in ommitted_keys:
        assert key not in combined_cif.keys()

    for key in from_keys:
        # combined vals should no longer have sus -> are only valid at convergence
        assert not any("(" in val for val in combined_cif[key])
        from_vals = from_cif[key]
        if any("(" in val for val in from_vals):
            from_vals, _ = split_su_array(from_vals)
            combined_vals = np.array([float(val) for val in combined_cif[key]])
            assert np.max(np.abs(from_vals - combined_vals)) < 1e-6
        else:
            for single_from, single_combined in zip(from_vals, combined_cif[key]):
                assert single_from == single_combined


@pytest.mark.not_implemented
def test_cif_atom_type_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the atom type block from the original CIF is copied correctly
    to the combined CIF.
    """
    raise NotImplementedError("This is a future feature")


def test_refinement_details_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that refinement details from the original CIF are copied correctly
    to the combined CIF.
    """
    from_cif, _, combined_cif = cif_with_replacement
    test_key = "_shelx_res_file"
    if test_key in combined_cif.keys() or test_key in from_cif.keys():
        assert combined_cif[test_key] == from_cif[test_key]

    test_key = "_iucr_refine_instructions_details"
    if test_key in combined_cif.keys() or test_key in from_cif.keys():
        assert combined_cif[test_key] == from_cif[test_key]


def test_space_group_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the space group block from the original CIF is copied correctly
    to the combined CIF.
    """
    # TODO add test case where this fails when not implemented properly
    from_cif, to_cif, combined_cif = cif_with_replacement

    to_keys = [key for key in to_cif.keys() if key.startswith("_space_group")]
    from_keys = [key for key in from_cif.keys() if key.startswith("_space_group")]

    ommitted_keys = [val for val in to_keys if val not in from_keys]
    for key in ommitted_keys:
        assert key not in combined_cif.keys()

    symop_keys = [key for key in from_keys if key.startswith("_space_group_symop")]

    for key in symop_keys:
        assert all(comb == fro for comb, fro in zip(combined_cif[key], from_cif[key]))

    remaining_keys = [key for key in from_keys if key not in symop_keys]

    assert all(from_cif[key] == combined_cif[key] for key in remaining_keys)


def test_cif_geom_deleted(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the geometry block (starting with `_geom`) is deleted in the
    combined CIF.
    """
    _, _, combined_cif = cif_with_replacement
    assert not any(key.startswith("_geom") for key in combined_cif.keys())


def test_cif_refine_deleted(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the refinement block (starting with `_refine` but not exceptions)
    is deleted in the combined CIF.
    """
    _, _, combined_cif = cif_with_replacement
    exceptions = ("_refine_ls_weighting", "_refine_ls_extinction")
    assert not any(
        key.startswith("_refine") and not key.startswith(exceptions) for key in combined_cif.keys()
    )


def test_cif_refln_calc_deleted(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the reflection calculation block (starting with `_refln_` and
    containing 'calc') is deleted in the combined CIF.
    """
    _, _, combined_cif = cif_with_replacement
    assert not any(key.startswith("_refln.") and "calc" in key for key in combined_cif.keys())


def test_cif_cell_kept(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the cell block (starting with `_cell`) from the CIF to be
    merged is kept intact in the combined CIF.
    """
    from_cif, to_cif, combined_cif = cif_with_replacement
    to_keys = [key for key in to_cif.keys() if key.startswith("_cell")]
    from_keys = [key for key in from_cif.keys() if key.startswith("_cell")]

    ommitted_keys = [val for val in from_keys if val not in to_keys]
    for key in ommitted_keys:
        assert key not in combined_cif.keys()

    for key in to_keys:
        assert to_cif[key] == combined_cif[key]


def test_reflns_kept(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the reflections block (starting with `_reflns`) from the CIF
    to be merged is kept intact in the combined CIF.
    """
    from_cif, to_cif, combined_cif = cif_with_replacement
    to_keys = [key for key in to_cif.keys() if key.startswith("_reflns")]
    from_keys = [key for key in from_cif.keys() if key.startswith("_reflns")]

    ommitted_keys = [val for val in from_keys if val not in to_keys]
    for key in ommitted_keys:
        assert key not in combined_cif.keys()

    for key in to_keys:
        assert to_cif[key] == combined_cif[key]


@pytest.mark.not_implemented
def test_laue_class_change_exception():
    raise NotImplementedError()


@pytest.mark.not_implemented
def test_keep_cif_entries():
    raise NotImplementedError()


@pytest.mark.not_implemented
def test_delete_cif_entries():
    raise NotImplementedError()


@pytest.mark.not_implemented
def test_check_transformation_matrix():
    raise NotImplementedError()


@pytest.mark.parametrize(
    "block1,block2,expected",
    [
        [{"_space_group.name_h-m_alt": "P n m a"}, {"_space_group.name_h-m_alt": "P n m a"}, True],
        [{"_space_group.name_h-m_alt": "P n m a"}, {"_space_group.name_h-m_alt": "C c c m"}, False],
        [{"_space_group.name_hall": "-P 2yac"}, {"_space_group.name_hall": "-P 2yac"}, True],
        [{"_space_group.name_hall": "-P 2yac"}, {"_space_group.name_hall": "P -2y"}, True],
        [{"_space_group.name_hall": "-P 2yac"}, {"_space_group.name_hall": "C -2y"}, False],
        [{"_space_group.name_hall": "-P 2yac"}, {"_space_group.name_hall": "-C 2y"}, False],
        [
            {"_space_group.name_h-m_alt": "P n m a", "_space_group.name_hall": "-P 2yac"},
            {"_space_group.name_h-m_alt": "P n m a", "_space_group.name_hall": "P 2yac"},
            True,
        ],
    ],
)
def test_lattice_centring_equal(block1, block2, expected):
    assert check_centring_equal(block1, block2) == expected


@pytest.mark.parametrize(
    "block1,block2,expected_error",
    [
        [{"_space_group.name_h-m_alt": "P n m a"}, {}, NoCentringFoundError],
        [{}, {"_space_group.name_h-m_alt": "P n m a"}, NoCentringFoundError],
        [
            {"_space_group.name_h-m_alt": "P n m a", "_space_group.name_hall": "-P 2yac"},
            {"_space_group.name_h-m_alt": "P n m a", "_space_group.name_hall": "C 2yac"},
            InConsistentCentringError,
        ],
    ],
)
def test_lattice_centring_equal_errors(block1, block2, expected_error):
    with pytest.raises(expected_error):
        check_centring_equal(block1, block2)


@pytest.mark.not_implemented
def test_crystal_system_equal(block1, block2, expected):
    raise NotImplementedError()
