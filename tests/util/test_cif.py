"""
Test module for `qcrboxtools.util.cif`.

This module contains unit tests for the various utility functions related to
crystallographic information file (CIF) manipulation found in `qcrboxtools.util.cif`.
It tests functionality like extracting estimated standard deviations (ESDs) from
formatted strings and manipulating CIF data structure.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from qcrboxtools.util.cif import (
    read_cif_safe,
    replace_structure_from_cif,
    split_esd_single,
    split_esds
)

@pytest.mark.parametrize('string, test_value, test_esd', [
    ('100(1)', 100.0, 1),
    ('-100(20)', -100.0, 20.0),
    ('0.021(2)', 0.021, 0.002),
    ('-0.0213(3)', -0.0213, 0.0003),
    ('2.1(1.3)', 2.1, 1.3),
    ('0.648461', 0.648461, np.nan)
])
def test_split_esd_single(string: str, test_value: float, test_esd: float):
    """Test the `split_esd_single` function with various formatted strings."""
    val, esd = split_esd_single(string)
    assert val == pytest.approx(test_value)
    if np.isnan(esd) or np.isnan(test_esd):
        assert np.isnan(esd) and np.isnan(test_esd)
    else:
        assert esd == pytest.approx(test_esd)

def test_split_esds():
    """Test the `split_esds` function."""
    strings = [
        '0.03527(13)',
        '0.02546(10)',
        '0.02949(11)',
        '0.00307(9)',
        '0.01031(9)',
        '-0.00352(8)'
    ]

    solutions = [
        (0.03527, 0.00013),
        (0.02546, 0.00010),
        (0.02949, 0.00011),
        (0.00307, 0.00009),
        (0.01031, 0.00009),
        (-0.00352, 0.00008)
    ]

    values, esds = split_esds(strings)

    for value, esd, (check_value, check_esd) in zip(values, esds, solutions):
        assert value == pytest.approx(check_value)
        assert esd == pytest.approx(check_esd)


@pytest.fixture(scope="module", name='cif_with_replacement')
def fixture_cif_with_replacement(
    tmp_path_factory: pytest.TempPathFactory
) -> Tuple[dict, dict, dict]:
    """
    Pytest fixture to replace the structure block of one CIF with another.

    This fixture replaces the structure block from the '80K_P_out.cif' CIF file
    with the structure block from the '105K_P_out.cif' CIF file.
    """
    from_cif_path = Path('./tests/util/cif_files/80K_P_out.cif')
    to_cif_path = Path('./tests/util/cif_files/105K_P_out.cif')
    from_cif_dataset = '80K_P'
    to_cif_dataset = '105K_P'
    combined_cif_path = tmp_path_factory.mktemp('output') / 'output.cif'

    replace_structure_from_cif(
        to_cif_path,
        to_cif_dataset,
        from_cif_path,
        from_cif_dataset,
        combined_cif_path
    )

    from_cif = read_cif_safe(from_cif_path)
    to_cif = read_cif_safe(to_cif_path)
    combined_cif = read_cif_safe(combined_cif_path)
    return from_cif['80K_P'], to_cif['105K_P'], combined_cif['105K_P']

def test_cif_atom_site_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the atom site block from the original CIF is copied correctly
    to the combined CIF.
    """
    from_cif, to_cif, combined_cif = cif_with_replacement
    to_keys = [key for key in to_cif.keys() if key.startswith('_atom_site')]
    from_keys = [key for key in from_cif.keys() if key.startswith('_atom_site')]

    ommitted_keys = [val for val in to_keys if val not in from_keys]
    for key in ommitted_keys:
        assert key not in combined_cif.keys()

    for key in from_keys:
        #combined vals should no longer have esds -> are only valid at convergence
        assert not any('(' in val for val in combined_cif[key])
        from_vals = from_cif[key]
        if any('(' in val for val in from_vals):
            from_vals, _ = split_esds(from_vals)
            combined_vals = np.array([float(val) for val in combined_cif[key]])
            assert np.max(np.abs(from_vals - combined_vals)) < 1e-6
        else:
            for single_from, single_combined in zip(from_vals, combined_cif[key]):
                assert single_from == single_combined

def test_cif_atom_type_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the atom type block from the original CIF is copied correctly
    to the combined CIF.
    """
    raise NotImplementedError('This is a future feature')

def test_refinement_details_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that refinement details from the original CIF are copied correctly
    to the combined CIF.
    """
    from_cif, _, combined_cif = cif_with_replacement
    test_key = '_shelx_res_file'
    if test_key in combined_cif.keys() or test_key in from_cif.keys():
        assert combined_cif[test_key] == from_cif[test_key]

    test_key = '_iucr_refine_instructions_details'
    if test_key in combined_cif.keys() or test_key in from_cif.keys():
        assert combined_cif[test_key] == from_cif[test_key]

def test_space_group_copied(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the space group block from the original CIF is copied correctly
    to the combined CIF.
    """
    # TODO add test case where this fails when not implemented properly
    from_cif, to_cif, combined_cif = cif_with_replacement

    to_keys = [key for key in to_cif.keys() if key.startswith('_space_group')]
    from_keys = [key for key in from_cif.keys() if key.startswith('_space_group')]

    ommitted_keys = [val for val in to_keys if val not in from_keys]
    for key in ommitted_keys:
        assert key not in combined_cif.keys()

    symop_keys = [key for key in from_keys if key.startswith('_space_group_symop')]

    for key in symop_keys:
        assert all(
            comb == fro for comb, fro in zip(combined_cif[key], from_cif[key])
        )

    remaining_keys = [key for key in from_keys if key not in symop_keys]

    assert all(from_cif[key] == combined_cif[key] for key in remaining_keys)


def test_cif_geom_deleted(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the geometry block (starting with `_geom`) is deleted in the
    combined CIF.
    """
    _, _, combined_cif = cif_with_replacement
    assert not any(key.startswith('_geom') for key in combined_cif.keys())

def test_cif_refine_deleted(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the refinement block (starting with `_refine` but not exceptions)
    is deleted in the combined CIF.
    """
    _, _, combined_cif = cif_with_replacement
    exceptions = ('_refine_ls_weighting','_refine_ls_extinction')
    assert not any(key.startswith('_refine') and not key.startswith(exceptions)
                   for key in combined_cif.keys())

def test_cif_refln_calc_deleted(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the reflection calculation block (starting with `_refln_` and
    containing 'calc') is deleted in the combined CIF.
    """
    _, _, combined_cif = cif_with_replacement
    assert not any(key.startswith('_refln_') and 'calc' in key for key in combined_cif.keys())

def test_cif_cell_kept(cif_with_replacement: Tuple[dict, dict, dict]):
    """
    Test that the cell block (starting with `_cell`) from the CIF to be
    merged is kept intact in the combined CIF.
    """
    from_cif, to_cif, combined_cif = cif_with_replacement
    to_keys = [key for key in to_cif.keys() if key.startswith('_cell')]
    from_keys = [key for key in from_cif.keys() if key.startswith('_cell')]

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
    to_keys = [key for key in to_cif.keys() if key.startswith('_reflns')]
    from_keys = [key for key in from_cif.keys() if key.startswith('_reflns')]

    ommitted_keys = [val for val in from_keys if val not in to_keys]
    for key in ommitted_keys:
        assert key not in combined_cif.keys()

    for key in to_keys:
        assert to_cif[key] == combined_cif[key]
