# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import pytest
from typing import Dict, List

from qcrboxtools.util.cif import (
    merge_cif_loops, NonExistingMergeKey, NonMatchingMergeKeys, merge_cif_blocks, merge_cif_files
)
from iotbx import cif

@pytest.fixture
def loop1():
    return cif.model.loop(data={
        '_atom_site.label': ['C1', 'C2'],
        '_atom_site.type_symbol': ['C', 'C'],
        '_atom_site.unique_to_loop1': ['U1', 'U2'],  # Unique column
    })

@pytest.fixture
def loop2():
    return cif.model.loop(data={
        '_atom_site.label': ['C1', 'N1'],
        '_atom_site.type_symbol': ['C', 'N'],
        '_atom_site.unique_to_loop2': ['U3', 'U4'],  # Unique column
    })

def test_merge_success_with_unique_columns(loop1, loop2):
    """
    Test that merging loops with unique and shared columns populates the merged
    loop correctly, placing new values in corresponding positions and filling
    missing entries with '?'.
    """
    merged_loop = merge_cif_loops(loop1, loop2, merge_on='_atom_site.label')

    # Check that shared labels end up in the same position
    assert list(merged_loop['_atom_site.label']) == ['C1', 'C2', 'N1']

    # Check for combined entries from shared columns
    assert list(merged_loop['_atom_site.type_symbol']) == ['C', 'C', 'N']

    # Check unique column from loop1 is present in merged_loop and missing entries filled with '?'
    assert list(merged_loop['_atom_site.unique_to_loop1']) == ['U1', 'U2', '?']

    # Check unique column from loop2 is present in merged_loop and original missing value is preserved
    assert list(merged_loop['_atom_site.unique_to_loop2']) == ['U3', '?', 'U4']

def test_merge_no_matching_keys(loop1, loop2):
    # Test merge with non-matching keys
    with pytest.raises(NonExistingMergeKey):
        merge_cif_loops(loop1, loop2, merge_on='_non_existing.label')

def test_merge_non_identical_keys(loop1, loop2):
    # Test merge with keys that do not have a one-to-one correspondence
    with pytest.raises(NonMatchingMergeKeys):
        merge_cif_loops(loop1, loop2, merge_on=['_atom_site.label', '_atom_site.unique_to_loop1'])


def test_merge_block_conflicts():
    cif_obj = cif.reader('./tests/util/cif_files/merge_me.cif').model()

    merged_block = merge_cif_blocks(*cif_obj.values())

    assert list(merged_block['_atom_site.label']) == ['Si1', 'C1', 'C2'], "Unique block 1 atom_site.label not correctly copied"
    assert '?' not in list(merged_block['_atom_site_aniso.u_11']), "atom_site_aniso not correctly merged from both blocks"
    assert '?' not in list(merged_block['_atom_site_aniso.u_23']), "atom_site_aniso not correctly merged from both blocks"
    assert list(merged_block['_diffrn_refln.test_column']).count('?') == 3, "Unknown values in diffrn_refln.test_column not filled as expected"
    assert merged_block['_space_group_symop.test_entry'][0] == 'copy this', "Additional value from block1 (merged by .id) not present"
    assert float(merged_block['_cell.length_c']) == 12.0, "Block2's cell.length_c does not overwrite block1 as expected"
    assert float(merged_block['_cell.volume']) == 1200.0, "Block2's unique value cell.volume not copied as expected"
    assert merged_block['_space_group.name_h-m_alt'] == 'P 1', "Block1's unique value space_group.name_h-m_alt not correctly copied"


def test_merge_cif_files(tmp_path):
    # Define the path to the original CIF file for input
    cif_path = './tests/util/cif_files/merge_me.cif'
    # Define a temporary output path using tmp_path
    output_path = tmp_path / "output_merged.cif"

    # Execute the merging function with block indices
    merge_cif_files(
        cif_path=cif_path,
        block_name='0',  # Assuming the first block is selected by index
        cif_path2=cif_path,
        block_name2='1',  # Assuming the second block is selected by index
        output_path=output_path,
        output_block_name='merged_block'
    )

    # Read the output CIF to verify results
    merged_cif = cif.reader(str(output_path)).model()
    merged_block = merged_cif['merged_block']

    # Perform assertions to verify correct merging
    assert list(merged_block['_atom_site.label']) == ['Si1', 'C1', 'C2'], "Unique block 1 atom_site.label not correctly copied"
    assert '?' not in list(merged_block['_atom_site_aniso.u_11']), "atom_site_aniso not correctly merged from both blocks"
    assert '?' not in list(merged_block['_atom_site_aniso.u_23']), "atom_site_aniso not correctly merged from both blocks"
    assert list(merged_block['_diffrn_refln.test_column']).count('?') == 3, "Unknown values in diffrn_refln.test_column not filled as expected"
    assert merged_block['_space_group_symop.test_entry'][0] == 'copy this', "Additional value from block1 (merged by .id) not present"
    assert float(merged_block['_cell.length_c']) == 12.0, "Block2's cell.length_c does not overwrite block1 as expected"
    assert float(merged_block['_cell.volume']) == 1200.0, "Block2's unique value cell.volume not copied as expected"
    assert merged_block['_space_group.name_h-m_alt'] == 'P 1', "Block1's unique value space_group.name_h-m_alt not correctly copied"
