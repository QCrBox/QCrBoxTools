# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import pytest
from typing import Dict, List

from qcrboxtools.util.cif import (
    merge_cif_loops, NonExistingMergeKey, NonMatchingMergeKeys, merge_cif_blocks
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

    # test unique block 1 copied
    assert list(merged_block['_atom_site.label']) == ['Si1', 'C1', 'C2']

    # atom_site_aniso successfully merged from both blocks (merged by .label)
    assert '?' not in list(merged_block['_atom_site_aniso.u_11'])
    assert '?' not in list(merged_block['_atom_site_aniso.u_23'])

    # test filling of unknown values (merged by refln.index)
    assert list(merged_block['_diffrn_refln.test_column']).count('?') == 3

    # test additional value from block1 (merged by .id)
    assert merged_block['_space_group_symop.test_entry'][0] == 'copy this'

    # test block2 overwrites block1
    assert float(merged_block['_cell.length_c']) == 12.0

    # test block2 alone copied
    assert float(merged_block['_cell.volume']) == 1200.0

    # test block1 alone copied
    assert merged_block['_space_group.name_h-m_alt'] == 'P 1'
