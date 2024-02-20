# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
from qcrboxtools.cif.entries import (
    cif_to_unified_keywords, entry_to_unified_keyword, block_to_unified_keywords,
    block_to_requested_keywords, cif_to_requested_keywords
)
from qcrboxtools.cif.entries import cif_entries_present

import pytest
from iotbx.cif import model


def test_to_unified_name():
    custom_categories = ['iucr', 'olex2', 'shelx']

    # Test with a name that should be converted using custom categories
    assert entry_to_unified_keyword('_iucr_entry_example', custom_categories) == '_iucr.entry_example'

    # Test with a name that should fall back to the alias mechanism
    assert entry_to_unified_keyword('_journal_date_accepted', custom_categories) == '_journal_date.accepted'

    # Test with a name that does not match custom categories or aliases
    assert entry_to_unified_keyword('_unmatched_entry', custom_categories) == '_unmatched_entry'

@pytest.fixture
def custom_categories():
    return ['mock']

@pytest.fixture
def mock_old_block():
    block = model.block()
    # Add mock data items
    block.add_data_item('_mock_entry', 'value1')
    block.add_data_item('_journal_date_accepted', '2020-01-01')  # This should get converted based on the alias

    # Add a mock loop
    loop_data = {
        '_mock_loop_entry1': ['val1', 'val2'],
        '_mock_loop_entry2': ['val3', 'val4'],
    }
    loop = model.loop(data=loop_data)
    block.add_loop(loop)

    return block

def test_block_to_unified_keywords(mock_old_block, custom_categories):
    converted_block = block_to_unified_keywords(mock_old_block, custom_categories)

    # Verify that the block is a CIF block object
    assert isinstance(converted_block, model.block), "The returned object is not a CIF block."

    # Check if entries have been correctly converted
    expected_names = ['_mock.entry', '_journal_date.accepted', '_mock.loop_entry1', '_mock.loop_entry2']
    for name in expected_names:
        assert name in converted_block, f"Expected entry '{name}' not found in the converted block."

@pytest.fixture
def mock_cif(mock_old_block):
    """
    Creates a mock CIF file containing two blocks for testing.
    Each block will be a copy of the mock_old_block to simulate a real CIF structure.
    """
    cif = model.cif()
    cif['block_1'] = mock_old_block
    cif['block_2'] = mock_old_block.copy()  # Assuming a copy method or similar functionality
    return cif

def test_cif_to_unified_keywords(mock_cif, custom_categories):
    """
    Test to ensure that `cif_to_unified_keywords` correctly converts all blocks within
    a CIF file using the specified custom categories.
    """
    unified_cif = cif_to_unified_keywords(mock_cif, custom_categories)

    # Verify that the CIF object contains the same number of blocks as the mock CIF
    assert len(unified_cif) == len(mock_cif), "The number of blocks in the unified CIF does not match the original."

    # Iterate through each block in the unified CIF to ensure conversions were applied
    for block_name, unified_block in unified_cif.items():
        assert isinstance(unified_block, model.block), f"The block '{block_name}' is not a CIF block."

        # Example verification that entries were converted (specific checks depend on mock data and custom categories)
        expected_names = ['_mock.entry', '_journal_date.accepted', '_mock.loop_entry1', '_mock.loop_entry2']
        for name in expected_names:
            assert name in unified_block, f"Expected entry '{name}' not found in block '{block_name}'."


@pytest.fixture
def unified_block():
    """
    Creates a unified CIF block with predefined entries and loops for testing.
    This block simulates a structure that has already been processed to unify names.
    """
    block = model.block()
    # Add unified data items
    block.add_data_item('_mock.entry', 'value1')
    block.add_data_item('_journal_date.accepted', '2020-01-01')

    # Add a unified loop
    loop_data = {
        '_mock.loop_entry1': ['val1', 'val2'],
        '_mock.loop_entry2': ['val3', 'val4'],
    }
    loop = model.loop(data=loop_data)
    block.add_loop(loop)

    return block

#def test_block_to_requested_keywords(unified_block, custom_categories):
#    requested_entries = ['_mock_entry', '_journal_date_accepted', '_mock_loop_entry1', '_mock_loop_entry1']
#
#    converted_block = block_to_requested_keywords(unified_block, requested_entries, custom_categories)
#
#    # Ensure all requested entries are present in the converted block
#    for entry_name in requested_entries:
#        assert entry_name in converted_block, f"Requested entry '{entry_name}' was not found in the converted block."


def test_block_to_requested_keywords(unified_block, custom_categories):
    requested_entries = ['_mock_entry', '_journal_date_accepted', '_mock_loop_entry1', '_nonexistent_entry']
    optional_entries = ['_nonexistent_entry', 'mock_entry']

    # Attempt conversion with a non-existent entry marked as optional
    converted_block = block_to_requested_keywords(unified_block, requested_entries, optional_entries, custom_categories)

    # Ensure all non-optional requested entries are present in the converted block
    for entry_name in requested_entries:
        if entry_name != '_nonexistent_entry':
            assert entry_name in converted_block, f"Requested entry '{entry_name}' was not found in the converted block."

    # Ensure the optional, non-existent entry does not cause an error and is rightly not present
    assert '_nonexistent_entry' not in converted_block, "Optional, non-existent entry was generated from nothing."

@pytest.fixture
def unified_cif(unified_block):
    """
    Creates a mock CIF file containing a unified block for testing.
    This simulates a CIF file that has already undergone the unification process.
    """
    cif = model.cif()
    cif['test_block'] = unified_block
    return cif

def test_cif_to_requested_keywords(unified_cif, custom_categories):
    requested_entries = ['_mock_entry', '_journal_date_accepted', '_mock_loop_entry1', '_nonexistent_entry']
    optional_entries = ['_nonexistent_entry', 'mock_entry']

    # Convert using cif_to_requested_keywords
    new_cif = cif_to_requested_keywords(unified_cif, requested_entries, optional_entries, custom_categories)

    # Ensure that each block in the new CIF contains only the requested entries
    for _, block in new_cif.items():
        for entry_name in requested_entries:
                if entry_name != '_nonexistent_entry':
                    assert entry_name in block, f"Requested entry '{entry_name}' was not found in the converted block."

        assert '_nonexistent_entry' not in block, "Optional, non-existent entry was generated from nothing."


@pytest.fixture
def mock_block():
    """
    Creates a mock CIF block with predefined entries for testing.
    """
    block = model.block()
    block.add_data_item('_existing_entry', 'value1')
    block.add_data_item('_another_entry', 'value2')
    return block

def test_cif_entries_present(mock_block):
    custom_categories = ['existing', 'another']
    present_entries = ['_existing.entry', '_another_entry']
    absent_entries = ['_missing_entry']

    # Test with entries that are present
    assert cif_entries_present(mock_block, custom_categories, present_entries) == True, \
        "Function should return True when all entries are present."

    # Test with at least one absent entry
    assert cif_entries_present(mock_block, custom_categories, present_entries + absent_entries) == False, \
        "Function should return False when any specified entry is absent."

