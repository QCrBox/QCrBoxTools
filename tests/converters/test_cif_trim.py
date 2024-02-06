# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
import tempfile
from pathlib import Path

import pytest
from iotbx.cif import model, reader

from qcrboxtools.converters.cif.trim import (
    keep_single_kw, trim_cif_block, trim_cif_file
)

@pytest.mark.parametrize("name,keep_only_regexes,delete_regexes,expected", [
    ("_keep_this_entry", [r"_keep.*"], [r"_delete.*"], True),
    ("_delete_this_entry", [r"_keep.*"], [r"_delete.*"], False),
    ("_ambiguous_entry", [r"_amb.*"], [r"_amb.*"], False),  # delete overwrites keep
    ("_keep_not_delete", [r"_keep.*"], [r"_not.*"], True)
])
def test_keep_single_kw(name, keep_only_regexes, delete_regexes, expected):
    assert keep_single_kw(name, keep_only_regexes, delete_regexes) == expected

@pytest.fixture
def mock_cif_block():
    block = model.block()
    block.add_data_item('_keep_this', 'value1')
    block.add_data_item('_delete_this', 'value2')
    block.add_data_item('_empty_entry', '?')
    block.add_data_item('_keep_also_this', 'value3')
    return block

def test_trim_cif_block(mock_cif_block):
    keep_only_regexes = [r"_empty.*", r"_keep.*"]
    delete_regexes = [r"_delete.*"]

    trimmed_block = trim_cif_block(
        mock_cif_block,
        keep_only_regexes,
        delete_regexes,
        delete_empty_entries=True
    )

    assert '_keep_this' in trimmed_block
    assert '_keep_also_this' in trimmed_block
    assert '_delete_this' not in trimmed_block
    assert '_empty_entry' not in trimmed_block

    trimmed_block = trim_cif_block(
        mock_cif_block,
        keep_only_regexes,
        delete_regexes,
        delete_empty_entries=False
    )

    assert '_keep_this' in trimmed_block
    assert '_keep_also_this' in trimmed_block
    assert '_delete_this' not in trimmed_block
    assert '_empty_entry' in trimmed_block

def test_trim_cif_file(mock_cif_block):
    # Convert the mock CIF block to a string representing CIF content
    mock_cif_content = str(model.cif({'mock_block': mock_cif_block}))

    # Create a temporary CIF file
    with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.cif') as tmpfile:
        tmpfile.write(mock_cif_content)
        tmpfile_path = tmpfile.name

    # Define your patterns and call the function
    keep_only_regexes = [r"_keep.*"]
    delete_regexes = [r"_delete.*"]
    trim_cif_file(tmpfile_path, 'mock_block', keep_only_regexes, delete_regexes, delete_empty_entries=True)

    # Convert back to cif model to check contents
    trimmed_cif = reader(tmpfile_path).model()
    trimmed_block = trimmed_cif['mock_block']

    assert '_keep_this' in trimmed_block
    assert '_keep_also_this' in trimmed_block
    assert '_delete_this' not in trimmed_block
    assert '_empty_entry' not in trimmed_block

    # Cleanup
    Path(tmpfile_path).unlink()