# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
import tempfile
from pathlib import Path

import pytest
from iotbx.cif import model, reader

from qcrboxtools.cif.trim import keep_single_kw, trim_cif, trim_cif_block, trim_cif_file


@pytest.mark.parametrize(
    "name,keep_only_regexes,delete_regexes,expected",
    [
        ("_keep_this_entry", [r"_keep.*"], [r"_delete.*"], True),
        ("_delete_this_entry", [r"_keep.*"], [r"_delete.*"], False),
        ("_ambiguous_entry", [r"_amb.*"], [r"_amb.*"], False),  # delete overwrites keep
        ("_keep_not_delete", [r"_keep.*"], [r"_not.*"], True),
        ("_keep_this_entry", [], [r"_delete.*"], True),
    ],
)
def test_keep_single_kw(name, keep_only_regexes, delete_regexes, expected):
    assert keep_single_kw(name, keep_only_regexes, delete_regexes) == expected


@pytest.fixture
def mock_cif_block():
    block = model.block()
    block.add_data_item("_keep_this", "value1")
    block.add_data_item("_delete_this", "value2")
    block.add_data_item("_empty_entry", "?")
    block.add_data_item("_keep_also_this", "value3")
    return block


def test_trim_cif_block(mock_cif_block):
    keep_only_regexes = [r"_empty.*", r"_keep.*"]
    delete_regexes = [r"_delete.*"]

    trimmed_block = trim_cif_block(mock_cif_block, keep_only_regexes, delete_regexes, delete_empty_entries=True)

    assert "_keep_this" in trimmed_block
    assert "_keep_also_this" in trimmed_block
    assert "_delete_this" not in trimmed_block
    assert "_empty_entry" not in trimmed_block

    trimmed_block = trim_cif_block(mock_cif_block, keep_only_regexes, delete_regexes, delete_empty_entries=False)

    assert "_keep_this" in trimmed_block
    assert "_keep_also_this" in trimmed_block
    assert "_delete_this" not in trimmed_block
    assert "_empty_entry" in trimmed_block


def test_trim_cif_block_with_loop():
    # Create a CIF block with a loop
    block = model.block()
    block.add_loop(
        model.loop(
            data={
                "_loop_key1": ["value1", "value2", "value3"],
                "_loop_key2": ["value4", "value5", "value6"],
                "_loop_key3": ["value7", "value8", "value9"],
            }
        )
    )

    # Define your patterns and call the function
    keep_only_regexes = [
        r"_loop_key1",
    ]
    delete_regexes = [r"_loop_key2"]
    trimmed_block = trim_cif_block(block, keep_only_regexes, delete_regexes, delete_empty_entries=True)

    # Check the contents of the trimmed block
    assert "_loop_key1" in trimmed_block
    assert "_loop_key2" not in trimmed_block
    assert "_loop_key3" not in trimmed_block


def test_trim_cif(mock_cif_block):
    cif = model.cif({"mock_block": mock_cif_block, "other_block": mock_cif_block})
    trimmed_cif = trim_cif(cif, [r"_empty.*", r"_keep.*"], [r"_delete.*"], delete_empty_entries=True)

    for block_name in ("mock_block", "other_block"):
        block = trimmed_cif[block_name]
        assert "_keep_this" in block
        assert "_keep_also_this" in block
        assert "_delete_this" not in block
        assert "_empty_entry" not in block


def test_trim_cif_file(mock_cif_block):
    # Convert the mock CIF block to a string representing CIF content
    mock_cif_content = str(model.cif({"mock_block": mock_cif_block}))

    # Create a temporary CIF file
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".cif") as tmpfile:
        tmpfile.write(mock_cif_content)
        tmpfile_path = tmpfile.name

    # Define your patterns and call the function
    keep_only_regexes = [r"_keep.*"]
    delete_regexes = [r"_delete.*"]
    trim_cif_file(tmpfile_path, "mock_block", keep_only_regexes, delete_regexes, delete_empty_entries=True)

    # Convert back to cif model to check contents
    trimmed_cif = reader(tmpfile_path).model()
    trimmed_block = trimmed_cif["mock_block"]

    assert "_keep_this" in trimmed_block
    assert "_keep_also_this" in trimmed_block
    assert "_delete_this" not in trimmed_block
    assert "_empty_entry" not in trimmed_block

    # Cleanup
    Path(tmpfile_path).unlink()
