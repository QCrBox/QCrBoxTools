# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import re
from pathlib import Path
from textwrap import dedent

import pytest

from qcrboxtools.cif.cif2cif import (
    cif_file_unified_to_keywords_merge_su, cif_file_unified_yml_instr,
    cif_entries_from_yml, cif_file_unify_split
)

@pytest.fixture
def cif_path(tmp_path):
    """Create a temporary CIF file for testing."""
    cif_content = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        loop_
        _test_loop_id
        _test_loop_value_with_su
        _test_loop_value_without_su
        1 2.34(5) 7.89
        2 3.45(6) 8.90
        """)
    cif_file = tmp_path / "test_data.cif"
    cif_file.write_text(cif_content)
    return cif_file


def test_cif_file_unify_split(cif_path, tmp_path):
    """
    Test the cif_file_unify_split function to ensure it correctly processes
    and writes a CIF file according to the specified parameters.
    """
    # Define the output file path
    output_cif_path = tmp_path / "output_test_data.cif"

    # Call the function under test with split SUs and without converting keywords
    cif_file_unify_split(
        input_cif_path=cif_path,
        output_cif_path=output_cif_path,
        convert_keywords=False,
        split_sus=True
    )

    # Read back the output file and verify its content
    output_cif_content = output_cif_path.read_text()

    # Expected content checks
    expected_lines = [
        "data_test",
        r"_test_value_with_su\s+1\.23",
        r"_test_value_with_su_su\s+0\.04",
        r"_test_value_without_su\s+5\.67",
        "loop_",
        "_test_loop_id",
        "_test_loop_value_with_su",
        "_test_loop_value_with_su_su",
        "_test_loop_value_without_su",
        r"\s*1\s+2\.34\s+0\.05\s+7\.89",
        r"\s*2\s+3\.45\s+0\.06\s+8\.90",
    ]

    for line in expected_lines:
        assert re.search(line, output_cif_content) is not None, f"Expected line not found: {line}"


@pytest.fixture
def temp_cif_file(tmp_path) -> Path:
    """
    Creates a temporary CIF file with pre-defined content for testing.

    Returns
    -------
    Path
        The path to the temporary CIF file.
    """
    cif_content = dedent("""
    data_test
    _custom.test 'something'
    _cell.length_a 10.0
    _cell.length_a_su 0.03
    _cell.length_b 20.0
    _cell.length_b_su 0.02
    loop_
      _atom_site.label
      _atom_site.fract_x
      _atom_site.fract_x_su
      _atom_site.fract_y
      _atom_site.fract_y_su
      _atom_site.fract_z
      _atom_site.fract_z_su
        C1 0.234 0.012 0.567 0.045 0.890 0.078
        O1 -0.345 0.023 0.678 0.0 -0.901 0.089
    """)
    cif_file = tmp_path / "test.cif"
    cif_file.write_text(cif_content, encoding='UTF-8')
    return cif_file

def test_cif_file_unified_to_keywords_merge_su(temp_cif_file, tmp_path):
    """
    Test the cif_file_unified_to_keywords_merge_su function to ensure it processes the CIF file
    as expected, merging SUs and filtering entries according to specified criteria.
    """
    output_cif_path = tmp_path / "output.cif"

    # Define compulsory and optional entries for the test
    compulsory_entries = ['_cell_length_a']
    optional_entries = ['_cell_length_b', '_cell_length_b_su', '_atom_site_fract_x', '_atom_site_fract_y']
    custom_categories = ['custom']  # Assuming custom categories functionality is part of your implementation

    # Call the function with merge_sus enabled
    cif_file_unified_to_keywords_merge_su(
        input_cif_path=temp_cif_file,
        output_cif_path=output_cif_path,
        compulsory_entries=compulsory_entries,
        optional_entries=optional_entries,
        custom_categories=custom_categories,
        merge_sus=True
    )

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding='UTF-8')
    search_patterns = (
        r'_cell_length_a\s+10.00\(3\)',
        r'_cell_length_b\s+20.0',
        '_cell_length_b_su', # cell_length_b_su is requested as entry and should not be merged
        '_atom_site_fract_x',
        '_atom_site_fract_y'
    )
    for pattern in search_patterns:
        assert re.search(pattern, output_content) is not None
    assert '_atom_site_fract_z' not in output_content, "Included _atom_site.fract_z entry unexpectedly"

def test_direct_cif_entries_extraction():
    """Test extraction of directly defined keywords."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "required_cif_entries": ["_cell_length_a", "_cell_length_b"],
                "optional_cif_entries": ["_atom_site.label"]
            }
        ]
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif")
    assert sorted(compulsory) == sorted(["_cell_length_a", "_cell_length_b"]), "Failed to extract compulsory keywords"
    assert sorted(optional) == sorted(["_atom_site.label"]), "Failed to extract optional keywords"

def test_cif_entries_extraction_via_sets():
    """Test extraction of keywords defined through keyword sets."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "required_cif_entry_sets": ["cell_dimensions"],
                "optional_cif_entry_sets": ["atom_sites"]
            }
        ],
        "cif_entry_sets": [
            {
                "name": "cell_dimensions",
                "required": ["_cell_length_a", "_cell_length_b"],
                "optional": []  # Example with an empty list
            },
            {
                "name": "atom_sites",
                "required": [],
                "optional": ["_atom_site.label", "_atom_site.occupancy"]
            }
        ]
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif")
    assert set(compulsory) == {"_cell_length_a", "_cell_length_b"}, "Failed to extract compulsory keywords from sets"
    assert set(optional) == {"_atom_site.label", "_atom_site.occupancy"}, "Failed to extract optional keywords from sets"

@pytest.mark.parametrize("missing_key", ["process_cif", "nonexistent_set"])
def test_error_for_missing_command_or_set(missing_key):
    """Test that the correct errors are raised for missing commands or keyword sets."""
    yml_dict = {
        'commands': [
            {"name": "nonexistent_set", "required_cif_entry_sets": ["missing"]}
        ]
    }
    with pytest.raises(KeyError):
        cif_entries_from_yml(yml_dict, missing_key)

def test_incorrect_entry_in_cif_entry_set():
    """Test detection of incorrect entries within keyword sets."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "required_cif_entry_sets": ["incorrect_set"]
            }
        ],
        "cif_entry_sets": [
            {
                "name": "incorrect_set",
                "wrong_entry": ["_cell_length_a"]  # Intentionally incorrect to trigger the error
            }
        ]

    }
    with pytest.raises(NameError):
        cif_entries_from_yml(yml_dict, "process_cif")

def test_unique_compulsory_cif_entries_from_multiple_sets():
    """Test that compulsory keywords from multiple sets are appended uniquely."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "required_cif_entry_sets": ["set1", "set2"],
            }
        ],
        "cif_entry_sets": [
            {
                "name": "set1",
                "required": ["_cell_length_a", "_cell_angle_alpha"],
                "optional": []
            },
            {
                "name": "set2",
                "required": ["_cell_length_a", "_cell_volume"],
                "optional": []
            }
        ]
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif")
    assert sorted(compulsory) == sorted(list(set(["_cell_length_a", "_cell_angle_alpha", "_cell_volume"]))), \
        "Compulsory keywords should be unique and include all items from both sets"
    assert optional == [], "No optional keywords should be present"

def test_unique_optional_cif_entries_from_multiple_sets():
    """
    Test that optional keywords from multiple sets are appended uniquely and exclude compulsory ones.
    Also test that optional keyword set entries all end up in optional
    """
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "optional_cif_entry_sets": ["set1", "set2"],
                "required_cif_entries": ["_cell_length_a"]  # Ensure exclusion of compulsory from optional
            }
        ],
        "cif_entry_sets": [
            {
                "name": "set1",
                "required": ["_cell_length_a", "_cell_angle_alpha"], # _cell_length_a should be excluded
                "optional": []
            },
            {
                "name": "set2",
                "required": ["_cell_volume"],
                "optional": ["_cell_length_b"]
            }
        ]
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif")
    assert compulsory == ["_cell_length_a"], "Only specified compulsory keywords should be present"
    # Ensure _cell_length_a is not duplicated in optional, despite being in both sets and compulsory
    assert sorted(optional) == sorted(list(set(["_cell_length_b", "_cell_angle_alpha", "_cell_volume"]))), \
        "Optional keywords should be unique and exclude compulsory keywords"

def test_cif_file_unified_yml_instr(temp_cif_file, tmp_path):
    output_cif_path = tmp_path / "output.cif"

    yml_path = tmp_path / "config.yml"

    # Mock YAML content
    yml_content = dedent("""
    cif_entry_sets :
      - name: test1
        required : [
          _cell_length_a
        ]
        optional : [
          _atom_site_fract_y
        ]
      - name: test2
        required : [
          _invalid_keyword, _atom_site_fract_x
        ]
    commands :
      - name: process_cif
        merge_cif_su: Yes
        custom_cif_categories: [custom]
        required_cif_entry_sets: [test1]
        optional_cif_entry_sets: [test2]
        required_cif_entries: [_cell_length_b]
        optional_cif_entries: [_atom_site_label]
    """)
    yml_path.write_text(yml_content)

    cif_file_unified_yml_instr(
        input_cif_path=temp_cif_file,
        output_cif_path=output_cif_path,
        yml_path=yml_path,
        command='process_cif'
    )

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding='UTF-8')
    search_patterns = (
        r'_cell_length_a\s+10.00\(3\)',
        r'_cell_length_b\s+20.00\(2\)',
        '_atom_site_fract_x',
        '_atom_site_fract_y'
    )
    for pattern in search_patterns:
        assert re.search(pattern, output_content) is not None
    assert '_atom_site_fract_z' not in output_content, "Included _atom_site.fract_z entry unexpectedly"