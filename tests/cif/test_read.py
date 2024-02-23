# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from iotbx import cif
import pytest
from textwrap import dedent
import re

from qcrboxtools.cif.read import (
    cifdata_str_or_index, read_cif_as_unified, cif_file_unify_split,
    cif_file_unified_to_keywords_merge_su, cif_file_unified_yml_instr,
    keywords_from_yml
)

def test_cifdata_str_or_index_by_str():
    # Setup CIF model with mock data
    cif_content = """
    data_test_block
    _tag value
    """
    model = cif.reader(input_string=cif_content).model()

    # Test retrieval by string identifier
    block, identifier = cifdata_str_or_index(model, 'test_block')
    assert identifier == 'test_block'
    assert '_tag' in block

def test_cifdata_str_or_index_by_index():
    # Setup CIF model with multiple blocks for testing index access
    cif_content = """
    data_first_block
    _tag_first value_first

    data_second_block
    _tag_second value_second
    """
    model = cif.reader(input_string=cif_content).model()

    # Test retrieval by index
    block, identifier = cifdata_str_or_index(model, 1)  # Assuming 0-based indexing
    assert identifier == 'second_block'
    assert '_tag_second' in block

def test_cifdata_str_or_index_invalid_str():
    # Setup CIF model with mock data
    cif_content = """
    data_test_block
    _tag value
    """
    model = cif.reader(input_string=cif_content).model()

    # Test retrieval with invalid string identifier
    with pytest.raises(ValueError):
        cifdata_str_or_index(model, 'nonexistent_block')

def test_cifdata_str_or_index_invalid_index():
    # Setup CIF model with a single block to test invalid index access
    cif_content = """
    data_test_block
    _tag value
    """
    model = cif.reader(input_string=cif_content).model()

    # Test retrieval with out-of-range index
    with pytest.raises(IndexError):
        cifdata_str_or_index(model, 2)  # Index out of range for this model

def test_cifdata_str_or_index_non_int_index():
    # Setup CIF model with mock data
    cif_content = """
    data_test_block
    _tag value
    """
    model = cif.reader(input_string=cif_content).model()

    # Test retrieval with a string that can't be converted to an int
    with pytest.raises(ValueError):
        cifdata_str_or_index(model, 'invalid_index')

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

def test_read_cif_as_unified(cif_path):
    """Test the read_cif_as_unified function for correctness."""
    for dataset in ['test', None]:
        # Test without any processing
        output = read_cif_as_unified(
            cif_path,
            dataset=dataset,
            split_sus=False,
            convert_keywords=False
        )
        if dataset is None:
            # also test cif conversion
            output = output['test']
        assert '_test_value_with_su' in output
        assert output['_test_value_with_su'] == '1.23(4)'

        # Test with standard uncertainties split
        output = read_cif_as_unified(
            cif_path,
            dataset=dataset,
            split_sus=True,
            convert_keywords=False
        )
        if dataset is None:
            output = output['test']
        assert '_test_value_with_su_su' in output
        assert output['_test_value_with_su'] == '1.23'
        assert output['_test_value_with_su_su'] == '0.04'

        # Test with standard uncertainties split and unified keywords
        output = read_cif_as_unified(
            cif_path,
            dataset=dataset,
            split_sus=True,
            convert_keywords=True,
            custom_categories=['test']
        )
        if dataset is None:
            output = output['test']
        assert '_test.value_with_su_su' in output
        assert output['_test.value_with_su'] == '1.23'
        assert output['_test.value_with_su_su'] == '0.04'

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
    optional_entries = ['_cell_length_b', '_atom_site_fract_x', '_atom_site_fract_y']
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
        r'_cell_length_b\s+20.00\(2\)',
        '_atom_site_fract_x',
        '_atom_site_fract_y'
    )
    for pattern in search_patterns:
        assert re.search(pattern, output_content) is not None
    assert '_atom_site_fract_z' not in output_content, "Included _atom_site.fract_z entry unexpectedly"

def test_direct_keywords_extraction():
    """Test extraction of directly defined keywords."""
    yml_dict = {
        "commands": {
            "process_cif": {
                "required_keywords": ["_cell_length_a", "_cell_length_b"],
                "optional_keywords": ["_atom_site.label"]
            }
        }
    }
    compulsory, optional = keywords_from_yml(yml_dict, "process_cif")
    assert sorted(compulsory) == sorted(["_cell_length_a", "_cell_length_b"]), "Failed to extract compulsory keywords"
    assert sorted(optional) == sorted(["_atom_site.label"]), "Failed to extract optional keywords"

def test_keywords_extraction_via_sets():
    """Test extraction of keywords defined through keyword sets."""
    yml_dict = {
        "commands": {
            "process_cif": {
                "required_keyword_sets": ["cell_dimensions"],
                "optional_keyword_sets": ["atom_sites"]
            }
        },
        "keyword_sets": {
            "cell_dimensions": {
                "required": ["_cell_length_a", "_cell_length_b"],
                "optional": []  # Example with an empty list
            },
            "atom_sites": {
                "required": [],
                "optional": ["_atom_site.label", "_atom_site.occupancy"]
            }
        }
    }
    compulsory, optional = keywords_from_yml(yml_dict, "process_cif")
    assert set(compulsory) == {"_cell_length_a", "_cell_length_b"}, "Failed to extract compulsory keywords from sets"
    assert set(optional) == {"_atom_site.label", "_atom_site.occupancy"}, "Failed to extract optional keywords from sets"

@pytest.mark.parametrize("missing_key", ["process_cif", "nonexistent_set"])
def test_error_for_missing_command_or_set(missing_key):
    """Test that the correct errors are raised for missing commands or keyword sets."""
    yml_dict = {
        'commands': {"nonexistent_set": {"required_keyword_sets": ["missing"]}}
    }
    with pytest.raises(KeyError):
        keywords_from_yml(yml_dict, missing_key)

def test_incorrect_entry_in_keyword_set():
    """Test detection of incorrect entries within keyword sets."""
    yml_dict = {
        "commands": {
            "process_cif": {
                "required_keyword_sets": ["incorrect_set"]
            }
        },
        "keyword_sets": {
            "incorrect_set": {
                "wrong_entry": ["_cell_length_a"]  # Intentionally incorrect to trigger the error
            }
        }
    }
    with pytest.raises(NameError):
        keywords_from_yml(yml_dict, "process_cif")

def test_unique_compulsory_keywords_from_multiple_sets():
    """Test that compulsory keywords from multiple sets are appended uniquely."""
    yml_dict = {
        "commands": {
            "process_cif": {
                "required_keyword_sets": ["set1", "set2"],
            }
        },
        "keyword_sets": {
            "set1": {
                "required": ["_cell_length_a", "_cell_angle_alpha"],
                "optional": []
            },
            "set2": {
                "required": ["_cell_length_a", "_cell_volume"],
                "optional": []
            }
        }
    }
    compulsory, optional = keywords_from_yml(yml_dict, "process_cif")
    assert sorted(compulsory) == sorted(list(set(["_cell_length_a", "_cell_angle_alpha", "_cell_volume"]))), \
        "Compulsory keywords should be unique and include all items from both sets"
    assert optional == [], "No optional keywords should be present"

def test_unique_optional_keywords_from_multiple_sets():
    """
    Test that optional keywords from multiple sets are appended uniquely and exclude compulsory ones.
    Also test that optional keyword set entries all end up in optional
    """
    yml_dict = {
        "commands": {
            "process_cif": {
                "optional_keyword_sets": ["set1", "set2"],
                "required_keywords": ["_cell_length_a"]  # Ensure exclusion of compulsory from optional
            }
        },
        "keyword_sets": {
            "set1": {
                "required": ["_cell_length_a", "_cell_angle_alpha"], # _cell_length_a should be excluded
                "optional": []
            },
            "set2": {
                "required": ["_cell_volume"],
                "optional": ["_cell_length_b"]
            }
        }
    }
    compulsory, optional = keywords_from_yml(yml_dict, "process_cif")
    assert compulsory == ["_cell_length_a"], "Only specified compulsory keywords should be present"
    # Ensure _cell_length_a is not duplicated in optional, despite being in both sets and compulsory
    assert sorted(optional) == sorted(list(set(["_cell_length_b", "_cell_angle_alpha", "_cell_volume"]))), \
        "Optional keywords should be unique and exclude compulsory keywords"

def test_cif_file_unified_yml_instr(temp_cif_file, tmp_path):
    output_cif_path = tmp_path / "output.cif"

    yml_path = tmp_path / "config.yml"

    # Mock YAML content
    yml_content = dedent("""
    keyword_sets :
      test1:
        required : [
          _cell_length_a
        ]
        optional : [
          _atom_site_fract_y
        ]
      test2:
        required : [
          _invalid_keyword, _atom_site_fract_x
        ]
    commands :
      process_cif:
        merge_su: true
        custom_cif_categories: [custom]
        required_keyword_sets: [test1]
        optional_keyword_sets: [test2]
        required_keywords: [_cell_length_b]
        optional_keywords: [_atom_site_label]
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