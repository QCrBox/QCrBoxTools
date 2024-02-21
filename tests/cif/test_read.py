# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from iotbx import cif
import pytest
from pathlib import Path
from textwrap import dedent
import re

from qcrboxtools.cif.read import (
    cifdata_str_or_index, read_cif_as_unified, cif_file_unify_split
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
