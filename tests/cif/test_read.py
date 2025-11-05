# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from textwrap import dedent

import pytest
from iotbx import cif

from qcrboxtools.cif.read import (
    cif_model_to_unified_su,
    cifdata_str_or_index,
    read_cif_as_unified,
    read_cif_safe,
)


def test_cifdata_str_or_index_by_str():
    # Setup CIF model with mock data
    cif_content = """
    data_test_block
    _tag value
    """
    model = cif.reader(input_string=cif_content).model()

    # Test retrieval by string identifier
    block, identifier = cifdata_str_or_index(model, "test_block")
    assert identifier == "test_block"
    assert "_tag" in block


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
    assert identifier == "second_block"
    assert "_tag_second" in block


def test_cifdata_str_or_index_invalid_str():
    # Setup CIF model with mock data
    cif_content = """
    data_test_block
    _tag value
    """
    model = cif.reader(input_string=cif_content).model()

    # Test retrieval with invalid string identifier
    with pytest.raises(ValueError):
        cifdata_str_or_index(model, "nonexistent_block")


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
        cifdata_str_or_index(model, "invalid_index")


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
    for dataset in ["test", None]:
        # Test without any processing
        output = read_cif_as_unified(cif_path, dataset=dataset, split_sus=False, convert_keywords=False)
        if dataset is None:
            # also test cif conversion
            output = output["test"]
        assert "_test_value_with_su" in output
        assert output["_test_value_with_su"] == "1.23(4)"

        # Test with standard uncertainties split
        output = read_cif_as_unified(cif_path, dataset=dataset, split_sus=True, convert_keywords=False)
        if dataset is None:
            output = output["test"]
        assert "_test_value_with_su_su" in output
        assert output["_test_value_with_su"] == "1.23"
        assert output["_test_value_with_su_su"] == "0.04"

        # Test with standard uncertainties split and unified keywords
        output = read_cif_as_unified(
            cif_path,
            dataset=dataset,
            split_sus=True,
            convert_keywords=True,
            custom_categories=["test"],
        )
        if dataset is None:
            output = output["test"]
        assert "_test.value_with_su_su" in output
        assert output["_test.value_with_su"] == "1.23"
        assert output["_test.value_with_su_su"] == "0.04"


def test_read_cif_safe_with_str(cif_path):
    """Test read_cif_safe with string path."""
    cif_model = read_cif_safe(str(cif_path))
    assert "test" in cif_model.keys()
    assert "_test_value_with_su" in cif_model["test"]


def test_read_cif_safe_with_path(cif_path):
    """Test read_cif_safe with pathlib.Path object."""
    cif_model = read_cif_safe(Path(cif_path))
    assert "test" in cif_model.keys()
    assert "_test_value_with_su" in cif_model["test"]


def test_cif_model_to_unified_su_no_processing():
    """Test cif_model_to_unified_su without any processing."""
    cif_content = dedent("""
        data_block1
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)
    cif_model = cif.reader(input_string=cif_content).model()
    
    result = cif_model_to_unified_su(cif_model, convert_keywords=False, split_sus=False)
    
    assert "_test_value_with_su" in result["block1"]
    assert result["block1"]["_test_value_with_su"] == "1.23(4)"


def test_cif_model_to_unified_su_split_only():
    """Test cif_model_to_unified_su with only SU splitting."""
    cif_content = dedent("""
        data_block1
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)
    cif_model = cif.reader(input_string=cif_content).model()
    
    result = cif_model_to_unified_su(cif_model, convert_keywords=False, split_sus=True)
    
    assert "_test_value_with_su" in result["block1"]
    assert "_test_value_with_su_su" in result["block1"]
    assert result["block1"]["_test_value_with_su"] == "1.23"
    assert result["block1"]["_test_value_with_su_su"] == "0.04"


def test_cif_model_to_unified_su_convert_only():
    """Test cif_model_to_unified_su with only keyword conversion."""
    cif_content = dedent("""
        data_block1
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)
    cif_model = cif.reader(input_string=cif_content).model()
    
    result = cif_model_to_unified_su(cif_model, convert_keywords=True, split_sus=False, custom_categories=["test"])
    
    assert "_test.value_with_su" in result["block1"]
    assert result["block1"]["_test.value_with_su"] == "1.23(4)"


def test_cif_model_to_unified_su_full_processing():
    """Test cif_model_to_unified_su with both keyword conversion and SU splitting."""
    cif_content = dedent("""
        data_block1
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)
    cif_model = cif.reader(input_string=cif_content).model()
    
    result = cif_model_to_unified_su(cif_model, convert_keywords=True, split_sus=True, custom_categories=["test"])
    
    assert "_test.value_with_su" in result["block1"]
    assert "_test.value_with_su_su" in result["block1"]
    assert result["block1"]["_test.value_with_su"] == "1.23"
    assert result["block1"]["_test.value_with_su_su"] == "0.04"
