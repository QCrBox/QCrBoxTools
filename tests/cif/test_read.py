# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from iotbx import cif
import numpy as np
import pytest

from qcrboxtools.cif.read import(
    cifdata_str_or_index,
    split_su_single,
    split_sus
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

@pytest.mark.parametrize('string, test_value, test_su', [
    ('100(1)', 100.0, 1),
    ('-100(20)', -100.0, 20.0),
    ('0.021(2)', 0.021, 0.002),
    ('-0.0213(3)', -0.0213, 0.0003),
    ('2.1(1.3)', 2.1, 1.3),
    ('0.648461', 0.648461, np.nan)
])
def test_split_su_single(string: str, test_value: float, test_su: float):
    """Test the `split_su_single` function with various formatted strings."""
    val, su = split_su_single(string)
    assert val == pytest.approx(test_value)
    if np.isnan(su) or np.isnan(test_su):
        assert np.isnan(su) and np.isnan(test_su)
    else:
        assert su == pytest.approx(test_su)

def test_split_sus():
    """Test the `split_sus` function."""
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

    values, sus = split_sus(strings)

    for value, su, (check_value, check_su) in zip(values, sus, solutions):
        assert value == pytest.approx(check_value)
        assert su == pytest.approx(check_su)