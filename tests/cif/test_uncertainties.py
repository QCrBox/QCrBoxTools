# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from iotbx.cif import model
import numpy as np
import pytest

from qcrboxtools.cif.uncertainties import (
    split_su_single,
    split_sus,
    split_su_block,
    split_su_cif
)

@pytest.mark.parametrize('string, test_value, test_su', [
    ('100(1)', 100.0, 1),
    ('-100(20)', -100.0, 20.0),
    ('0.021(2)', 0.021, 0.002),
    ('-0.0213(3)', -0.0213, 0.0003),
    ('2.1(1.3)', 2.1, 1.3),
    ('0.648461', 0.648461, 0.0)
])
def test_split_su_single(string: str, test_value: float, test_su: float):
    """Test the `split_su_single` function with various formatted strings."""
    val, su = split_su_single(string)
    assert val == pytest.approx(test_value)
    #if np.isnan(su) or np.isnan(test_su):
    #    assert np.isnan(su) and np.isnan(test_su)
    #else:
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

@pytest.fixture
def test_block() -> model.block:
    """
    A pytest fixture that creates a test CIF block with mixed data for testing.

    Returns
    -------
    cif_model.block
        A CIF block with predefined data items and loops for testing.
    """
    block = model.block()
    # Non-looped entries
    block.add_data_item('_test.value_with_su', '1.23(4)')
    block.add_data_item('_test.value_without_su', '5.67')
    # Looped entries
    loop_data = {
        '_test.loop_value_with_su': ['2.34(5)', '3.45(6)', '1.02'],
        '_test.loop_value_without_su': ['7.89', '8.90', '12.12'],
    }
    block.add_loop(model.loop(data=loop_data))
    return block

def test_split_su_block(test_block):
    """
    Test that `split_su_block` correctly splits values with SUs and leaves others unchanged.
    """
    split_block = split_su_block(test_block)

    # Test non-looped entries
    assert split_block['_test.value_with_su'] == '1.23'
    assert split_block['_test.value_with_su_su'] == '0.04'
    assert split_block['_test.value_without_su'] == '5.67'
    assert '_test.value_without_su_su' not in split_block

    # Test looped entries
    loop = split_block.get_loop('_test')
    assert list(loop['_test.loop_value_with_su']) == ['2.34', '3.45', '1.02']
    assert list(loop['_test.loop_value_with_su_su']) == ['0.05', '0.06', '0']
    assert list(loop['_test.loop_value_without_su']) == ['7.89', '8.90', '12.12']

@pytest.fixture
def cif_model_with_blocks(test_block):
    """
    Create a CIF model with two blocks for testing the split_su_cif function. This setup
    uses the modified `test_block` fixture to simulate different scenarios within each block.
    """
    cif = model.cif()
    # First block directly from the test_block fixture
    cif['block1'] = test_block

    # Second block, slightly modified to differentiate from the first block
    block2 = test_block.deepcopy()
    # Modify some values in block2 to test the function's ability to handle variations
    block2['_test.value_with_su'] = '9.01(2)'  # Change the value and SU
    loop_data = block2.get_loop('_test')
    loop_data['_test.loop_value_with_su'] = ['4.56(7)', '5.67(8)', '1.02']  # Modify loop values
    cif['block2'] = block2

    return cif

def test_split_su_cif(cif_model_with_blocks):
    """
    Test the split_su_cif function to ensure it correctly processes multiple blocks within
    a CIF model, accurately splitting values with SUs and leaving others unchanged.
    """
    processed_cif = split_su_cif(cif_model_with_blocks)

    # Assertions for block1, similar to those in test_split_su_block
    block1 = processed_cif['block1']
    assert block1['_test.value_with_su'] == '1.23'
    assert block1['_test.value_with_su_su'] == '0.04'
    assert block1['_test.value_without_su'] == '5.67'
    # Check looped entries in block1
    loop1 = block1.get_loop('_test')
    assert list(loop1['_test.loop_value_with_su']) == ['2.34', '3.45', '1.02']
    assert list(loop1['_test.loop_value_with_su_su']) == ['0.05', '0.06', '0']
    assert list(loop1['_test.loop_value_without_su']) == ['7.89', '8.90', '12.12']

    # Assertions for block2, ensuring modifications are processed correctly
    block2 = processed_cif['block2']
    assert block2['_test.value_with_su'] == '9.01'
    assert block2['_test.value_with_su_su'] == '0.02'
    # Check looped entries in block2
    loop2 = block2.get_loop('_test')
    assert list(loop2['_test.loop_value_with_su']) == ['4.56', '5.67', '1.02']
    assert list(loop2['_test.loop_value_with_su_su']) == ['0.07', '0.08', '0']

    # Ensure entries without SUs remain unchanged in both blocks
    for block in [block1, block2]:
        assert list(block.get_loop('_test')['_test.loop_value_without_su']) == ['7.89', '8.90', '12.12']
