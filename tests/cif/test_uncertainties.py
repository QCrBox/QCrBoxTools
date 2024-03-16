# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import pytest
from iotbx.cif import model

from qcrboxtools.cif.uncertainties import (
    get_su_order,
    merge_su_array,
    merge_su_block,
    merge_su_cif,
    merge_su_single,
    split_su_array,
    split_su_block,
    split_su_cif,
    split_su_single,
)


@pytest.mark.parametrize(
    "string, test_value, test_su",
    [
        ("100(1)", 100.0, 1),
        ("-100(20)", -100.0, 20.0),
        ("0.021(2)", 0.021, 0.002),
        ("-0.0213(3)", -0.0213, 0.0003),
        ("2.1(13)", 2.1, 1.3),  # the correct one
        ("2.1(1.3)", 2.1, 1.3),  # the sensible one, might also occur
        ("0.648461", 0.648461, 0.0),
    ],
)
def test_split_su_single(string: str, test_value: float, test_su: float):
    """Test the `split_su_single` function with various formatted strings."""
    val, su = split_su_single(string)
    assert val == pytest.approx(test_value)
    # if np.isnan(su) or np.isnan(test_su):
    #    assert np.isnan(su) and np.isnan(test_su)
    # else:
    assert su == pytest.approx(test_su)


def test_split_sus():
    """Test the `split_sus` function."""
    strings = [
        "0.03527(13)",
        "0.02546(10)",
        "0.02949(11)",
        "0.00307(9)",
        "0.01031(9)",
        "-0.00352(8)",
    ]

    solutions = [
        (0.03527, 0.00013),
        (0.02546, 0.00010),
        (0.02949, 0.00011),
        (0.00307, 0.00009),
        (0.01031, 0.00009),
        (-0.00352, 0.00008),
    ]

    values, sus = split_su_array(strings)

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
    block.add_data_item("_test.value_with_su", "1.23(4)")
    block.add_data_item("_test.value_without_su", "5.67")
    # Looped entries
    loop_data = {
        "_test.loop_value_with_su": ["2.34(5)", "3.45(6)", "1.02"],
        "_test.loop_value_without_su": ["7.89", "8.90", "12.12"],
    }
    block.add_loop(model.loop(data=loop_data))
    return block


def test_split_su_block(test_block):
    """
    Test that `split_su_block` correctly splits values with SUs and leaves others unchanged.
    """
    split_block = split_su_block(test_block)

    # Test non-looped entries
    assert split_block["_test.value_with_su"] == "1.23"
    assert split_block["_test.value_with_su_su"] == "0.04"
    assert split_block["_test.value_without_su"] == "5.67"
    assert "_test.value_without_su_su" not in split_block

    # Test looped entries
    loop = split_block.get_loop("_test")
    assert list(loop["_test.loop_value_with_su"]) == ["2.34", "3.45", "1.02"]
    assert list(loop["_test.loop_value_with_su_su"]) == ["0.05", "0.06", "0"]
    assert list(loop["_test.loop_value_without_su"]) == ["7.89", "8.90", "12.12"]


@pytest.fixture
def cif_model_with_blocks(test_block):
    """
    Create a CIF model with two blocks for testing the split_su_cif function. This setup
    uses the modified `test_block` fixture to simulate different scenarios within each block.
    """
    cif = model.cif()
    # First block directly from the test_block fixture
    cif["block1"] = test_block

    # Second block, slightly modified to differentiate from the first block
    block2 = test_block.deepcopy()
    # Modify some values in block2 to test the function's ability to handle variations
    block2["_test.value_with_su"] = "9.01(2)"  # Change the value and SU
    loop_data = block2.get_loop("_test")
    loop_data["_test.loop_value_with_su"] = ["4.56(7)", "5.67(8)", "1.02"]  # Modify loop values
    cif["block2"] = block2

    return cif


def test_split_su_cif(cif_model_with_blocks):
    """
    Test the split_su_cif function to ensure it correctly processes multiple blocks within
    a CIF model, accurately splitting values with SUs and leaving others unchanged.
    """
    processed_cif = split_su_cif(cif_model_with_blocks)

    # Assertions for block1, similar to those in test_split_su_block
    block1 = processed_cif["block1"]
    assert block1["_test.value_with_su"] == "1.23"
    assert block1["_test.value_with_su_su"] == "0.04"
    assert block1["_test.value_without_su"] == "5.67"
    # Check looped entries in block1
    loop1 = block1.get_loop("_test")
    assert list(loop1["_test.loop_value_with_su"]) == ["2.34", "3.45", "1.02"]
    assert list(loop1["_test.loop_value_with_su_su"]) == ["0.05", "0.06", "0"]
    assert list(loop1["_test.loop_value_without_su"]) == ["7.89", "8.90", "12.12"]

    # Assertions for block2, ensuring modifications are processed correctly
    block2 = processed_cif["block2"]
    assert block2["_test.value_with_su"] == "9.01"
    assert block2["_test.value_with_su_su"] == "0.02"
    # Check looped entries in block2
    loop2 = block2.get_loop("_test")
    assert list(loop2["_test.loop_value_with_su"]) == ["4.56", "5.67", "1.02"]
    assert list(loop2["_test.loop_value_with_su_su"]) == ["0.07", "0.08", "0"]

    # Ensure entries without SUs remain unchanged in both blocks
    for block in [block1, block2]:
        assert list(block.get_loop("_test")["_test.loop_value_without_su"]) == [
            "7.89",
            "8.90",
            "12.12",
        ]


@pytest.mark.parametrize(
    "value, su, n_digits_no_su, expected_output",
    [
        (1.23456, 0.019, None, "1.235(19)"),  # Correct formatting for SU < 2
        (1.23456, 0.0014, None, "1.2346(14)"),  # SU < 2, precise to more decimals
        (1.23456, 0.0099, None, "1.235(10)"),  # Close to 1, two digits for SU
        (1.23456, 0.021, None, "1.23(2)"),  # SU > 2, normal rounding
        (1234.56, 1.9, None, "1234.6(19)"),  # 1.0 < SU < 2.0, see IToC G (2006) p. 23
        (123456, 1900, None, "123500(1900)"),  # Larger numbers, precise SU
        (-1.23456, 0.019, None, "-1.235(19)"),  # Negative value, precise SU
        (1.23456, 0, 4, "1.2346"),  # Zero SU with specified digits
        (123456, 190, None, "123460(190)"),  # Precise SU handling
        (2.34567, 0.000451, None, "2.3457(5)"),  # Precise to more decimals, SU > 2 round up
        (4.56789, 0.044, None, "4.57(4)"),  # SU less decimals round down
        (0.11645, 0.00009, None, "0.11645(9)"),  # small su
    ],
)
def test_merge_su_single(value, su, n_digits_no_su, expected_output):
    assert merge_su_single(value, su, n_digits_no_su) == expected_output


def test_merge_su_single_raises_error():
    with pytest.raises(AssertionError):
        merge_su_single(1.23456, 0)  # Missing n_digits_no_su argument


@pytest.mark.parametrize(
    "input_su, expected_order",
    [
        (0.05, -2),  # Positive SU less than 1
        (21, 1),  # SU greater than 1
        (0.1, -2),  # SU less than 1 but greater than 0.1
        (0.2, -1),  # Boundary case where SU normalized is exactly 2
        (0.00003, -5),  # Very small SU
    ],
)
def test_get_su_order(input_su, expected_order):
    assert get_su_order(input_su) == expected_order, f"Failed on SU={input_su}"


@pytest.mark.parametrize(
    "invalid_su",
    [
        -0.01,  # Negative SU
        0,  # Zero SU
    ],
)
def test_get_su_order_invalid_inputs(invalid_su):
    with pytest.raises(ValueError):
        get_su_order(invalid_su)


@pytest.mark.parametrize(
    "values, sus, expected_output",
    [
        # Scenario with no zero SUs
        ([1.234, 2.345], [0.01, 0.02], ["1.234(10)", "2.35(2)"]),
        # Scenario with mixed SUs (including zero, non-SU has highest SU precision)
        ([1.2341, 2.3452, 3.4560], [0.01, 0, 0.03], ["1.234(10)", "2.345", "3.46(3)"]),
        # Scenario with all zero SUs
        ([1.23456789, 2.3456], [0, 0], ["1.234568", "2.345600"]),
        # Scenario with all zero SUs and small values
        ([0.0000123456789, 2.3456], [0, 0], ["0.00001234568", "2.34560000000"]),
    ],
)
def test_merge_su_array(values, sus, expected_output):
    assert (
        merge_su_array(values, sus) == expected_output
    ), f"Failed for values={values} and SUs={sus}"


@pytest.fixture
def sample_block_with_su() -> model.block:
    """
    A pytest fixture that creates a test CIF block with numerical values and their
    corresponding standard uncertainties, both for non-looped and looped entries,
    specifically designed for testing the merge_su_block function. This fixture uses
    revised naming conventions and value ranges as specified.

    Returns
    -------
    model.block
        A CIF block with predefined numerical values and standard uncertainties,
        incorporating both cell length measurements and atomic site fractional
        coordinates.
    """
    block = model.block()
    # Non-looped entries using the revised naming convention
    block.add_data_item("_cell.length_a", 10.0)
    block.add_data_item("_cell.length_a_su", 0.03)
    block.add_data_item("_cell.length_b", 20.0)
    block.add_data_item("_cell.length_b_su", 0.02)
    block.add_data_item("_cell.length_c_su", 0.04)
    # Looped entries for atomic site fractional coordinates
    loop_data = {
        "_atom_site.fract_x": [0.234, -0.345, 0.456],
        "_atom_site.fract_x_su": [0.012, 0.023, 0.034],
        "_atom_site.fract_y": [0.567, 0.678, -0.789],
        "_atom_site.fract_y_su": [0.045, 0.0, 0.067],
        "_atom_site.fract_z": [0.890, -0.901, -0.012],
        "_atom_site.fract_z_su": [0.078, 0.089, 0.009],
    }
    block.add_loop(model.loop(data=loop_data))
    return block


def test_merge_su_block(sample_block_with_su):
    # Specify entries to exclude from merging
    exclude_entries = ["_cell.length_a", "_atom_site.fract_x"]

    # Perform merge with exclusion
    merged_block = merge_su_block(sample_block_with_su, exclude=exclude_entries)

    # Tests for non-looped entries with exclusion
    assert merged_block["_cell.length_a"] == "10.0", "Failed to exclude _cell.length_a from merging"
    assert (
        "_cell.length_a_su" in merged_block
    ), "_cell.length_a_su should not be deleted when _cell.length_a is excluded"

    # Test for su entry without an existing base entry
    assert (
        "_cell.length_c_su" in merged_block
    ), "_cell.length_c_su should not be deleted when _cell.length_c does not exist"

    # Test for looped entries with exclusion
    assert (
        merged_block["_atom_site.fract_x"][0] == "0.234"
    ), "Failed to exclude _atom_site.fract_x from merging"
    assert (
        "_atom_site.fract_x_su" in merged_block
    ), "_atom_site.fract_x_su should not be deleted when _atom_site.fract_x is excluded"

    # Ensure other entries not in exclude list are merged correctly
    assert (
        merged_block["_cell.length_b"] == "20.00(2)"
    ), "Failed to merge cell.length.b and its SU correctly"
    assert (
        "_cell.length_b_su" not in merged_block
    ), "Did not delete SU entry where corresponding entry existed"

    # Additional tests for looped entries not excluded
    assert (
        merged_block["_atom_site.fract_y"][1] == "0.68"
    ), "Failed to merge _atom_site.fract_y and format correctly"
    assert (
        merged_block["_atom_site.fract_z"][2] == "-0.012(9)"
    ), "Failed to merge _atom_site.fract_z and its SU correctly"
    assert (
        "_atom_site.fract_y_su" not in merged_block
    ), "Did not delete SU entry where corresponding entry existed"
    assert (
        "_atom_site.fract_z_su" not in merged_block
    ), "Did not delete SU entry where corresponding entry existed"


@pytest.fixture
def cif_model_with_mergable_blocks(sample_block_with_su) -> model.cif:
    """
    Create a CIF model with two blocks prepped for testing the merge_su_cif function.
    This setup utilizes the `sample_block_with_su` fixture to simulate a CIF model
    containing blocks with numerical values and their standard uncertainties ready for merging.

    Returns
    -------
    model.cif
        A CIF model containing two distinct blocks, each with data items and loops
        including numerical values and their corresponding standard uncertainties.
    """
    cif = model.cif()
    # First block directly from the sample_block_with_su fixture
    cif["block1"] = sample_block_with_su

    # Second block, slightly modified to differentiate from the first block
    block2 = sample_block_with_su.deepcopy()
    # Modify some values in block2 to test the function's ability to handle variations
    block2["_cell.length_a"] = 11.0  # Change the cell length value
    block2["_cell.length_a_su"] = 0.04  # Change the SU for the cell length
    block2["_cell.length_c"] = 30.0  # Add a new cell length measurement
    # Assume block.add_loop() and block.add_data_item() modify the block in-place
    loop_data = {
        "_atom_site.fract_x": [0.123, -0.234, 0.345],
        "_atom_site.fract_x_su": [0.011, 0.022, 0.033],
        "_atom_site.fract_y": [0.456, 0.567, -0.678],
        "_atom_site.fract_y_su": [0.044, 0.055, 0.066],
        "_atom_site.fract_z": [0.789, -0.890, -0.123],
        "_atom_site.fract_z_su": [0.077, 0.088, 0.099],
    }

    for key, values in loop_data.items():
        block2[key] = values
    cif["block2"] = block2

    return cif


def test_merge_su_cif(cif_model_with_mergable_blocks):
    """
    Test the merge_su_cif function to ensure it accurately processes multiple blocks
    within a CIF model, merging numerical values with their standard uncertainties (SUs)
    into a unified format, while leaving other entries unchanged.
    """
    exclude_entries = ["_cell.length_a", "_atom_site.fract_x"]
    # Process the CIF model with merge_su_cif
    processed_cif = merge_su_cif(cif_model_with_mergable_blocks, exclude=exclude_entries)

    # Assertions for block1
    block1 = processed_cif["block1"]
    assert block1["_cell.length_a"] == "10.0", "Block1: Failed to exclude _cell.length_a from merge"
    assert (
        block1["_cell.length_b"] == "20.00(2)"
    ), "Block1: Failed to merge _cell.length_b and its SU correctly"
    assert (
        "_cell.length_c" not in block1
    ), "Block1: Unexpectedly found _cell.length_c which shouldn't exist"
    # Check looped entries in block1 for correct merging
    assert (
        block1["_atom_site.fract_x"][0] == "0.234"
    ), "Block1: Failed to exclude _atom_site.fract_x from merge"
    assert (
        block1["_atom_site.fract_y"][2] == "-0.79(7)"
    ), "Block1: Failed to merge _atom_site.fract_y and its SU correctly"
    assert (
        block1["_atom_site.fract_z"][1] == "-0.90(9)"
    ), "Block1: Failed to merge _atom_site.fract_z and its SU correctly"

    # Assertions for block2, ensuring modifications are processed and merged correctly
    block2 = processed_cif["block2"]
    assert block2["_cell.length_a"] == "11.0", "Block2: Failed to exclude _cell.length_a from merge"
    assert (
        block2["_cell.length_b"] == "20.00(2)"
    ), "Block2: Failed to merge _cell.length_b and its SU correctly"
    assert (
        block2["_cell.length_c"] == "30.00(4)"
    ), "Block2: Failed to add and merge _cell.length_c and its SU correctly"
    # Check looped entries in block2 for correct merging
    assert (
        block2["_atom_site.fract_x"][0] == "0.123"
    ), "Block1: Failed to exclude _atom_site.fract_x from merge"
    assert (
        block2["_atom_site.fract_y"][2] == "-0.68(7)"
    ), "Block2: Failed to merge _atom_site.fract_y and its SU correctly"
    assert (
        block2["_atom_site.fract_z"][1] == "-0.89(9)"
    ), "Block2: Failed to merge _atom_site.fract_z and its SU correctly"
