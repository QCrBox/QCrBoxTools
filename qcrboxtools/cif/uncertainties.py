# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import re
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from iotbx.cif import model


def is_num_su(string: str, allow_brackets_missing: bool = True) -> bool:
    """
    Check if a string is compatible with numerical values and standard uncertainties.

    Parameters
    ----------
    string : str
        The string to be checked.
    allow_brackets_missing : bool
        If True, values without brackets will also be recognised as valid.

    Returns
    -------
    bool
        True if the string only contains characters valid for numerical values
        and standard uncertainties, False otherwise.
    """
    contains_brackets = "(" in string and ")" in string
    only_num_and_brackets = re.search(r"[^\d\.\-\+\(\)]", string) is None
    return (contains_brackets or allow_brackets_missing) and only_num_and_brackets


def split_su_single(input_string: str) -> Tuple[float, float]:
    """
    Extract the value and standard uncertainty from a CIF formatted string.

    Parameters
    ----------
    input_string : str
        String containing a numeric value and possibly an SU.

    Returns
    -------
    Tuple[float, float]
        The numeric value and its standard uncertainty.
    """
    input_string = str(input_string)

    if not is_num_su(input_string, allow_brackets_missing=True):
        raise ValueError(f"{input_string} is not a valid string to split into value(su)")
    su_pattern = r"([^\.]+)\.?(.*?)\(([\d\.]+)\)"
    match = re.match(su_pattern, input_string)
    if match is None:
        return float(input_string), 0.0
    if len(match.group(2)) == 0:
        return float(match.group(1)), float(match.group(3))
    magnitude = 10.0 ** (-len(match.group(2)))
    if match.group(1).startswith("-"):
        sign = -1
    else:
        sign = 1
    # append the strings to reduce floating point errors (do not use magnitude)
    value = float(match.group(1)) + sign * float("0." + match.group(2))
    su = magnitude * float(match.group(3).replace(".", ""))
    return value, su


def split_su_array(input_strings: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract values and standard uncertainties from a list of formatted strings.

    Parameters
    ----------
    input_strings : Iterable[str]
        A list of input strings, each containing a numeric value and possibly an SU.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of numeric values and their associated standard uncertainties.
    """
    values, sus = zip(*map(split_su_single, input_strings))
    return list(values), list(sus)


def split_su_block(block: model.block) -> model.block:
    """
    Splits numerical values and their standard uncertainties (SUs) into separate entries
    within a cctbx CIF block. Standard uncertainties are identified by parentheses in the
    CIF format (e.g., "1.23(4)" where "1.23" is the value and "0.04" is the SU).

    This function processes both single data items and loops within the CIF block,
    adding "_su" to the original entry name for the uncertainty values if appropriate.
    Values without uncertainty are copied as they are in the orginal block.

    Parameters
    ----------
    block : cif_model.block
        The CIF block to process.

    Returns
    -------
    cif_model.block
        A new CIF block with values and their uncertainties split into separate entries.
    """
    entry2loop_name = {}
    new_loops = defaultdict(dict)
    # handle loops first to reconstruct them
    for loop_name, loop_entries in block.loops.items():
        for entry_name, entry_vals in loop_entries.items():
            entry2loop_name[entry_name] = loop_name
            if any(is_num_su(val, allow_brackets_missing=False) for val in entry_vals):
                non_su_vals, su_vals = split_su_array(entry_vals)
                new_loops[loop_name][entry_name] = non_su_vals
                new_loops[loop_name][entry_name + "_su"] = su_vals
            else:
                new_loops[loop_name][entry_name] = entry_vals

    converted_block = model.block()

    for entry, entry_val in block.items():
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                converted_block.add_loop(model.loop(data=new_loop))
                new_loops[entry2loop_name[entry]] = None
        elif is_num_su(entry_val, allow_brackets_missing=False):
            non_su_val, su_val = split_su_single(entry_val)
            converted_block.add_data_item(entry, non_su_val)
            converted_block.add_data_item(entry + "_su", su_val)
        else:
            converted_block.add_data_item(entry, entry_val)

    return converted_block


def split_su_cif(cif: model.cif) -> model.cif:
    """
    Applies the split_su_block function to every block in a CIF model.

    This function iterates through all blocks in the given CIF model, applies
    the split_su_block function to split values and standard uncertainties (SUs)
    for each entry within the blocks, and returns a new CIF model with the
    processed blocks.

    Parameters
    ----------
    cif : iotbx.cif.model.cif
        The CIF model to process.

    Returns
    -------
    iotbx.cif.model.cif
        A new CIF model with each block processed to split values and SUs.
    """
    processed_cif = model.cif()

    for block_name, block in cif.items():
        processed_block = split_su_block(block)
        processed_cif[block_name] = processed_block

    return processed_cif


def get_su_order(su: float) -> int:
    """
    Calculate the order of magnitude of the standard uncertainty (SU).

    Determines the order of magnitude of a given standard uncertainty, adjusting
    downwards if the SU, normalized to its order of magnitude, is less than 2.

    Parameters
    ----------
    su : float
        The standard uncertainty value.

    Returns
    -------
    int
        The order of magnitude of the standard uncertainty.

    Examples
    --------
    >>> get_su_order(0.03)
    -2

    >>> get_su_order(0.012)
    -3
    """
    if su <= 0.0:
        raise ValueError(f"Received a zero or negative value for SU: {su}")
    order = np.floor(np.log10(su))
    if su < 2 * 10 ** (order):
        order -= 1
    return int(order)


def merge_su_single(value: Union[float, str], su: Union[float, str], n_digits_no_su: Optional[int] = None) -> str:
    """
    Convert a numerical value and its standard uncertainty to a string representation.

    This function formats a numerical value and its standard uncertainty (SU) into a
    string following the format "value(SU)", where the SU is rounded and placed
    within parentheses. If the standard uncertainty is zero, a specific number of
    digits (n_digits_no_su) must be specified for rounding the value.

    Parameters
    ----------
    value : Union[float, str]
        The numerical value to be formatted.
    su : Union[float, str]
        The standard uncertainty of the value. If SU is 0, n_digits_no_su must be set.
    n_digits_no_su : Optional[int], default=None
        The number of digits to round the value to if the standard uncertainty is zero.
        Must be specified if su is 0.

    Returns
    -------
    str
        The formatted string representing the value with its standard uncertainty.

    Raises
    ------
    AssertionError
        If su is 0 and n_digits_no_su is not set.

    Examples
    --------
    >>> merge_su_single(1.23456, 0.02)
    '1.23(2)'

    >>> merge_su_single(1.23456, 0, 4)
    '1.2346'
    """
    value = float(value)
    su = float(su)
    if su < 1e-30:
        assert n_digits_no_su is not None, "Encountered su=0, but n_digits_no_su not set."
        return f"{np.round(value, n_digits_no_su)}"
    order = get_su_order(su)
    if order <= 0:
        format_dict = {
            "val": np.round(value, -order),
            "su_val": np.round(su, -order) / 10 ** (order),
        }
        n_prec_digits = -order
    else:
        format_dict = {"val": np.round(value, -order), "su_val": np.round(su, -order)}
        n_prec_digits = 0
    string = f"{{val:0.{n_prec_digits}f}}({{su_val:0.0f}})"

    return string.format(**format_dict)


def merge_su_array(values: List[float], sus: List[float]) -> List[str]:
    """
    Convert arrays of values and their standard uncertainties to string representations.

    Formats each pair of value and its standard uncertainty (SU) from the given lists
    into a string following the format "value(SU)". Adjusts the number of significant
    digits for values without an SU based on the smallest non-zero SU or the precision
    required to represent the smallest value distinctly if all SUs are zero.

    Parameters
    ----------
    values : List[float]
        The list of numerical values to be formatted.
    sus : List[float]
        The list of standard uncertainties corresponding to the values.

    Returns
    -------
    List[str]
        A list of formatted strings representing each value with its standard uncertainty.

    Examples
    --------
    >>> merge_su_array([1.2345, 2.3456], [0.01, 0.02])
    ['1.23(1)', '2.35(2)']

    >>> merge_su_array([1.23456, 0.98765], [0, 0], 4)
    ['1.2346', '0.9877']
    """
    values = [float(val) for val in values]
    sus = [float(su) for su in sus]
    nonzero_sus = [su for su in sus if su > 1e-30]
    if len(nonzero_sus) > 0:
        min_order = min(map(get_su_order, nonzero_sus))
        if min_order < 0:
            digits = -min_order
        else:
            digits = 0
        return [merge_su_single(val, su, digits) for val, su in zip(values, sus)]
    # there are no non-zero standard uncertainties
    min_abs_val = min(abs(float(val)) for val in values if abs(float(val)) > 1e-30)
    n_prec_digits = -get_su_order(min_abs_val) + 5
    if n_prec_digits < 0:
        n_prec_digits = 0
    format_string = f"{{val:0.{n_prec_digits}f}}"
    return [format_string.format(val=val) for val in values]


def merge_su_block(block: model.block, exclude: Optional[List[str]] = None) -> model.block:
    """
    Merge numerical values with their standard uncertainties within a CIF block
    according to the crystallographic information framework (CIF)
    standards. The function handles both singular data items and looped entries,
    ensuring that numerical values and their SUs are properly combined and updated
    within the new block structure.

    Parameters
    ----------
    block : model.block
        The input CIF block containing numerical values and potential SUs.
    exclude : List[str]
        A list of strings that represent entries that will not be merged with their
        su, optional.

    Returns
    -------
    model.block
        A new CIF block where numerical values and their corresponding SUs have been
        merged. This block maintains the original structure of data items and loops,
        with updated values where applicable.

    Notes
    -----
    - Assumes the standard uncertainty for a numerical value is represented with the
      same tag suffixed by '_su'.
    - Skips merging for entries without a corresponding SU.
    - If an entry has a corresponding '_su' entry, the numerical value and its SU are
      merged using the `merge_su_array` function for looped data or `merge_su_single`
      for singular data items.
    - Entries exclusively containing SUs (ending in '_su') will only be output in
      the final block if no base entry was present and merging was therefore not
      possible.

    Examples
    --------
    Given a block with entries like:
    ```
    _cell.length_a 10.0
    _cell.length_a_su 0.03
    _cell.length_b 20.0
    _cell.length_b_su 0.02
    ```
    the returned block will contain merged entries like:
    ```
    _cell.length_a 10.0(3)
    _cell.length_b 20.0(2)
    ```
    """
    entry2loop_name = {}
    new_loops = defaultdict(dict)
    if exclude is None:
        exclude = []

    for loop_name, loop_entries in block.loops.items():
        for entry_name, entry_vals in loop_entries.items():
            entry2loop_name[entry_name] = loop_name
            is_merged_su = all(
                (
                    entry_name.endswith("_su"),
                    entry_name[:-3] in loop_entries,
                    entry_name[:-3] not in exclude,
                )
            )
            if is_merged_su:
                continue
            if entry_name + "_su" in loop_entries and entry_name not in exclude:
                merged_vals = merge_su_array(entry_vals, loop_entries[entry_name + "_su"])
                new_loops[loop_name][entry_name] = merged_vals
            else:
                new_loops[loop_name][entry_name] = entry_vals

    converted_block = model.block()

    for entry, entry_val in block.items():
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                converted_block.add_loop(model.loop(data=new_loop))
                new_loops[entry2loop_name[entry]] = None
        elif all((entry.endswith("_su"), entry[:-3] in block, entry[:-3] not in exclude)):
            continue
        elif entry + "_su" in block and entry not in exclude:
            merged_entry_val = merge_su_single(entry_val, block[entry + "_su"])
            converted_block.add_data_item(entry, merged_entry_val)
        else:
            converted_block.add_data_item(entry, entry_val)

    return converted_block


def merge_su_cif(cif: model.cif, exclude: Optional[List[str]] = None) -> model.cif:
    """
    Merges numerical values with their standard uncertainties (SUs) across all blocks
    in a CIF model for both single data items and entries within loops adhering to the
    crystallographic information framework (CIF) conventions.

    Parameters
    ----------
    cif : model.cif
        The input CIF model, consisting of one or more blocks, each potentially
        containing numerical values alongside their standard uncertainties.
    exclude : List[str]
        A list of strings that represent entries that will not be merged with their
        su, optional.

    Returns
    -------
    model.cif
        A newly created CIF model where each block's numerical values and corresponding
        SUs have been merged. The original model's structural layout is preserved, with
        the content of each block reflecting the merged data.

    Notes
    -----
    - This function uses `merge_su_block` internally to perform the merging operation
      on each block, ensuring a consistent approach across the entire CIF model.
    - Numerical values presented without an associated SU are not modified.

    Examples
    --------
    For a CIF model containing blocks with entries such as:
    ```
    data_myblock
    _cell_length_a 10.0
    _cell_length_a_su 0.03
    _cell_length_b 20.0
    _cell_length_b_su 0.02
    ```
    the resulting CIF model will feature blocks updated to:
    ```
    data_myblock
    _cell_length_a 10.0(3)
    _cell_length_b 20.0(2)
    ```
    """
    processed_cif = model.cif()

    for block_name, block in cif.items():
        processed_block = merge_su_block(block, exclude=exclude)
        processed_cif[block_name] = processed_block

    return processed_cif
