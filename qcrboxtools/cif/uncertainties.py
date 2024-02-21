# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import re
from typing import Tuple, Iterable
from collections import defaultdict
import numpy as np
from iotbx.cif import model

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
        raise ValueError(f'{input_string} is not a valid string to split into value(su)')
    su_pattern = r'([^\.]+)\.?(.*?)\(([\d\.]+)\)'
    match = re.match(su_pattern, input_string)
    if match is None:
        return float(input_string), 0.0
    if len(match.group(2)) == 0:
        return float(match.group(1)), float(match.group(3))
    magnitude = 10.0**(-len(match.group(2)))
    if match.group(1).startswith('-'):
        sign = -1
    else:
        sign = 1
    # append the strings to reduce floating point errors (do not use magnitude)
    value = float(match.group(1)) + sign * float('0.' + match.group(2))
    su = magnitude * float(match.group(3).replace('.', ''))
    return value, su


def split_sus(input_strings: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
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

def is_num_su(string: str, allow_brackets_missing: bool = False) -> bool:
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
    contains_brackets = '(' in string and ')' in string
    only_num_and_brackets = re.search(r'[^\d\.\-\+\(\)]', string) is None
    return (contains_brackets or allow_brackets_missing) and only_num_and_brackets

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
    #handle loops first to reconstruct them
    for loop_name, loop_entries in block.loops.items():
        for entry_name, entry_vals in loop_entries.items():
            entry2loop_name[entry_name] = loop_name
            if any(is_num_su(val) for val in entry_vals):
                non_su_vals, su_vals = split_sus(entry_vals)
                new_loops[loop_name][entry_name] = non_su_vals
                new_loops[loop_name][entry_name + '_su'] = su_vals
            else:
                new_loops[loop_name][entry_name] = entry_vals

    converted_block = model.block()

    for entry, entry_val in block.items():
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                converted_block.add_loop(model.loop(data=new_loop))
                new_loops[entry2loop_name[entry]] = None
        elif is_num_su(entry_val):
            non_su_val, su_val = split_su_single(entry_val)
            converted_block.add_data_item(entry, non_su_val)
            converted_block.add_data_item(entry + '_su', su_val)
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
    # Initialize a new CIF model to store the processed blocks
    processed_cif = model.cif()

    # Iterate over each block in the original CIF model
    for block_name, block in cif.items():
        # Apply the split_su_block function to the current block
        processed_block = split_su_block(block)
        # Add the processed block to the new CIF model
        processed_cif[block_name] = processed_block

    return processed_cif
