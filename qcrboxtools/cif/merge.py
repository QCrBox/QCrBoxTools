# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

"""
This module provides utility functions for handling CIF (Crystallographic Information File) files.
It includes functionalities such as:
- Safely reading CIF files.
- Retrieving CIF datasets based on an index or identifier.
- Extracting numerical values and estimated standard uncertainties (sus) from formatted CIF strings.
- Replacing the structure block of a CIF file with another.
"""

import re
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Union

from iotbx import cif

from .read import cifdata_str_or_index, read_cif_safe
from .uncertainties import is_num_su, split_su_array


class NoCentringFoundError(Exception):
    """
    Exception raised when no centring information is found in a CIF file.
    """


class InConsistentCentringError(Exception):
    """
    Exception raised for inconsistent centring information in CIF files.
    """


def del_atom_site_condition(key: str) -> bool:
    """Check if to be deleted because an atom_site entry (these are replaced)"""
    return key.startswith("_atom_site")


def del_geom_condition(key: str) -> bool:
    """Check if to be deleted because an geom entry (these are based on outdate info)"""
    return key.startswith("_geom")


def del_refine_condition(key: str) -> bool:
    """Check if to be deleted because an refine entry (these combined cif is not converged)"""
    exceptions = ("_refine_ls_weighting", "_refine_ls_extinction")
    return key.startswith("_refine") and not key.startswith(exceptions)


def del_refln_condition(key: str) -> bool:
    """Check if to be deleted because a refln entry with calc (these are based on outdated info)"""
    return key.startswith("_refln") and "calc" in key


def del_refine_file(key: str) -> bool:
    """Check if to be deleted because if containes the res/ins file (is replaced)"""
    test_keys = (
        "_shelx.res_file",  # shelxl
        "_iucr.refine_instructions_details",  # olex2
    )
    return any(key == test_key for test_key in test_keys)


def replace_structure_from_cif(
    cif_path: Union[str, Path],
    cif_dataset: str,
    structure_cif_path: Union[str, Path],
    structure_cif_dataset: str,
    output_cif_path: Union[str, Path],
) -> None:
    """
    Replace the structural information in a CIF file with that from another CIF file.

    Parameters
    ----------
    cif_path : Union[str, Path]
        Path to the original CIF file.
    cif_dataset : str
        Dataset name in the original CIF to be replaced.
    structure_cif_path : Union[str, Path]
        Path to the CIF file with the replacement structure.
    structure_cif_dataset : str
        Dataset name in the replacement CIF file.
    output_cif_path : Union[str, Path]
        Path to save the modified CIF file.

    Returns
    -------
    None
    """
    cif_obj = read_cif_safe(cif_path=cif_path)
    structure_cif_obj = read_cif_safe(cif_path=structure_cif_path)

    new_cif_obj = cif_obj.copy()

    new_block, cif_dataset = cifdata_str_or_index(cif_obj, cif_dataset)

    structure_cif_block, structure_cif_dataset = cifdata_str_or_index(structure_cif_obj, structure_cif_dataset)

    conditions = (
        del_atom_site_condition,
        del_geom_condition,
        del_refine_condition,
        del_refln_condition,
        del_refine_file,
    )

    # delete loops
    keys = tuple(new_block.loops.keys())
    for key in keys:
        if any(condition(key) for condition in conditions):
            del new_block[key]

    # delete items
    for key in new_block.keys():
        if any(condition(key) for condition in conditions):
            del new_block[key]

    copy_loops = ("_atom_site", "_atom_site_aniso")

    for loop_key in copy_loops:
        loop = structure_cif_block[loop_key]

        for key, vals in loop.items():
            num_su_column = all(map(is_num_su, vals))
            brackets_column = any("(" in val for val in vals)
            if num_su_column and brackets_column:
                values, _ = split_su_array(vals)
                for i, new_val in enumerate(f"{val}" for val in values):
                    loop[key][i] = new_val

        new_block.add_loop(loop)

    keys = [key for key in structure_cif_block if key.startswith("_atom_sites")]

    add_if_present = ("_shelx.res_file", "_iucr.refine_instructions_details")
    for key in add_if_present:
        if key in structure_cif_block:
            keys.append(key)

    for key in keys:
        new_block.add_data_item(key, structure_cif_block[key])

    new_cif_obj[cif_dataset] = new_block

    with open(output_cif_path, "w", encoding="UTF-8") as fobj:
        fobj.write(str(new_cif_obj))


def check_centring_equal(block1: cif.model.block, block2: cif.model.block) -> bool:
    """
    Check if the lattice centring is the same in two CIF blocks.

    Parameters
    ----------
    block1 : cif.model.block
        The first CIF block.
    block2 : cif.model.block
        The second CIF block.

    Returns
    -------
    bool
        True if lattice centring is the same, False otherwise.

    Raises
    ------
    NoCentringFoundError
        If no centring information is found in either block.
    InConsistentCentringError
        If centring information is inconsistent within a block.
    """
    block1_centrings = []
    block2_centrings = []

    space_group_entries = ("_space_group.name_h-m_alt", "_space_group.name_hall")

    for entry in space_group_entries:
        if entry in block1:
            block1_centrings.append(block1[entry].replace("-", "")[0].upper())
        if entry in block2:
            block2_centrings.append(block2[entry].replace("-", "")[0].upper())

    if len(block1_centrings) == 0:
        raise NoCentringFoundError("No _space_group.name_h-m_alt/hall found in cif1")
    if len(block2_centrings) == 0:
        raise NoCentringFoundError("No _space_group.name_h-m_alt/hall found in cif2")
    if any(check_centring != block1_centrings[0] for check_centring in block1_centrings[1:]):
        raise InConsistentCentringError("Centrings from entries do not agree for cif1")
    if any(check_centring != block2_centrings[0] for check_centring in block2_centrings[1:]):
        raise InConsistentCentringError("Centrings from entries do not agree for cif2")
    return block1_centrings[0] == block2_centrings[0]


def check_crystal_system(block1: cif.model.block, block2: cif.model.block) -> bool:
    """
    Check if two CIF blocks belong to the same crystal system.

    Parameters
    ----------
    block1 : cif.model.block
        The first CIF block.
    block2 : cif.model.block
        The second CIF block.

    Returns
    -------
    bool
        True if both blocks belong to the same crystal system, False otherwise.
    """
    return block1["_space_group.crystal_system"] == block2["_space_group.crystal_system"]


class NonMatchingMergeKeys(Exception):
    """
    Exception raised when the keys intended for merging in CIF loops do not match.
    """


class NonExistingMergeKey(Exception):
    """
    Exception raised when no matching keys are found for merging in CIF loops.
    """


def loop_to_row_dict(loop: cif.model.loop, merge_keys: Tuple[str]) -> Dict[Tuple, Dict[str, str]]:
    """
    Convert a CIF loop to a dictionary with keys as tuples of merge key values and
    values as dictionaries of non-merge key-value pairs.

    Parameters
    ----------
    loop : cif.model.loop
        The CIF loop to be converted.
    merge_keys : Tuple[str]
        A tuple of strings representing the keys on which to merge.

    Returns
    -------
    Dict[Tuple, Dict[str, str]]
        A dictionary where each key is a tuple of values from the merge keys, and
        each value is a dictionary of non-merge key-value pairs, with default values
        as '?' for missing entries.

    Notes
    -----
    This function is a helper intended to facilitate merging CIF loops based on
    specified keys. It organizes loop data for easy comparison and merging.
    """
    keys = (tuple(vals) for vals in zip(*(loop[key] for key in merge_keys)))
    nonmerge_keys = [key for key in loop.keys() if key not in merge_keys]

    values = [
        defaultdict(lambda: "?", [(key, val) for key, val in zip(nonmerge_keys, row_vals)])
        for row_vals in zip(*(loop[key] for key in nonmerge_keys))
    ]
    return dict(zip(keys, values))


def merge_cif_loops(
    loop1: cif.model.loop, loop2: cif.model.loop, merge_on: Union[str, List[str]] = r"*\.label"
) -> cif.model.loop:
    """
    Merge two CIF loops based on matching keys specified by `merge_on`.

    Parameters
    ----------
    loop1 : cif.model.loop
        The first CIF loop to merge.
    loop2 : cif.model.loop
        The second CIF loop to merge.
    merge_on : Union[str, List[str]], optional
        A string or list of strings representing the regex patterns of keys on
        which the loops should be merged. Defaults to r'*\\.label' to match any
        key ending with '.label'.

    Returns
    -------
    cif.model.loop
        A new CIF loop that is the result of merging `loop1` and `loop2` based
        on the specified `merge_on` keys. Missing values are filles with '?'.

    Raises
    ------
    NonExistingMergeKey
        If no keys match the `merge_on` pattern in either loop.
    NonMatchingMergeKeys
        If the keys found to merge on do not have a one-to-one correspondence
        between the two loops.
    """
    if isinstance(merge_on, str):
        merge_on = [merge_on]

    merge_keys = [key for key in loop1.keys() if any(re.match(pattern, key) is not None for pattern in merge_on)]
    merge_keys_check = [key for key in loop2.keys() if any(re.match(pattern, key) is not None for pattern in merge_on)]

    keys_identical = all(key1 == key2 for key1, key2 in zip(sorted(merge_keys), sorted(merge_keys_check)))

    if len(merge_keys) == 0 and len(merge_keys_check) == 0:
        raise NonExistingMergeKey("No matching keys found for merging.")

    equal_number_keys = len(merge_keys) == len(merge_keys_check)
    if not (keys_identical and equal_number_keys):
        raise NonMatchingMergeKeys(
            (
                "Found keys for matching loop are not identical. "
                + f"loop1:{sorted(merge_keys)} loop2:{sorted(merge_keys_check)}"
            )
        )

    start_dict = loop_to_row_dict(loop1, merge_keys)
    add_dict = loop_to_row_dict(loop2, merge_keys)

    for merge_key, inner_dict in add_dict.items():
        if merge_key not in start_dict:
            start_dict[merge_key] = inner_dict
        else:
            start_dict[merge_key].update(inner_dict)

    new_dict = {key: val for key, val in zip(merge_keys, zip(*start_dict.keys()))}
    nonmerge_keys = set(
        [key for key in loop1.keys() if key not in merge_keys] + [key for key in loop2.keys() if key not in merge_keys]
    )

    nonmerge_columns = zip(*[[row[key] for key in nonmerge_keys] for row in start_dict.values()])
    non_merge_dict = {key: val for key, val in zip(nonmerge_keys, nonmerge_columns)}
    new_dict.update(non_merge_dict)

    return cif.model.loop(data=new_dict)


class NonUniqueBlockMerging(Exception):
    """
    Exception raised when a CIF block's loops are attempted to be merged more
    than once with different loops.
    """


MERGE_CIF_DEFAULT = 1


def merge_cif_blocks(
    block1: cif.model.block,
    block2: cif.model.block,
    possible_markers: Union[int, List[str], str] = MERGE_CIF_DEFAULT,
) -> cif.model.block:
    """
    Merge two CIF blocks by attempting to merge their loops based on specified key patterns.

    This function iterates over loops within each block, merging them based on regex patterns
    defined in `possible_markers`. It aims to create a unified block from two sources, handling
    key mismatches or overlaps as specified.

    Parameters
    ----------
    block1 : cif.model.block
        The first CIF block to merge.
    block2 : cif.model.block
        The second CIF block to merge. Values take precedence over block1.
    possible_markers : Union[int, List[str], str], optional
        A list of regex patterns, a single regex pattern for keys to merge loops on, or an integer
        flag to use a default set of markers. Defaults to MERGE_CIF_DEFAULT, applying a default
        set of patterns.

    Returns
    -------
    cif.model.block
        A new CIF block that results from merging `block1` and `block2`.

    Raises
    ------
    NonUniqueBlockMerging
        If a loop within either block is merged more than once with different loops, indicating
        a conflict in merging criteria or an attempt to merge non-unique loops.
    """

    if possible_markers == MERGE_CIF_DEFAULT:
        possible_markers = [
            r".*\.id",
            r".*.label_?\d*",
            r".*_refln\.index.*",
            r"_atom_type\.symbol",
        ]
    if isinstance(possible_markers, str):
        possible_markers = [possible_markers]
    # merge blocks
    used_loops1 = []
    used_loops2 = []
    new_loops = {}
    entry2loop_name = {}
    iter_product = product(block1.loops.items(), block2.loops.items(), possible_markers)
    for (loop1_name, loop1), (loop2_name, loop2), marker in iter_product:
        try:
            merged_loop = merge_cif_loops(loop1, loop2, merge_on=marker)
            if loop1_name in used_loops1:
                raise NonUniqueBlockMerging(
                    (f"loop1: {loop1_name} merged at least twice. " + f"Second merge with {loop2_name} of block2")
                )
            if loop2_name in used_loops2:
                raise NonUniqueBlockMerging(
                    (f"loop2: {loop2_name} merged at least twice. " + f"Second merge with {loop1_name} of block1")
                )
            used_loops1.append(loop1_name)
            used_loops2.append(loop2_name)
            entry2loop_name.update({entry: merged_loop.name for entry in merged_loop.keys()})
            new_loops[merged_loop.name] = merged_loop
        except (NonExistingMergeKey, NonMatchingMergeKeys):
            pass
    for block, used_loops in zip((block1, block2), (used_loops1, used_loops2)):
        missing_loops = [name for name in block.loops.keys() if name not in used_loops]
        for name in missing_loops:
            new_loops[name] = block.loops[name]
            entry2loop_name.update({entry: name for entry in block.loops[name].keys()})

    merged_block = cif.model.block()

    for entry in block1.keys():
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                merged_block.add_loop(new_loop)
                new_loops[entry2loop_name[entry]] = None
        else:
            merged_block.add_data_item(entry, block1[entry])

    for entry in block2.keys():
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                merged_block.add_loop(new_loop)
                new_loops[entry2loop_name[entry]] = None
        else:
            merged_block.add_data_item(entry, block2[entry])

    return merged_block


def merge_cif_files(
    cif_path: Union[str, Path],
    block_name: str,
    cif_path2: Union[str, Path],
    block_name2: str,
    output_path: Union[str, Path],
    output_block_name: str,
    possible_markers: Union[int, List[str], str] = MERGE_CIF_DEFAULT,
):
    """
    Merges two specified blocks from two Crystallographic Information File (CIF)
    paths into a single block, and writes the result to a new CIF file.

    This function reads two CIF files, extracts specified blocks from each,
    merges these blocks based on common indices or specified markers, and
    writes the merged block to a new CIF file with a specified block name.
    The identification and extraction of blocks are flexible, allowing by name
    or by index if the block name is not found. The merging process respects
    specified markers to align columns and rows from two different blocks
    of CIF data. The output is a new CIF file with the merged block.

    Parameters
    ----------
    cif_path : Union[str, Path]
        Path to the first CIF file.
    block_name : str
        Name or index of the block to extract from the first CIF file.
        If block_name is not present in the CIF file, the function will
        attempt to call it into an integer to select by index.
    cif_path2 : Union[str, Path]
        Path to the second CIF file.
    block_name2 : str
        Name or index of the block to extract from the second CIF file.
        If block_name2 is not present in the CIF file, the function will
        attempt to call it into an integer to select by index.
    output_path : Union[str, Path]
        Path for the output CIF file containing the merged block.
    output_block_name : str
        Name for the block in the output CIF file.
    possible_markers : Union[int, List[str], str], optional
        Regex marker(s) to use for merging the blocks. If not specified,
        uses a default set of markers or an index.
    """
    model1 = read_cif_safe(cif_path)
    if block_name in model1:
        block1 = model1[block_name]
    else:
        block1, _ = cifdata_str_or_index(model1, int(block_name))

    model2 = read_cif_safe(cif_path2)
    if block_name2 in model2:
        block2 = model2[block_name2]
    else:
        block2, _ = cifdata_str_or_index(model2, int(block_name2))

    output_block = merge_cif_blocks(block1, block2, possible_markers)

    output_cif = cif.model.cif()
    output_cif[output_block_name] = output_block

    Path(output_path).write_text(str(output_cif), encoding="UTF-8")
