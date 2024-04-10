# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
import re
from collections import defaultdict
from pathlib import Path
from typing import List

from iotbx.cif import model, reader


def trim_cif_file(
    file_path: Path,
    block_name: str,
    keep_only_regexes: List[str],
    delete_regexes: List[str],
    delete_empty_entries: bool = True,
) -> None:
    """
    Trims entries in a specified CIF block of a file based on regex patterns.

    Reads a CIF file, extracts a block, and trims its entries based on provided
    regex patterns for keeping or deleting. The modified block is then written
    back to the same CIF file.

    Parameters
    ----------
    file_path : Path
        The path to the CIF file to be modified.
    block_name : str
        The name of the block within the CIF file to trim.
    keep_only_regexes : List[str]
        Regex patterns specifying which entries to keep when any is fulfilled.
        If empty, keep all entries.
    delete_regexes : List[str]
        Regex patterns specifying which entries to delete when any is fulfilled.
    delete_empty_entries : bool, optional
        Indicates whether to delete entries with '?' as their value, by default True.

    """
    cif = reader(str(file_path)).model()

    block = cif[block_name]

    new_block = trim_cif_block(block, keep_only_regexes, delete_regexes, delete_empty_entries)

    cif[block_name] = new_block

    Path(file_path).write_text(str(cif), encoding="UTF-8")


def keep_single_kw(name: str, keep_only_regexes: List[str], delete_regexes: List[str]) -> bool:
    """
    Determines if a CIF entry name should be kept based on regex patterns.

    Evaluates if a given entry name matches any of the `keep_only_regexes` and
    does not match any of the `delete_regexes`.

    Parameters
    ----------
    name : str
        The name of the CIF entry to evaluate.
    keep_only_regexes : List[str]
        Regex patterns specifying which entries to keep if any is fulfilled.
        If empty, keep all entries.
    delete_regexes : List[str]
        Regex patterns specifying which entries to delete if any is fulfilled.

    Returns
    -------
    bool
        True if the entry matches any keep condition and does not match any
        delete condition, False otherwise.

    """
    if len(keep_only_regexes) == 0:
        condition1 = True
    else:
        condition1 = any(re.fullmatch(pattern, name) is not None for pattern in keep_only_regexes)

    condition2 = all(re.fullmatch(pattern, name) is None for pattern in delete_regexes)
    return condition1 and condition2


def trim_cif_block(
    old_block: model.block,
    keep_only_regexes: List[str],
    delete_regexes: List[str],
    delete_empty_entries: bool = True,
) -> model.block:
    """
    Trims entries from a CIF block based on regex patterns.

    Processes a given CIF block, removing or keeping entries based on the
    specified regex patterns. Optionally, entries with '?' as their value can
    also be removed.

    Parameters
    ----------
    old_block : model.block
        The original CIF block to trim.
    keep_only_regexes : List[str]
        Regex patterns specifying which entries to keep. If empty, keep all entries.
    delete_regexes : List[str]
        Regex patterns specifying which entries to delete.
    delete_empty_entries : bool, optional
        Indicates whether to delete entries with '?' as their value, by default True.

    Returns
    -------
    model.block
        The trimmed CIF block with only the desired entries retained.

    """

    entry2loop_name = {}
    for loop_name, loop_entries in old_block.loops.items():
        for entry_name in loop_entries.keys():
            entry2loop_name[entry_name] = loop_name

    keep_kws = list(old_block.keys())

    keep_kws = [kw for kw in keep_kws if keep_single_kw(kw, keep_only_regexes, delete_regexes)]

    new_loops = defaultdict(dict)
    for entry in keep_kws:
        if entry in entry2loop_name:
            new_loops[entry2loop_name[entry]][entry] = old_block[entry]

    converted_block = model.block()

    for entry in keep_kws:
        if entry in entry2loop_name:
            new_loop = new_loops[entry2loop_name[entry]]
            if new_loop is not None:
                converted_block.add_loop(model.loop(data=new_loop))
                new_loops[entry2loop_name[entry]] = None
        elif old_block[entry] == "?" and delete_empty_entries:
            continue
        else:
            converted_block.add_data_item(entry, old_block[entry])

    return converted_block
