# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from typing import List

from iotbx.cif import model

from .entry_conversion import to_unified_name, to_unified_kw_block

def cif_entries_present(
    block: model.block,
    custom_categories: List[str],
    cif_entries: List[str]
) -> bool:
    """
    Determine if all given CIF entries or one of their aliases are present in a
    block.

    Parameters
    ----------
    block : model.block
        CIF block to be checked.
    custom_categories : List[str]
        User defined categories (e.g. 'iucr', 'olex2' or 'shelx') that can be taken
        into account. Only needs to be provided if the categories may have been
        converted before
    cif_entries : List[str]
        Entries to check for in the unified block.

    Returns
    -------
    bool
        True if all entries are present in the unified block, False otherwise.
    """
    unified_block = to_unified_kw_block(
        block,
        custom_categories
    )

    unified_entries = [to_unified_name(entry, custom_categories) for entry in cif_entries]

    return all(entry in unified_block for entry in unified_entries)
