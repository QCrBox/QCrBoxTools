# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
from collections import defaultdict
from typing import List

from iotbx.cif import model

from .entry_lookup import load_aliases


def entry_to_unified_keyword(old_name: str, custom_categories: List[str]) -> str:
    """
    Convert a cif entry to one common name for all aliases and some deprecated values
    where possible. With custom categories user defined categories (e.g. 'iucr', 'olex2'
    or 'shelx') can be taken into account where an entry _category_example would be
    converted into _category.example.

    Parameters
    ----------
    old_name : str
        The original name of the entry to be converted.
    custom_categories : List[str]
        A list of custom categories to check against the `old_name`.

    Returns
    -------
    str
        The converted name with the underscore prefix.
    """
    aliases = load_aliases()
    cut_name = old_name[1:]
    for category in custom_categories:
        if cut_name.startswith(category):
            return f"_{category}.{cut_name[len(category)+1:]}"
    return "_" + aliases.get(cut_name, cut_name)


def block_to_unified_keywords(block: model.block, custom_categories: List[str] = None) -> model.block:
    """
    Converts entries and loops within a given block to unified names using the
    'to_unified_name' function. Ordering of the entries within the block is
    retained.

    Parameters
    ----------
    block : model.block
        The original CIF block to convert.
    custom_categories : List[str], optional
        Custom categories to use for name conversion. Defaults to an empty list if
        None is provided.

    Returns
    -------
    model.block
        A new CIF block with converted entry names.
    """

    if custom_categories is None:
        custom_categories = []
    converted_block = model.block()

    # we want to retain the ordering so we pull out the loops first and insert them later
    loop_lookup = {}
    converted_loops = {}

    for _, loop in block.loops.items():
        new_loop = model.loop(
            data={
                entry_to_unified_keyword(entry_name, custom_categories): entry_data
                for entry_name, entry_data in loop.items()
            }
        )
        loop_lookup.update({entry: new_loop.name() for entry in list(loop)})
        converted_loops[new_loop.name()] = new_loop

    for entry_name, entry_content in block.items():
        if entry_name in loop_lookup and converted_loops[loop_lookup[entry_name]] is not None:
            loop_to_add = converted_loops[loop_lookup[entry_name]]
            if loop_to_add is not None:
                converted_block.add_loop(loop_to_add)
                # add loops only once
                converted_loops[loop_lookup[entry_name]] = None
            continue
        converted_block.add_data_item(entry_to_unified_keyword(entry_name, custom_categories), entry_content)
    return converted_block


def cif_to_unified_keywords(cif: model.cif, custom_categories=None):
    """
    Converts all blocks in a CIF file to use unified keyword formats.

    Iterates over each block in the given CIF file, applying `to_unified_kw_block` to
    convert entry names and loops to a unified format based on the provided custom
    categories and the `to_unified_name` function.

    Parameters
    ----------
    cif : model.cif
        The CIF file to convert.
    custom_categories : List[str], optional
        User defined categories (e.g. 'iucr', 'olex2' or 'shelx') that can be taken
        into account where an entry _category_example would be converted into
        _category.example.

    Returns
    -------
    model.cif
        A new CIF file with all blocks converted to common keywords.

    """
    new_cif = model.cif(
        {block_name: block_to_unified_keywords(block, custom_categories) for block_name, block in cif.items()}
    )
    return new_cif


def cif_to_specific_keywords(
    cif: model.cif,
    compulsory_entries: List[str],
    optional_entries: List[str],
    custom_categories: List[str],
) -> model.cif:
    """
    Converts CIF file entries to match a set of requested entries from a cif file that has
    previously been converted to the unified set of keywords. Also reverts conversions based
    on custom categories if these are provided. Will only output the requested entries in
    the given order. Entries that belong to loops in the cif objects are accumulated and
    the loops is output at the first occurence of one of its entries.

    Parameters
    ----------
    cif : model.cif
        The CIF object to be converted.
    compulsory_entries : List[str]
        A list of entry names to be included in the converted CIF file. Need to be either
        entries in cif, aliases of entries in cif or entries renamed via the custom categories.
    optional_entries : List[str]
        Entries within this list are declared to be optional and therefore will not raise
        an error when not found. They still need to be in requested entries to enable the
        output in a specific position within the newly generated cif.
    custom_categories : List[str]
        User defined categories (e.g. 'iucr', 'olex2' or 'shelx') that can be taken
        into account where an entry _category_example would look up _category.example
        in a block of the provided cif.

    Returns
    -------
    model.cif
        A new CIF object containing only the requested entries.

    """
    new_cif = model.cif(
        {
            block_name: block_to_specific_keywords(block, compulsory_entries, optional_entries, custom_categories)
            for block_name, block in cif.items()
        }
    )

    return new_cif


def block_to_specific_keywords(
    block: model.block,
    compulsory_entries: List[str],
    optional_entries: List[str],
    custom_categories: List[str],
) -> model.block:
    """
    Converts a CIF block's entries to match a set of requested entries, from a cif block
    containing unified keywords. Custom categories are used to revert
    conversions, ensuring only requested entries are output, preserving their order.
    Accumulates loop entries, outputting loops at the first occurrence of any of its
    entries

    Parameters
    ----------
    block : model.block
        The CIF block to be converted, should only contain unified keywords or keywords
        using custom categories.
    compulsory_entries : List[str]
        Entry names to include. If they are not found a ValueError will be raised
    optional_entries : List[str]
        Entries within this list are included if they are present and ignored if they
        are not within the old
    custom_categories : List[str]
        Categories to reverse unify names, matching custom formatted entries.

    Returns
    -------
    model.block
        A CIF block containing only the requested entries in specified order.

    """
    unified_comp_entries = [entry_to_unified_keyword(entry, custom_categories) for entry in compulsory_entries]
    for original_entry, unified_entry in zip(compulsory_entries, unified_comp_entries):
        if unified_entry not in block.keys():
            raise ValueError(
                f'The corresponding entry "{unified_entry}" for the requested '
                + f"non-optional entry {original_entry} could not be found in cif block."
            )

    unified_opt_entries = [entry_to_unified_keyword(entry, custom_categories) for entry in optional_entries]

    requested_entries = list(compulsory_entries) + list(optional_entries)

    entry_dict = dict(zip(unified_comp_entries + unified_opt_entries, requested_entries))

    output_entries = [(val, entry_dict[val]) for val in block if val in entry_dict]

    entry2loop_name = {}
    for loop_name, loop_entries in block.loops.items():
        for entry_name in loop_entries.keys():
            entry2loop_name[entry_name] = loop_name

    new_loops = defaultdict(dict)
    for lookup_name, entry in output_entries:
        # lookup_name = entry_to_unified_keyword(entry, custom_categories)
        if lookup_name in entry2loop_name:
            new_loops[entry2loop_name[lookup_name]][entry] = block[lookup_name]

    converted_block = model.block()

    for lookup_name, entry in output_entries:
        # lookup_name = entry_to_unified_keyword(entry, custom_categories)
        # if lookup_name not in block and entry in optional_entries:
        #    continue
        if lookup_name in entry2loop_name:
            new_loop = new_loops[entry2loop_name[lookup_name]]
            if new_loop is not None:
                converted_block.add_loop(model.loop(data=new_loop))
                new_loops[entry2loop_name[lookup_name]] = None
        else:
            converted_block.add_data_item(entry, block[lookup_name])

    return converted_block
