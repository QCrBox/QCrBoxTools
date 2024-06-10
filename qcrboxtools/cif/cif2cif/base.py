# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import List, Optional, Union

from ..entries import cif_to_specific_keywords, cif_to_unified_keywords, entry_to_unified_keyword
from ..read import read_cif_as_unified, read_cif_safe
from ..uncertainties import merge_su_cif


def cif_file_to_unified(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    convert_keywords: bool = True,
    custom_categories: Optional[List[str]] = None,
    split_sus: bool = True,
) -> None:
    """
    Reads, processes, and writes a CIF file with optional modifications.

    Processes an input CIF file by optionally converting keywords and splitting
    standard uncertainties (SUs), then saves the modified content to a new file.

    Parameters
    ----------
    input_cif_path : Union[str, Path]
        The input CIF file path.
    output_cif_path : Union[str, Path]
        The output file path for the processed CIF.
    convert_keywords : bool, optional
        If True, converts keywords to a unified format.
    custom_categories : Optional[List[str]], optional
        Custom categories for keyword conversion, if applicable.
    split_sus : bool, optional
        If True, splits values from their SUs in the CIF content.

    Returns
    -------
    None
    """
    cif_model = read_cif_as_unified(
        input_cif_path,
        convert_keywords=convert_keywords,
        custom_categories=custom_categories,
        split_sus=split_sus,
    )

    # Write the modified CIF model to the specified output file.
    Path(output_cif_path).write_text(str(cif_model), encoding="UTF-8")


def cif_file_to_specific(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    required_entries: List[str] = None,
    optional_entries: List[str] = None,
    custom_categories: List[str] = None,
    merge_su: bool = False,
):
    """
    Processes a CIF file, optionally merges standard uncertainties, and filters by specified
    keywords.

    Reads a CIF file from the specified input path and performs a series of processing steps:
    optionally merging numerical values with their standard uncertainties (SUs), filtering the data
    to retain only specified required and optional entries along with any custom categories,
    and finally writing the processed content to a new file at the output path.
    If neither required, nor optional entries are provided, all entries will be written to
    file.

    Parameters
    ----------
    input_cif_path : Union[str, Path]
        The file path to the input CIF file to be processed.
    output_cif_path : Union[str, Path]
        The file path where the processed CIF content will be written.
    required_entries : List[str], optional
        A list of entry names that must be included in the converted CIF file. These need to be
        either entries in the CIF, aliases of entries in the CIF, or entries renamed via the
        custom categories. The keyword "all_unified" can be passed in any entry list to convert
        all present CIF entries into unified CIF entries and otherwise ignore optional
        and required entries.
    optional_entries : List[str], optional
        Entries within this list are declared to be optional and will be included if present,
        but do not raise an error if they are missing. The keyword "all_unified" can be passed
        in any entry list to convert all present CIF entries into unified CIF entries and
        otherwise ignore optional and required entries.
    custom_categories : List[str], optional
        User-defined categories (e.g., 'iucr', 'olex2', or 'shelx') that can be taken
        into account where an entry "_category.example" would be cpnverted to "_category_example"
        in a block of the provided CIF, facilitating reversions based on these categories.
    merge_su : bool, default=False
        If True, numerical values and their standard uncertainties in the CIF model are
        merged before any other processing, if the su (or an alias) is not included as a
        required or optional CIF entry. If False, the CIF model is processed without merging SUs.

    Returns
    -------
    None
        The function writes directly to the file specified by `output_cif_path`.
    """
    cif_model = read_cif_safe(input_cif_path)

    if required_entries is None:
        required_entries = []
    if optional_entries is None:
        optional_entries = []
    if custom_categories is None:
        custom_categories = []

    all_keywords = set(required_entries + optional_entries)

    if merge_su:
        unified_entries = [entry_to_unified_keyword(entry, custom_categories) for entry in all_keywords]

        # entries are explicitely requested and therefore should not be merged
        exclude_entries = [entry[:-3] for entry in unified_entries if entry.endswith("_su")]
        cif_model = merge_su_cif(cif_model, exclude=exclude_entries)

    if "all_unified" in all_keywords:
        cif_model = cif_to_unified_keywords(cif_model, custom_categories)
    elif len(all_keywords) > 0:
        cif_model = cif_to_specific_keywords(cif_model, required_entries, optional_entries, custom_categories)

    Path(output_cif_path).write_text(str(cif_model), encoding="UTF-8")
