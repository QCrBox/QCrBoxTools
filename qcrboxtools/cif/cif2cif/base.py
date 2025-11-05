# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import List, Optional, Union

from iotbx.cif import model, reader

from ..entries import cif_to_specific_keywords, cif_to_unified_keywords, entry_to_unified_keyword
from ..read import cif_model_to_unified_su, read_cif_as_unified, read_cif_safe
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


def cif_text_to_unified(
    cif_text: str,
    convert_keywords: bool = True,
    custom_categories: Optional[List[str]] = None,
    split_sus: bool = True,
) -> str:
    """
    Processes CIF content from a string, applying optional keyword conversion and SU splitting.

    Parameters
    ----------
    cif_text : str
        The CIF content as a string.
    convert_keywords : bool, optional
        If True, converts keywords to a unified format.
    custom_categories : Optional[List[str]], optional
        Custom categories for keyword conversion, if applicable.
    split_sus : bool, optional
        If True, splits values from their SUs in the CIF content.

    Returns
    -------
    str
        The processed CIF content as a string.
    """
    cif_model = reader(input_string=cif_text).model()

    if convert_keywords or split_sus:
        cif_model = cif_model_to_unified_su(
            cif_model,
            convert_keywords=convert_keywords,
            custom_categories=custom_categories,
            split_sus=split_sus,
        )

    return str(cif_model)


def is_text_cif(cif_text: str) -> bool:
    """
    Checks if the provided text is in CIF format.

    Parameters
    ----------
    cif_text : str
        The text to be checked.

    Returns
    -------
    bool
        True if the text is in CIF format, False otherwise.
    """
    for line in cif_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("data_"):
            return True
        if len(stripped) > 0 and not stripped.startswith("#"):
            # found non-empty non-comment line before any data_ line
            return False
    return False


def bytes_to_unified_if_cif(
    data: bytes,
    convert_keywords: bool = True,
    custom_categories: Optional[List[str]] = None,
    split_sus: bool = True,
) -> Optional[bytes]:
    """
    Converts bytes data to a unified CIF format if the data is valid CIF content.

    Parameters
    ----------
    data : bytes
        The CIF data as bytes.
    convert_keywords : bool, optional
        If True, converts keywords to a unified format.
    custom_categories : Optional[List[str]], optional
        Custom categories for keyword conversion, if applicable.
    split_sus : bool, optional
        If True, splits values from their SUs in the CIF content.

    Returns
    -------
    Optional[bytes]
        The processed CIF data as bytes, or None if the data is not valid CIF.
    """
    try:
        cif_text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    if not is_text_cif(cif_text):
        return data

    processed_cif_text = cif_text_to_unified(
        cif_text,
        convert_keywords=convert_keywords,
        custom_categories=custom_categories,
        split_sus=split_sus,
    )
    return processed_cif_text.encode("utf-8")


def cif_model_to_specific(
    cif_model: model.cif,
    required_entries: Optional[List[str]] = None,
    optional_entries: Optional[List[str]] = None,
    custom_categories: Optional[List[str]] = None,
    merge_su: bool = False,
) -> model.cif:
    """
    Filters and processes an iotbx CIF model to include only specific entries.

    Converts the CIF model to include only the required and optional entries
    and, if specified, merges standard uncertainties (SUs).

    Parameters
    ----------
    cif_model : model
        The CIF model object to be processed.
    required_entries : Optional[List[str]], optional
        List of required entry names that must be included in the CIF model.
        If "all_unified" is contained in this list a unified cif is returned
        instead.
        Default is an empty list.
    optional_entries : Optional[List[str]], optional
        List of optional entry names that will be included if present.
        Default is an empty list.
    custom_categories : Optional[List[str]], optional
        List of custom categories for keyword conversion, if applicable.
        Default is an empty list.
    merge_su : bool, optional
        If True, merges numerical values with their SUs before other processing.
        Default is False.

    Returns
    -------
    model
        The processed CIF model with the specified modifications.
    """
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

    return cif_model


def cif_file_to_specific(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    required_entries: Optional[List[str]] = None,
    optional_entries: Optional[List[str]] = None,
    custom_categories: Optional[List[str]] = None,
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
    """
    cif_model = read_cif_safe(input_cif_path)

    cif_model = cif_model_to_specific(
        cif_model,
        required_entries,
        optional_entries,
        custom_categories,
        merge_su,
    )

    Path(output_cif_path).write_text(str(cif_model), encoding="UTF-8")
