# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Tuple

import yaml

from .entries import cif_to_specific_keywords, cif_to_unified_keywords
from .entries.entry_conversion import entry_to_unified_keyword
from .read import read_cif_as_unified, read_cif_safe
from .uncertainties import merge_su_cif

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
    compulsory_entries: List[str] = None,
    optional_entries: List[str] = None,
    custom_categories: List[str] = None,
    merge_sus: bool = False,
):
    """
    Processes a CIF file, optionally merges standard uncertainties, and filters by specified
    keywords.

    Reads a CIF file from the specified input path and performs a series of processing steps:
    optionally merging numerical values with their standard uncertainties (SUs), filtering the data
    to retain only specified compulsory and optional entries along with any custom categories,
    and finally writing the processed content to a new file at the output path.
    If neither compulsory, nor optional entries are provided, all entries will be written to
    file.

    Parameters
    ----------
    input_cif_path : Union[str, Path]
        The file path to the input CIF file to be processed.
    output_cif_path : Union[str, Path]
        The file path where the processed CIF content will be written.
    compulsory_entries : List[str], optional
        A list of entry names that must be included in the converted CIF file. These need to be
        either entries in the CIF, aliases of entries in the CIF, or entries renamed via the
        custom categories. The keyword "all_unified" can be passed in any entry list to convert
        all present cif entries into unified cif entries and otherwise ignore optional
        and compulsory entries.
    optional_entries : List[str], optional
        Entries within this list are declared to be optional and will be included if present,
        but do not raise an error if they are missing. The keyword "all_unified" can be passed
        in any entry list to convert all present cif entries into unified cif entries and
        otherwise ignore optional and compulsory entries.
    custom_categories : List[str], optional
        User-defined categories (e.g., 'iucr', 'olex2', or 'shelx') that can be taken
        into account where an entry "_category.example" would be cpnverted to "_category_example"
        in a block of the provided CIF, facilitating reversions based on these categories.
    merge_sus : bool, default=False
        If True, numerical values and their standard uncertainties in the CIF model are
        merged before any other processing, if the su (or an alias) is not included as a
        compulsory or optional cif entry. If False, the CIF model is processed without merging SUs.

    Returns
    -------
    None
        The function writes directly to the file specified by `output_cif_path`.
    """
    cif_model = read_cif_safe(input_cif_path)

    if compulsory_entries is None:
        compulsory_entries = []
    if optional_entries is None:
        optional_entries = []
    if custom_categories is None:
        custom_categories = []

    all_keywords = set(compulsory_entries + optional_entries)

    if merge_sus:
        unified_entries = [entry_to_unified_keyword(entry, custom_categories) for entry in all_keywords]

        # entries are explicitely requested and therefore should not be merged
        exclude_entries = [entry[:-3] for entry in unified_entries if entry.endswith("_su")]
        cif_model = merge_su_cif(cif_model, exclude=exclude_entries)

    if "all_unified" in all_keywords:
        cif_model = cif_to_unified_keywords(cif_model, custom_categories)
    elif len(all_keywords) > 0:
        cif_model = cif_to_specific_keywords(cif_model, compulsory_entries, optional_entries, custom_categories)

    Path(output_cif_path).write_text(str(cif_model), encoding="UTF-8")


class NoKeywordsError(BaseException):
    """
    Exception raised when no keywords are found for a given command in the YAML configuration.

    Attributes
    ----------
    message : str
        Explanation of the error
    """

def cif_entries_from_yml(yml_dict: Dict[str, Any], command: str, input_or_output: str) -> Tuple[List[str], List[str]]:
    """
    Extracts compulsory and optional cif_entries for a given command from a YAML configuration
    dictionary.

    Parses the YAML configuration dictionary to find and compile lists of compulsory and optional
    cif entries specified under a given command. This includes directly specified keywords and those
    included in keyword sets referenced by the command.

    Parameters
    ----------
    yml_dict : dict
        The dictionary obtained from parsing the YAML configuration file.
    command : str
        The command name to look up in the YAML dictionary for extracting cif entry specifications.
    input_or_output : str
        The section of the command to look up in the YAML dictionary. Must be either 'input' or 'output'.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists: the first list contains compulsory cif entries, and the second
        list contains optional cif entries. Both lists are de-duplicated.

    Raises
    ------
    KeyError
        If the specified command or referenced cif entry sets are not found in the YAML dictionary.
    NameError
        If the keyword set contains entries other than 'name', 'required' or 'optional', indicating
        a possible typo.

    Examples
    --------
    >>> yml_dict = {
    ...     "commands": [
    ...         {
                    "name": "process_cif",
                    "cif_input": {
        ...             "required_cif_entries": ["_cell_length_a"],
        ...             "optional_cif_entries": ["_atom_site_label"]
        ...         },
    ...         }
    ...     ]
    ... }
    >>> cif_entries_from_yml(yml_dict, "process_cif", "input")
    (['_cell_length_a'], ['_atom_site_label'])
    """
    entry_sets = {eset["name"]: eset for eset in yml_dict.get("cif_entry_sets", [])}
    if input_or_output == "input":
        lookup_section = "cif_input"
    elif input_or_output == "output":
        lookup_section = "cif_output"
    else:
        raise ValueError("input_or_output must be either 'input' or 'output'.")

    try:
        command_dict = next(cmd for cmd in yml_dict["commands"] if cmd["name"] == command)
    except KeyError as exc:
        raise KeyError("One or more commands are missing a name entry in the yml_dict.") from exc
    except StopIteration as exc:
        raise KeyError(f"Command {command} not found in yml_dict.") from exc

    try:
        options = command_dict[lookup_section]
    except KeyError as exc:
        raise KeyError(f"No {lookup_section} section found in yml definition of command {command}.") from exc

    possible_entries = (
        "required_cif_entry_sets",
        "required_cif_entries",
        "optional_cif_entry_sets",
        "optional_cif_entries",
    )
    if not any(entry in options for entry in possible_entries):
        raise NoKeywordsError("Command {command} has no entries defining optional or necessary keywords.")
    compulsory_kws = options.get("required_cif_entries", [])
    optional_kws = options.get("optional_cif_entries", [])
    for kwset in options.get("required_cif_entry_sets", []):
        try:
            kwset_dict = entry_sets[kwset]
        except KeyError as exc:
            raise KeyError(f"Keyword set {kwset} not found.") from exc
        if any(key not in ("name", "required", "optional") for key in kwset_dict):
            raise NameError(
                ('Found entry other than "name", "required" or "optional"' + f"in keyword set {kwset}. Typo?")
            )
        compulsory_kws += kwset_dict.get("required", [])
        optional_kws += kwset_dict.get("optional", [])

    for kwset in options.get("optional_cif_entry_sets", []):
        try:
            kwset_dict = entry_sets[kwset]
        except KeyError as exc:
            raise KeyError(f"Keyword set {kwset} not found.") from exc
        if any(key not in ("name", "required", "optional") for key in kwset_dict):
            raise NameError(
                ('Found entry other than "name", "required" or "optional"' + f"in keyword set {kwset}. Typo?")
            )
        optional_kws += kwset_dict.get("required", [])
        optional_kws += kwset_dict.get("optional", [])
    compulsory_kws = list(set(compulsory_kws))
    optional_kws = list(set(kw for kw in optional_kws if kw not in compulsory_kws))
    return compulsory_kws, optional_kws


def cif_file_to_specific_by_yml(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    yml_path: Union[str, Path],
    command: str,
) -> None:
    """
    Processes a CIF file based on instructions defined in a YAML configuration, applying
    specified keyword transformations and standard uncertainty mergers.

    This function reads the CIF file specified by `input_cif_path` and processes it according
    to instructions defined in a YAML file (`yml_path`) under a specific command. It supports
    operations such as merging standard uncertainties and filtering CIF entries based on
    compulsory and optional keywords derived from the YAML configuration. The processed CIF
    content is written to the path specified by `output_cif_path`.

    Parameters
    ----------
    input_cif_path : Union[str, Path]
        The file path to the input CIF file to be processed.
    output_cif_path : Union[str, Path]
        The file path where the processed CIF content will be written.
    yml_path : Union[str, Path]
        The file path to the YAML file containing processing instructions.
    command : str
        The specific command within the YAML file to follow for processing the CIF file.

    Raises
    ------
    KeyError
        If the specified command is not found in the YAML configuration.

    Notes
    -----
    This file was developed for exposeing commands within QCrBox. See this project or the
    test of this function for an example of how such a yml file might look like.
    """
    with open(yml_path, "r", encoding="UTF-8") as fobj:
        yml_dict = yaml.safe_load(fobj)

    try:
        options = next(cmd for cmd in yml_dict["commands"] if cmd["name"] == command)
    except StopIteration as exc:
        raise KeyError(f"Command {command} not found in {yml_path}.") from exc

    compulsory_entries, optional_entries = cif_entries_from_yml(yml_dict, command, "input")
    merge_sus = options.get("merge_cif_su", False)
    custom_categories = options.get("custom_cif_categories", [])

    cif_file_to_specific(
        input_cif_path,
        output_cif_path,
        compulsory_entries,
        optional_entries,
        custom_categories,
        merge_sus,
    )
