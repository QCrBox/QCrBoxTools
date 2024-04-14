# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

from collections import namedtuple
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from .entries import cif_to_specific_keywords, cif_to_unified_keywords
from .entries.entry_conversion import entry_to_unified_keyword
from .read import read_cif_as_unified, read_cif_safe
from .trim import trim_cif
from .uncertainties import merge_su_cif, split_su_cif


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


class NoKeywordsError(BaseException):
    """
    Exception raised when no keywords are found for a given command in the YAML configuration.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


class UnnamedCommandError(BaseException):
    """
    Exception raised when a command in the YAML configuration is missing a name entry.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


class UnknownCommandError(BaseException):
    """
    Exception raised when a command is not found in the YAML configuration.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


class EmptyCommandError(BaseException):
    """
    Exception raised when a command is found in the YAML configuration but has no content.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


class NonExistentEntrySetError(BaseException):
    """
    Exception raised when a keyword set is referenced in the YAML configuration but does not exist.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


class InvalidEntrySetError(BaseException):
    """
    Exception raised when an entry set in the YAML configuration contains an entry that is not 'name',
    'required', or 'optional' or an entry set does not have a 'name' entry.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


class MissingSectionError(BaseException):
    """
    Exception raised when cif_input or cif_output is missing in a command in the YAML configuration.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


class InvalidSectionEntryError(BaseException):
    """
    Exception raised when an invalid entry is found in the cif_input or cif_output section of a command
    in the YAML configuration.

    Attributes
    ----------
    message : str
        Explanation of the error
    """


def command_dict_from_yml(yml_dict: Dict[str, Any], command: str) -> Dict[str, Any]:
    """
    Returns the command dictionary for a given command from a YAML configuration dictionary.

    Parameters
    ----------
    yml_dict : dict
        The dictionary obtained from parsing the YAML configuration file.
    command : str
        The command name to look up in the YAML dictionary for extracting the command dictionary.

    Returns
    -------
    dict
        The command dictionary for the specified command.

    Raises
    ------
    KeyError
        If the specified command is not found in the YAML configuration.
    """
    try:
        selected_command = next(cmd for cmd in yml_dict["commands"] if cmd["name"] == command).copy()
        selected_command.pop("name")
        if len(selected_command) == 0:
            raise EmptyCommandError(f"Command {command} has no content.")
        return selected_command
    except StopIteration as exc:
        raise UnknownCommandError(f"Command {command} not found in yml_dict.") from exc
    except KeyError as exc:
        raise UnnamedCommandError("One or more commands are missing a name entry in the yml_dict.") from exc


def cif_entry_sets_from_yml(yml_dict: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Extracts keyword sets from a YAML configuration dictionary.

    Parameters
    ----------
    yml_dict : dict
        The dictionary obtained from parsing the YAML configuration file.

    Returns
    -------
    dict
        A dictionary of keyword sets, where the key is the keyword set name and the value is a dictionary
        containing 'required', and 'optional' entries.
    """
    try:
        return_dict = {}
        for eset in deepcopy(yml_dict.get("cif_entry_sets", [])):
            name = eset["name"]
            eset.pop("name")
            return_dict[name] = eset
            if any(key not in ("required", "optional") for key in eset.keys()):
                raise InvalidEntrySetError(
                    f'Found entry other than "name", "required" or "optional" in keyword set {name}. Typo?'
                )
        return return_dict
    except KeyError as exc:
        raise InvalidEntrySetError("Not all entry sets have a 'name' entry.") from exc


def cif_entries_from_entry_set(
    entry_set_names: List[str], entry_sets: Dict[str, Dict[str, List[str]]]
) -> Tuple[List[str], List[str]]:
    """
    Assembles required and optional CIF entries from a list of keyword sets.

    Parameters
    ----------
    entry_set_names : list
        A list of keyword set names to assemble the entries for.
    entry_sets : dict
        A dictionary of keyword sets, where the key is the keyword set name and the value is a dictionary
        containing 'required', and 'optional' entries.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists: the first list contains required entries, and the second list
        contains optional entries.
    """
    required = []
    optional = []
    for entry_set_name in entry_set_names:
        try:
            entry_set = entry_sets[entry_set_name]
        except KeyError as exc:
            raise NonExistentEntrySetError(f"Keyword set {entry_set_name} not found.") from exc
        required += entry_set.get("required", [])
        optional += entry_set.get("optional", [])
    return required, optional


def cif_entries_from_yml_section(
    io_section: Dict[str, Any], entry_sets: Dict[str, Dict[str, List[str]]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extracts required and optional CIF entries from a YAML configuration section.

    Parameters
    ----------
    io_section : dict
        The section of the YAML configuration dictionary containing the CIF entries.
    entry_sets : dict
        A dictionary of keyword sets, where the key is the keyword set name and the value is a dictionary
        containing 'required', and 'optional' entries.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        A tuple containing three lists: the first list contains required CIF entries, the second list
        contains optional CIF entries, and the third list contains custom categories. All lists are de-duplicated.

    Raises
    ------
    NoKeywordsError
        If no entries defining optional or necessary keywords are found in the YAML configuration section.
    """
    possible_entries = (
        "required_entry_sets",
        "required_entries",
        "optional_entry_sets",
        "optional_entries",
    )
    if not any(entry in io_section for entry in possible_entries):
        raise NoKeywordsError("No entries defining optional or necessary keywords found.")

    required_kws = io_section.get("required_entries", [])
    optional_kws = io_section.get("optional_entries", [])

    required, optional = cif_entries_from_entry_set(io_section.get("required_entry_sets", []), entry_sets)
    required_kws += required
    optional_kws += optional

    # for optional entry sets required entries are optional as well
    required, optional = cif_entries_from_entry_set(io_section.get("optional_entry_sets", []), entry_sets)
    optional_kws += required
    optional_kws += optional

    optional_kws = (kw for kw in optional_kws if kw not in required_kws)
    custom_categories = io_section.get("custom_categories", [])

    return (list(set(entries)) for entries in (required_kws, optional_kws, custom_categories))


YmlCifInputSettings = namedtuple(
    "YmlInputSettings", ["required_entries", "optional_entries", "custom_categories", "merge_su"]
)
YmlCifInputSettings.__doc__ = """
Named tuple for storing input settings from a YAML configuration.

Attributes
----------
required_entries : list
    A list of required CIF entries, Functions will throw exceptions if any of these entries is missing.
optional_entries : list
    A list of optional CIF entries, which will be included if present but do not raise an error if missing.
custom_categories : list
    A list of custom categories for keyword conversion, if applicable.
merge_su : bool
    A boolean indicating whether standard uncertainties should be merged.
"""


def cif_input_entries_from_yml(yml_dict: Dict[str, Any], command: str) -> YmlCifInputSettings:
    """
    Extracts required and optional cif_entries for a given command from a YAML configuration
    dictionary.

    Parses the YAML configuration dictionary to find and compile lists of required and optional
    CIF entries specified under a given command. This includes directly specified keywords and those
    included in keyword sets referenced by the command.

    Parameters
    ----------
    yml_dict : dict
        The dictionary obtained from parsing the YAML configuration file.
    command : str
        The command name to look up in the YAML dictionary for extracting CIF entry specifications.

    Returns
    -------
    YmlCifInputSettings
        A named tuple containing required and optional CIF entries, custom categories, and a boolean
        indicating whether standard uncertainties should be merged.
    """
    entry_sets = cif_entry_sets_from_yml(yml_dict)
    command_dict = command_dict_from_yml(yml_dict, command)
    try:
        input_section = command_dict["cif_input"]
    except KeyError as exc:
        raise MissingSectionError(f"No cif_input section found in command {command}.") from exc

    possible_entries = (
        "required_entry_sets",
        "required_entries",
        "optional_entry_sets",
        "optional_entries",
        "custom_categories",
        "merge_su",
    )
    invalid_entries = [entry for entry in input_section if entry not in possible_entries]

    if len(invalid_entries) > 0:
        entry_strings = []
        for entry in invalid_entries:
            close_matches = get_close_matches(entry, possible_entries, n=3)
            if len(close_matches) > 0:
                entry_strings.append(f"Invalid entry is '{entry}'. Did you mean '{close_matches[0]}'?")
            else:
                entry_strings.append(f"Invalid entry is '{entry}'.")
        raise InvalidSectionEntryError(
            f"Found one or more invalid setting entries in cif_input section of command {command}:\n"
            + "\n".join(entry_strings)
        )

    try:
        required_entries, optional_entries, custom_categories = cif_entries_from_yml_section(input_section, entry_sets)
    except NoKeywordsError as exc:
        raise NoKeywordsError(
            f"Command {command} has no entries defining optional or necessary keywords"
            + " in section cif_input of the yml_dict."
        ) from exc

    merge_su = command_dict["cif_input"].get("merge_su", False)

    return YmlCifInputSettings(required_entries, optional_entries, custom_categories, merge_su)


YmlCifOutputSettings = namedtuple(
    "YmlOutputSettings",
    ["required_entries", "optional_entries", "invalidated_entries", "custom_categories", "select_block"],
)
YmlCifOutputSettings.__doc__ = """
Named tuple for storing output settings from a YAML configuration.

Attributes
----------
required_entries : list
    A list of required CIF entries, Functions will throw exceptions if any of these entries is missing.
optional_entries : list
    A list of optional CIF entries, which will be included if present but do not raise an error if missing.
invalidated_entries : list
    A list of invalidated CIF entries, which will be removed from the CIF content of the original input cif.
custom_categories : list
    A list of custom categories for keyword conversion, if applicable.
select_block : str
    The block number to select from the new/work CIF file. Default is '0', so the first block. Plain strings
    are also accepted to select a block by name.
"""


def cif_output_entries_from_yml(yml_dict: Dict[str, Any], command: str) -> YmlCifOutputSettings:
    """
    Extracts required and optional cif_entries for a given command from a YAML configuration
    dictionary.

    Parses the YAML configuration dictionary to find and compile lists of required and optional
    CIF entries specified under a given command. This includes directly specified keywords and those
    included in keyword sets referenced by the command.

    Parameters
    ----------
    yml_dict : dict
        The dictionary obtained from parsing the YAML configuration file.
    command : str
        The command name to look up in the YAML dictionary for extracting CIF entry specifications.

    Returns
    -------
    YmlCifOutputSettings
        A named tuple containing required, optional and invalidated CIF entries, custom categories,
        and a block number or string to select from the new/work CIF file.
    """
    entry_sets = cif_entry_sets_from_yml(yml_dict)
    command_dict = command_dict_from_yml(yml_dict, command)
    try:
        output_section = command_dict["cif_output"]
    except KeyError as exc:
        raise MissingSectionError(f"No cif_output section found in command {command}.") from exc

    possible_entries = (
        "required_entry_sets",
        "required_entries",
        "optional_entry_sets",
        "optional_entries",
        "custom_categories",
        "invalidated_entry_sets",
        "invalidated_entries",
        "select_block",
    )
    invalid_entries = [entry for entry in output_section if entry not in possible_entries]

    if len(invalid_entries) > 0:
        entry_strings = []
        for entry in invalid_entries:
            close_matches = get_close_matches(entry, possible_entries, n=3)
            if len(close_matches) > 0:
                entry_strings.append(f"Invalid entry is '{entry}'. Did you mean '{close_matches[0]}'?")
            else:
                entry_strings.append(f"Invalid entry is '{entry}'.")
        raise InvalidSectionEntryError(
            f"Found one or more invalid setting entries in cif_output section of command {command}:\n"
            + "\n".join(entry_strings)
        )

    try:
        required_entries, optional_entries, custom_categories = cif_entries_from_yml_section(output_section, entry_sets)
    except NoKeywordsError as exc:
        raise NoKeywordsError(
            f"Command {command} has no entries defining optional or necessary keywords"
            + " in section cif_output of the yml_dict."
        ) from exc

    invalidated_kws = output_section.get("invalidated_entries", [])
    required, optional = cif_entries_from_entry_set(output_section.get("invalidated_entry_sets", []), entry_sets)
    invalidated_kws += required
    invalidated_kws += optional
    invalidated_kws = list(set(invalidated_kws))

    select_block = output_section.get("select_block", "0")

    return YmlCifOutputSettings(required_entries, optional_entries, invalidated_kws, custom_categories, select_block)


def cif_file_to_specific_by_yml(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    yml_path: Union[str, Path],
    command: str,
) -> None:
    """
    Processes a CIF file based on instructions defined in a YAML configuration, applying
    specified keyword transformations defined in a commands "cif_input" section and standard
    uncertainty merge settings.

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

    Notes
    -----
    This file was developed for exposing commands within QCrBox. See this project or the
    test of this function for an example of how such a yml file might look like.
    """
    with open(yml_path, "r", encoding="UTF-8") as fobj:
        yml_dict = yaml.safe_load(fobj)

    yml_input_settings = cif_input_entries_from_yml(yml_dict, command)

    cif_file_to_specific(
        input_cif_path,
        output_cif_path,
        yml_input_settings.required_entries,
        yml_input_settings.optional_entries,
        yml_input_settings.custom_categories,
        yml_input_settings.merge_su,
    )


def cif_file_to_unified_by_yml(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    yml_path: Union[str, Path],
    command: str,
) -> None:
    """
    Processes a CIF file based on instructions defined in a YAML configuration, reducing
    the CIF content to the unified equivalents of the entries defined in a commands
    "cif_output" section.

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
    cif_model = read_cif_safe(input_cif_path)

    with open(yml_path, "r", encoding="UTF-8") as fobj:
        yml_dict = yaml.safe_load(fobj)

    yml_output_settings = cif_output_entries_from_yml(yml_dict, command)

    all_entries = yml_output_settings.required_entries + yml_output_settings.optional_entries

    new_cif = trim_cif(cif_model, keep_only_regexes=all_entries, delete_regexes=[], delete_empty_entries=True)

    entries_in_cif = []
    for block in new_cif.values():
        entries_in_cif += list(block.keys())

    missing_entries = set(yml_output_settings.required_entries) - set(entries_in_cif)

    if len(missing_entries) > 0:
        raise ValueError(f"Required entries missing in loaded CIF file: {missing_entries}")

    unified_entries_cif = cif_to_unified_keywords(new_cif, yml_output_settings.custom_categories)
    output_cif = split_su_cif(unified_entries_cif)

    Path(output_cif_path).write_text(str(output_cif), encoding="UTF-8")
