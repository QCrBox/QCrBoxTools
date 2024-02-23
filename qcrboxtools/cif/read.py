# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from typing import List, Union, Optional, Tuple

from iotbx import cif
import yaml

from .uncertainties import split_su_block, split_su_cif, merge_su_cif
from .entries import block_to_unified_keywords, cif_to_unified_keywords, cif_to_requested_keywords

def read_cif_safe(cif_path: Union[str, Path]) -> cif.model.cif:
    """
    Reads a CIF file and returns its content as a CIF model, supporting both string
    paths and pathlib.Path objects.

    Parameters
    ----------
    cif_path : Union[str, Path]
        The path to the CIF file to be read.

    Returns
    -------
    cif.model.cif
        The CIF model parsed from the given file.
    """
    cif_path = Path(cif_path)

    return cif.reader(input_string=cif_path.read_text(encoding='UTF-8')).model()


def cifdata_str_or_index(
    model: cif.model.cif,
    dataset: Union[int, str]
) -> Tuple[cif.model.block, str]:
    """
    Retrieve a CIF dataset block from the model using an index or identifier.

    Parameters
    ----------
    model : iotbx.cif.model.cif
        The CIF model containing datasets.
    dataset : Union[int, str]
        Index or string identifier for the dataset.

    Returns
    -------
    Tuple[cif.model.block, str]
        The CIF dataset block and its identifier.
    """
    if dataset in model.keys():
        return model[dataset], dataset
    try:
        dataset_index = int(dataset)
        keys = list(model.keys())
        dataset = keys[dataset_index]
        return model[dataset], dataset
    except ValueError as exc:
        raise ValueError(
            'Dataset does not exist in cif and cannot be cast into int as index. '
            + f'Got: {dataset}'
        ) from exc
    except IndexError as exc:
        raise IndexError(
            'The given dataset does not exists and integer index is out of range. '
            + f'Got: {dataset}'
        ) from exc

def read_cif_as_unified(
    cif_path: Union[str, Path],
    dataset: Optional[str] = None,
    convert_keywords: bool = True,
    custom_categories: Optional[List[str]] = None,
    split_sus: bool = True
) -> Union[cif.model.block, cif.model.cif]:
    """
    Read a CIF file, optionally process it for unified keywords and split standard
    uncertainties, and return the CIF model or a specific block.

    Parameters
    ----------
    cif_path : Union[str, Path]
        The path to the CIF file.
    dataset : Optional[str], optional
        The identifier for a specific dataset block within the CIF file. If not provided,
        the entire CIF model is processed and returned.
    convert_keywords : bool, default True
        Whether to convert CIF data item names to a unified naming scheme.
    custom_categories : Optional[List[str]], optional
        User defined categories (e.g. 'iucr', 'olex2' or 'shelx') that can be taken
        into account for entry name conversion, where an entry _category_example would
        look up _category.example in a block of the provided cif. Only used if
        `convert_keywords` is True.
    split_sus : bool, default True
        Whether to split values and their standard uncertainties (if present) into separate entries.

    Returns
    -------
    Union[cif.model.block, cif.model.cif]
        Depending on the input, either a single CIF block or the entire CIF model,
        processed according to the specified parameters.
    """
    cif_model = read_cif_safe(cif_path)
    if dataset is not None:
        block, _ = cifdata_str_or_index(cif_model, dataset)
        if convert_keywords:
            block = block_to_unified_keywords(block, custom_categories)
        if split_sus:
            block = split_su_block(block)
        return block
    if convert_keywords:
        cif_model = cif_to_unified_keywords(cif_model, custom_categories)
    if split_sus:
        cif_model = split_su_cif(cif_model)
    return cif_model

def cif_file_unify_split(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    convert_keywords: bool = True,
    custom_categories: Optional[List[str]] = None,
    split_sus: bool = True
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
        split_sus=split_sus
    )

    # Write the modified CIF model to the specified output file.
    Path(output_cif_path).write_text(str(cif_model), encoding='UTF-8')


def cif_file_unified_to_keywords_merge_su(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    compulsory_entries: List[str] = None,
    optional_entries: List[str] = None,
    custom_categories: List[str] = None,
    merge_sus: bool = False
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
        custom categories.
    optional_entries : List[str], optional
        Entries within this list are declared to be optional and therefore will not raise
        an error when not found in the CIF. They still need to be in the requested entries
        to enable their output in a specific position within the newly generated CIF.
    custom_categories : List[str], optional
        User-defined categories (e.g., 'iucr', 'olex2', or 'shelx') that can be taken
        into account where an entry "_category.example" would be cpnverted to "_category_example"
        in a block of the provided CIF, facilitating reversions based on these categories.
    merge_sus : bool, default=False
        If True, numerical values and their standard uncertainties in the CIF model are
        merged before any other processing. If False, the CIF model is processed without
        merging SUs.

    Returns
    -------
    None
        The function writes directly to the file specified by `output_cif_path`.
    """
    cif_model = read_cif_safe(input_cif_path)
    if merge_sus:
        cif_model = merge_su_cif(cif_model)

    if compulsory_entries is not None or optional_entries is not None:
        if compulsory_entries is None:
            compulsory_entries = []
        if optional_entries is None:
            optional_entries = []
        if custom_categories is None:
            custom_categories = []
        cif_model = cif_to_requested_keywords(
            cif_model, compulsory_entries, optional_entries, custom_categories
        )

    Path(output_cif_path).write_text(str(cif_model), encoding='UTF-8')

class NoKeywordsError(BaseException):
    """
    Exception raised when no keywords are found for a given command in the YAML configuration.

    Attributes
    ----------
    message : str
        Explanation of the error
    """

def keywords_from_yml(yml_dict, command):
    """
    Extracts compulsory and optional keywords for a given command from a YAML configuration dictionary.

    Parses the YAML configuration dictionary to find and compile lists of compulsory and optional
    keywords specified under a given command. This includes directly specified keywords and those
    included in keyword sets referenced by the command.

    Parameters
    ----------
    yml_dict : dict
        The dictionary obtained from parsing the YAML configuration file.
    command : str
        The command name to look up in the YAML dictionary for extracting keyword specifications.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple containing two lists: the first list contains compulsory keywords, and the second
        list contains optional keywords. Both lists are de-duplicated.

    Raises
    ------
    KeyError
        If the specified command or referenced keyword sets are not found in the YAML dictionary.
    NameError
        If the keyword set contains entries other than 'required' or 'optional', indicating a possible typo.

    Examples
    --------
    >>> yml_dict = {
    ...     "commands": {
    ...         "process_cif": {
    ...             "required_keywords": ["_cell_length_a"],
    ...             "optional_keywords": ["_atom_site_label"]
    ...         }
    ...     }
    ... }
    >>> keywords_from_yml(yml_dict, "process_cif")
    (['_cell_length_a'], ['_atom_site_label'])
    """
    options = yml_dict['commands'][command]
    possible_entries = (
        'required_keyword_sets', 'required_keywords', 'optional_keyword_sets', 'optional_keywords'
    )
    if not any(entry in options for entry in possible_entries):
        raise NoKeywordsError(
            'Command {command} has no entries defining optional or necessary keywords.'
        )
    compulsory_kws = options.get('required_keywords', [])
    optional_kws = options.get('optional_keywords', [])
    for kwset in options.get('required_keyword_sets', []):
        try:
            kwset_dict = yml_dict['keyword_sets'][kwset]
        except KeyError as exc:
            raise KeyError(f'Keyword set {kwset} not found.') from exc
        if any(key not in ('required', 'optional') for key in kwset_dict):
            raise NameError(
                f'Found entry other than "required" or "optional" in keyword set {kwset}. Typo?'
            )
        compulsory_kws += kwset_dict.get('required', [])
        optional_kws += kwset_dict.get('optional', [])

    for kwset in options.get('optional_keyword_sets', []):
        try:
            kwset_dict = yml_dict['keyword_sets'][kwset]
        except KeyError as exc:
            raise KeyError(f'Keyword set {kwset} not found.') from exc
        if any(key not in ('required', 'optional') for key in kwset_dict):
            raise NameError(
                f'Found entry other than "required" or "optional" in keyword set {kwset}. Typo?'
            )
        optional_kws += kwset_dict.get('required', [])
        optional_kws += kwset_dict.get('optional', [])
    compulsory_kws = list(set(compulsory_kws))
    optional_kws = list(set(kw for kw in optional_kws if kw not in compulsory_kws))
    return compulsory_kws, optional_kws

def cif_file_unified_yml_instr(
    input_cif_path: Union[str, Path],
    output_cif_path: Union[str, Path],
    yml_path: Union[str, Path],
    command: str
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
    with open(yml_path, 'r', encoding='UTF-8') as fobj:
        yml_dict = yaml.safe_load(fobj)

    try:
        options = yml_dict['commands'][command]
    except KeyError as exc:
        raise KeyError(f'Command {command} not found in {yml_path}.') from exc

    compulsory_entries, optional_entries = keywords_from_yml(yml_dict, command)
    merge_sus = options['merge_su']
    custom_categories = options['custom_cif_categories']

    cif_file_unified_to_keywords_merge_su(
        input_cif_path,
        output_cif_path,
        compulsory_entries,
        optional_entries,
        custom_categories,
        merge_sus
    )
