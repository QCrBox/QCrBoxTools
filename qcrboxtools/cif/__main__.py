# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
"""
CIF Processing Command Line Interface (CLI)

This module provides a command-line interface for converting and processing Crystallographic
Information Files (CIF). It supports converting CIF files from any keywor convention to
QCrBox's unified keyword set and back, merging standard uncertainties (SUs) with their
corresponding values, splitting SUs from their values, and filtering entries based on
specified criteria. The operations are based on configurations defined either directly
through command-line arguments or via YAML files for comprehensive processing with QCrBox
command definitions.

The CLI offers four main commands:
1. specific_by_yml: Processes a CIF file according to specifications in a QCrBox YAML
   configuration file. It utilizes the cif_input section to convert keywords and merge
   standard uncertainties as specified.
2. unified_by_yml: Processes a CIF file to a unified format based on a QCrBox YAML
   configuration. It trims the CIF to the keywords from cif_output before conversion,
   before converting the remaining entries to unified format, including splitting
   their standart uncertainties.
3. to_specific: Directly processes a CIF file by merging SUs, filtering by specified keywords,
   and applying custom categories for entry conversion. This command only uses command-line
   arguments.
4. to_unified: Reads, processes, and writes a CIF file with optional modifications such as
   keyword conversion and SU splitting. This command only uses command-line arguments.

Usage:
    The CLI is invoked with one of the commands followed by the required arguments. Optional
    flags and parameters allow for customization of the processing behavior. Each command has
    its own set of arguments and options, detailed in the help text accessible via `--help`
    after specifying a command.

Example:
    To process a CIF file using a YAML configuration:
    `$ python -m qcrboxtools.cif specific_by_yml input.cif output.cif config.yml process_command`

    To convert a CIF file to unified format based on YAML configuration:
    `$ python -m qcrboxtools.cif unified_by_yml input.cif output.cif config.yml command_name`

    To merge standard uncertainties and filter by specified keywords:
    `$ python -m qcrboxtools.cif to_specific input.cif output.cif --compulsory_entries
       _cell_length_a --merge_su`

    To convert keywords and split SUs in a CIF file:
    `$ python -m qcrboxtools.cif to_unified input.cif output.cif --convert_keywords --split_sus`
"""

import argparse
from pathlib import Path

from .cif2cif import cif_file_to_specific, cif_file_to_specific_by_yml, cif_file_to_unified, cif_file_to_unified_by_yml


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description=("Convert cif files from one keyword convention to another." + " Split/Merge SUs as needed.")
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Parser for cif_file_unified_yml_instr
    parser_yml = subparsers.add_parser(
        "specific_by_yml",
        help="Process a CIF file based on a QCrBox YAML configuration using the keywords from cif_input",
    )
    parser_yml.add_argument("input_cif_path", type=Path, help="Input CIF file path.")
    parser_yml.add_argument("output_cif_path", type=Path, help="Output CIF file path.")
    parser_yml.add_argument("yml_path", type=Path, help="YAML configuration file path.")
    parser_yml.add_argument("yml_command", type=str, help="Command within the YAML file for processing.")

    # Parser for cif_file_unified_yml_instr
    parser_unified_by_yml = subparsers.add_parser(
        "unified_by_yml",
        help=(
            "Process a CIF file to unified format based on a QCrBox YAML configuration, trimming to the keywords from"
            + "cif_output before conversion"
        ),
    )
    parser_unified_by_yml.add_argument("input_cif_path", type=Path, help="Input CIF file path.")
    parser_unified_by_yml.add_argument("output_cif_path", type=Path, help="Output CIF file path.")
    parser_unified_by_yml.add_argument("yml_path", type=Path, help="YAML configuration file path.")
    parser_unified_by_yml.add_argument("yml_command", type=str, help="Command within the YAML file for processing.")

    # Parser for cif_file_to_specific
    parser_to_specific = subparsers.add_parser(
        "to_specific",
        help="Processes a CIF file, optionally merges SUs, and filters by specified keywords.",
    )
    parser_to_specific.add_argument("input_cif_path", type=Path, help="Input CIF file path.")
    parser_to_specific.add_argument("output_cif_path", type=Path, help="Output CIF file path.")
    parser_to_specific.add_argument(
        "--compulsory_entries", nargs="*", default=[], help="Compulsory entries to include."
    )
    parser_to_specific.add_argument(
        "--optional_entries", nargs="*", default=[], help="Optional entries to include if present."
    )
    parser_to_specific.add_argument(
        "--custom_categories", nargs="*", default=[], help="Custom categories for entry conversion."
    )
    parser_to_specific.add_argument(
        "--merge_su",
        action="store_true",
        help="Merge numerical values with their standard uncertainties.",
    )
    parser_to_specific.set_defaults(merge_su=False)

    # Parser for cif_file_to_unified
    parser_to_unified = subparsers.add_parser(
        "to_unified", help="Read, process, and write a CIF file with optional modifications."
    )
    parser_to_unified.add_argument("input_cif_path", type=Path, help="Input CIF file path.")
    parser_to_unified.add_argument("output_cif_path", type=Path, help="Output CIF file path.")
    parser_to_unified.add_argument(
        "--convert_keywords", action="store_true", help="Convert keywords to a unified format."
    )
    parser_to_unified.add_argument(
        "--no_convert_keywords",
        action="store_false",
        dest="convert_keywords",
        help="Do not convert keywords to a unified format.",
    )
    parser_to_unified.set_defaults(convert_keywords=True)
    parser_to_unified.add_argument("--custom_categories", nargs="*", help="Custom categories for keyword conversion.")
    parser_to_unified.add_argument(
        "--split_sus", action="store_true", help="Split values from their SUs in the CIF content."
    )
    parser_to_unified.add_argument(
        "--no_split_sus",
        action="store_false",
        dest="split_sus",
        help="Do not split values from their SUs in the CIF content.",
    )
    parser_to_unified.set_defaults(split_sus=True)

    args = parser.parse_args()

    # Handling function calls based on command
    if args.command == "specific_by_yml":
        cif_file_to_specific_by_yml(args.input_cif_path, args.output_cif_path, args.yml_path, args.yml_command)
    elif args.command == "unified_by_yml":
        cif_file_to_unified_by_yml(args.input_cif_path, args.output_cif_path, args.yml_path, args.yml_command)
    elif args.command == "to_unified":
        cif_file_to_unified(
            args.input_cif_path,
            args.output_cif_path,
            args.convert_keywords,
            args.custom_categories,
            args.split_sus,
        )
    elif args.command == "to_specific":
        cif_file_to_specific(
            args.input_cif_path,
            args.output_cif_path,
            args.compulsory_entries,
            args.optional_entries,
            args.custom_categories,
            args.merge_su,
        )


if __name__ == "__main__":
    main()
