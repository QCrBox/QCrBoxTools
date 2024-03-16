# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0
"""
CIF Processing Command Line Interface (CLI)

This module provides a command-line interface for converting and processing Crystallographic
Information File (CIF) formats using various operations. It supports converting CIF files
from one keyword convention to another, merging standard uncertainties (SUs) with their
corresponding values, splitting SUs from their values, and filtering entries based on
specified criteria. The operations are based on configurations defined either directly
through command-line arguments or via YAML files for using QCrBox command definitions.

The CLI offers three main commands:
1. keywords_yml: Processes a CIF file according to the specifications in a QCrBox YAML
   configuration file. This command allows for complex processing instructions,
   including keyword conversions and SU mergers, defined in a structured YAML document.
2. keywords: Directly processes a CIF file by merging SUs, filtering by specified keywords
   or one of the keyword aliases and applying custom categories for entry conversion.
   This command is versatile for quick adjustments or specific transformations without
   the need for an external configuration file.
3. unify: Reads, processes, and writes a CIF file with optional modifications such as keyword
   conversion and SU splitting. This command is designed for straightforward CIF modifications,
   providing flexibility with simple command-line arguments.

Usage:
    The CLI is invoked with one of the commands followed by the required arguments. Optional
    flags and parameters allow for customization of the processing behavior. Each command has
    its own set of arguments and options, detailed in the help text accessible via `--help`
    after specifying a command.

Example:
    To process a CIF file using a YAML configuration, one might use:
    `$ python -m qcrboxtools.cif keywords_yml input.cif output.cif config.yml process_command`

    To merge standard uncertainties and filter by specified keywords:
    `$ python -m qcrboxtools.cif keywords input.cif output.cif --compulsory_entries
       _cell_length_a --merge_sus`

    To convert keywords and split SUs in a CIF file:
    `$ python -m qcrboxtools.cif unify input.cif output.cif --convert_keywords --split_sus`
"""

import argparse
from pathlib import Path

from .cif2cif import (
    cif_file_unified_to_keywords_merge_su,
    cif_file_unified_yml_instr,
    cif_file_unify_split,
)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description=("Convert cif files from one keyword convention to another." + " Split/Merge SUs as needed.")
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Parser for cif_file_unified_yml_instr
    parser_unify_yml = subparsers.add_parser(
        "keywords_yml", help="Process a CIF file based on a QCrBox YAML configuration."
    )
    parser_unify_yml.add_argument("input_cif_path", type=Path, help="Input CIF file path.")
    parser_unify_yml.add_argument("output_cif_path", type=Path, help="Output CIF file path.")
    parser_unify_yml.add_argument("yml_path", type=Path, help="YAML configuration file path.")
    parser_unify_yml.add_argument("command", type=str, help="Command within the YAML file for processing.")

    # Parser for cif_file_unified_to_keywords_merge_su
    parser_keywords_merge_su = subparsers.add_parser(
        "keywords",
        help="Processes a CIF file, optionally merges SUs, and filters by specified keywords.",
    )
    parser_keywords_merge_su.add_argument("input_cif_path", type=Path, help="Input CIF file path.")
    parser_keywords_merge_su.add_argument("output_cif_path", type=Path, help="Output CIF file path.")
    parser_keywords_merge_su.add_argument(
        "--compulsory_entries", nargs="*", default=[], help="Compulsory entries to include."
    )
    parser_keywords_merge_su.add_argument(
        "--optional_entries", nargs="*", default=[], help="Optional entries to include if present."
    )
    parser_keywords_merge_su.add_argument(
        "--custom_categories", nargs="*", default=[], help="Custom categories for entry conversion."
    )
    parser_keywords_merge_su.add_argument(
        "--merge_sus",
        action="store_true",
        help="Merge numerical values with their standard uncertainties.",
    )
    parser_keywords_merge_su.set_defaults(merge_sus=False)

    # Parser for cif_file_unify_split
    parser_unify_split = subparsers.add_parser(
        "unify", help="Read, process, and write a CIF file with optional modifications."
    )
    parser_unify_split.add_argument("input_cif_path", type=Path, help="Input CIF file path.")
    parser_unify_split.add_argument("output_cif_path", type=Path, help="Output CIF file path.")
    parser_unify_split.add_argument(
        "--convert_keywords", action="store_true", help="Convert keywords to a unified format."
    )
    parser_unify_split.add_argument(
        "--no_convert_keywords",
        action="store_false",
        dest="convert_keywords",
        help="Do not convert keywords to a unified format.",
    )
    parser_unify_split.set_defaults(convert_keywords=True)
    parser_unify_split.add_argument("--custom_categories", nargs="*", help="Custom categories for keyword conversion.")
    parser_unify_split.add_argument(
        "--split_sus", action="store_true", help="Split values from their SUs in the CIF content."
    )
    parser_unify_split.add_argument(
        "--no_split_sus",
        action="store_false",
        dest="split_sus",
        help="Do not split values from their SUs in the CIF content.",
    )
    parser_unify_split.set_defaults(split_sus=True)

    args = parser.parse_args()

    # Handling function calls based on command
    if args.command == "keywords_yml":
        cif_file_unified_yml_instr(args.input_cif_path, args.output_cif_path, args.yml_path, args.command)
    elif args.command == "unify":
        cif_file_unify_split(
            args.input_cif_path,
            args.output_cif_path,
            args.convert_keywords,
            args.custom_categories,
            args.split_sus,
        )
    elif args.command == "keywords":
        cif_file_unified_to_keywords_merge_su(
            args.input_cif_path,
            args.output_cif_path,
            args.compulsory_entries,
            args.optional_entries,
            args.custom_categories,
            args.merge_sus,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
