# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import re
import subprocess
from itertools import product
from pathlib import Path
from textwrap import dedent

import pytest

from qcrboxtools.cif.cif2cif import (
    NoKeywordsError,
    cif_entries_from_yml,
    cif_file_to_specific,
    cif_file_to_specific_by_yml,
    cif_file_to_unified,
)


@pytest.fixture(name="test_cif_file_merged")
def fixture_test_cif_file_merged(tmp_path):
    """Create a temporary CIF file for testing."""
    cif_content = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        loop_
        _test_loop_id
        _test_loop_value_with_su
        _test_loop_value_without_su
        1 2.34(5) 7.89
        2 3.45(6) 8.90
        """)
    cif_file = tmp_path / "test_data.cif"
    cif_file.write_text(cif_content)
    return cif_file


def test_cif_file_to_unified(test_cif_file_merged, tmp_path):
    """
    Test the cif_file_unify_split function to ensure it correctly processes
    and writes a CIF file according to the specified parameters.
    """
    # Define the output file path
    output_cif_path = tmp_path / "output_test_data.cif"

    # Call the function under test with split SUs and without converting keywords
    cif_file_to_unified(
        input_cif_path=test_cif_file_merged,
        output_cif_path=output_cif_path,
        convert_keywords=False,
        split_sus=True,
    )

    # Read back the output file and verify its content
    output_cif_content = output_cif_path.read_text()

    # Expected content checks
    expected_lines = [
        "data_test",
        r"_test_value_with_su\s+1\.23",
        r"_test_value_with_su_su\s+0\.04",
        r"_test_value_without_su\s+5\.67",
        "loop_",
        "_test_loop_id",
        "_test_loop_value_with_su",
        "_test_loop_value_with_su_su",
        "_test_loop_value_without_su",
        r"\s*1\s+2\.34\s+0\.05\s+7\.89",
        r"\s*2\s+3\.45\s+0\.06\s+8\.90",
    ]

    for line in expected_lines:
        assert re.search(line, output_cif_content) is not None, f"Expected line not found: {line}"


@pytest.fixture(name="test_cif_file_unmerged")
def fixture_test_cif_file_unmerged(tmp_path) -> Path:
    """
    Creates a temporary CIF file with pre-defined content for testing.

    Returns
    -------
    Path
        The path to the temporary CIF file.
    """
    cif_content = dedent("""
    data_test
    _custom.test 'something'
    _cell.length_a 10.0
    _cell.length_a_su 0.03
    _cell.length_b 20.0
    _cell.length_b_su 0.02
    loop_
      _atom_site.label
      _atom_site.fract_x
      _atom_site.fract_x_su
      _atom_site.fract_y
      _atom_site.fract_y_su
      _atom_site.fract_z
      _atom_site.fract_z_su
        C1 0.234 0.012 0.567 0.045 0.890 0.078
        O1 -0.345 0.023 0.678 0.0 -0.901 0.089
    """)
    cif_file = tmp_path / "test.cif"
    cif_file.write_text(cif_content, encoding="UTF-8")
    return cif_file


def test_cif_file_to_specific(test_cif_file_unmerged, tmp_path):
    """
    Test the cif_file_unified_to_keywords_merge_su function to ensure it processes the CIF file
    as expected, merging SUs and filtering entries according to specified criteria.
    """
    output_cif_path = tmp_path / "output.cif"

    # Define compulsory and optional entries for the test
    kwdict = {
        "compulsory_entries": ["_cell_length_a"],
        "optional_entries": [
            "_cell_length_b",
            "_cell_length_b_su",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_custom_test",
        ],
        "custom_categories": ["custom"],
    }

    patterns_from_kwdict = {
        "compulsory_entries": [r"_cell_length_a\s+10.00\(3\)"],
        "optional_entries": [
            r"_cell_length_b\s+20.0",
            "_cell_length_b_su",  # cell_length_b_su is requested as entry and should not be merged
            "_atom_site_fract_x",
            "_atom_site_fract_y",
        ],
        "custom_categories": [r"_custom_test\s+something"],
    }

    # Call the function with merge_sus enabled
    for include_compulsory, include_optional, include_custom in product([True, False], repeat=3):
        included_kws = {}
        check_patterns = []
        anti_check_patterns = []
        if include_compulsory:
            included_kws["compulsory_entries"] = kwdict["compulsory_entries"]
            check_patterns.extend(patterns_from_kwdict["compulsory_entries"])
        else:
            anti_check_patterns.extend(patterns_from_kwdict["compulsory_entries"])

        if include_optional:
            included_kws["optional_entries"] = kwdict["optional_entries"]
            check_patterns.extend(patterns_from_kwdict["optional_entries"])
        else:
            anti_check_patterns.extend(patterns_from_kwdict["optional_entries"])

        if include_custom and include_optional:
            included_kws["custom_categories"] = kwdict["custom_categories"]
            check_patterns.extend(patterns_from_kwdict["custom_categories"])
        else:
            anti_check_patterns.extend(patterns_from_kwdict["custom_categories"])

        cif_file_to_specific(
            input_cif_path=test_cif_file_unmerged,
            output_cif_path=output_cif_path,
            merge_sus=True,
            **included_kws,
        )

        # Read the output CIF content
        output_content = output_cif_path.read_text(encoding="UTF-8")

        for pattern in check_patterns:
            assert re.search(pattern, output_content) is not None, f"Expected pattern not found: {pattern}"
        for pattern in anti_check_patterns:
            assert re.search(pattern, output_content) is None, f"Unexpected pattern found: {pattern}"
        # assert "_atom_site_fract_z" not in output_content, "Included _atom_site.fract_z entry unexpectedly"


def test_cif_file_to_specific_all_unified_su(test_cif_file_unmerged, tmp_path):
    """
    Test the cif_file_unified_to_keywords_merge_su function to ensure it processes the CIF file
    as expected, merging SUs and filtering entries according to specified criteria.
    """
    output_cif_path = tmp_path / "output.cif"

    # write an entry that is not unified at the moment
    with open(test_cif_file_unmerged, "a", encoding='UTF-8') as fobj:
        fobj.write("\n_custom_test2  'something else'\n")

    # Define compulsory and optional entries for the test
    compulsory_entries = ["_cell_length_a", "_cell.length_b_su"]
    optional_entries = ["all_unified"]
    custom_categories = ["custom"]

    # Call the function with merge_sus enabled
    cif_file_to_specific(
        input_cif_path=test_cif_file_unmerged,
        output_cif_path=output_cif_path,
        compulsory_entries=compulsory_entries,
        optional_entries=optional_entries,
        custom_categories=custom_categories,
        merge_sus=True,
    )

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding="UTF-8")
    search_patterns = (
        r"_cell\.length_a\s+10.00\(3\)",
        r"_cell\.length_b\s+20.0",
        r"_cell\.length_b_su",  # cell_length_b_su is requested as entry and should not be merged
        r"_atom_site\.fract_x",
        r"_atom_site\.fract_y",
        r"_atom_site\.fract_z",  # also include as all_unified in keywords
        r"_custom\.test2",  # all unified should output unified keywords -> convert via category
    )
    for pattern in search_patterns:
        assert re.search(pattern, output_content) is not None
    assert "_cell_length_a" not in output_content, "Renaming should be skipped entirely"

@pytest.mark.parametrize("input_or_output", ["input", "output"])
def test_direct_cif_entries_extraction(input_or_output):
    """Test extraction of directly defined keywords."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                f"cif_{input_or_output}": {
                    "required_cif_entries": ["_cell_length_a", "_cell_length_b"],
                    "optional_cif_entries": ["_atom_site.label"],
                }
            }
        ]
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif", input_or_output)
    assert sorted(compulsory) == sorted(["_cell_length_a", "_cell_length_b"]), "Failed to extract compulsory keywords"
    assert sorted(optional) == sorted(["_atom_site.label"]), "Failed to extract optional keywords"

@pytest.mark.parametrize("input_or_output", ["input", "output"])
def test_cif_entries_io_section_missing(input_or_output):
    """Test that an error is raised if the input or output section is missing."""
    yml_dict = {"commands": [{"name": "process_cif"}]}
    with pytest.raises(KeyError):
        cif_entries_from_yml(yml_dict, "process_cif", input_or_output)

def test_cif_entries_io_invalid():
    """Test that an error is raised if the input or output section is invalid."""
    yml_dict = {}
    with pytest.raises(ValueError):
        cif_entries_from_yml(yml_dict, "process_cif", "notavailable")

def test_cif_entries_extraction_via_sets():
    """Test extraction of keywords defined through keyword sets."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "cif_input": {
                    "required_cif_entry_sets": ["cell_dimensions"],
                    "optional_cif_entry_sets": ["atom_sites"],
                }
            }
        ],
        "cif_entry_sets": [
            {
                "name": "cell_dimensions",
                "required": ["_cell_length_a", "_cell_length_b"],
                "optional": [],  # Example with an empty list
            },
            {
                "name": "atom_sites",
                "required": [],
                "optional": ["_atom_site.label", "_atom_site.occupancy"],
            },
        ],
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif", "input")
    assert set(compulsory) == {
        "_cell_length_a",
        "_cell_length_b",
    }, "Failed to extract compulsory keywords from sets"
    assert set(optional) == {
        "_atom_site.label",
        "_atom_site.occupancy",
    }, "Failed to extract optional keywords from sets"


@pytest.mark.parametrize(
    "missing_key, tested_set",
    [
        ("process_cif", "required_cif_entry_sets"),
        ("nonexistent_set", "required_cif_entry_sets"),
        ("process_cif", "optional_cif_entry_sets"),
        ("nonexistent_set", "optional_cif_entry_sets"),
    ],
)
def test_errors_for_missing_command_or_set(missing_key, tested_set):
    """Test that the correct errors are raised for missing commands or keyword sets."""
    yml_dict = {"commands": [{"name": "nonexistent_set", "cif_input": {tested_set: ["missing"]}}]}
    with pytest.raises(KeyError):
        cif_entries_from_yml(yml_dict, missing_key, input_or_output="input")


def test_no_entries_in_command():
    # test command has no sets
    yml_dict = {"commands": [{"name": "no_entries", "cif_input": {}}]}
    with pytest.raises(NoKeywordsError):
        cif_entries_from_yml(yml_dict, "no_entries", "input")


@pytest.mark.parametrize("tested_set", ["required_cif_entry_sets", "optional_cif_entry_sets"])
def test_incorrect_entry_in_cif_entry_set(tested_set):
    """Test detection of incorrect entries within keyword sets."""
    yml_dict = {
        "commands": [{"name": "process_cif", "cif_input": {tested_set: ["incorrect_set"]}}],
        "cif_entry_sets": [
            {
                "name": "incorrect_set",
                "wrong_entry": ["_cell_length_a"],  # Intentionally incorrect to trigger the error
            }
        ],
    }
    with pytest.raises(NameError):
        cif_entries_from_yml(yml_dict, "process_cif", "input")


def test_unique_compulsory_cif_entries_from_multiple_sets():
    """Test that compulsory keywords from multiple sets are appended uniquely."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "cif_input": {
                    "required_cif_entry_sets": ["set1", "set2"],
                }
            }
        ],
        "cif_entry_sets": [
            {"name": "set1", "required": ["_cell_length_a", "_cell_angle_alpha"], "optional": []},
            {"name": "set2", "required": ["_cell_length_a", "_cell_volume"], "optional": []},
        ],
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif", "input")
    assert sorted(compulsory) == sorted(
        list(set(["_cell_length_a", "_cell_angle_alpha", "_cell_volume"]))
    ), "Compulsory keywords should be unique and include all items from both sets"
    assert optional == [], "No optional keywords should be present"


def test_unique_optional_cif_entries_from_multiple_sets():
    """
    Test that optional keywords from multiple sets are appended uniquely and exclude compulsory ones.
    Also test that optional keyword set entries all end up in optional
    """
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "cif_input": {
                    "optional_cif_entry_sets": ["set1", "set2"],
                    "required_cif_entries": ["_cell_length_a"],  # Ensure exclusion of compulsory from optional
                }
            }
        ],
        "cif_entry_sets": [
            {
                "name": "set1",
                "required": [
                    "_cell_length_a",
                    "_cell_angle_alpha",
                ],  # _cell_length_a should be excluded
                "optional": [],
            },
            {"name": "set2", "required": ["_cell_volume"], "optional": ["_cell_length_b"]},
        ],
    }
    compulsory, optional = cif_entries_from_yml(yml_dict, "process_cif", "input")
    assert compulsory == ["_cell_length_a"], "Only specified compulsory keywords should be present"
    # Ensure _cell_length_a is not duplicated in optional, despite being in both sets and compulsory
    assert sorted(optional) == sorted(
        list(set(["_cell_length_b", "_cell_angle_alpha", "_cell_volume"]))
    ), "Optional keywords should be unique and exclude compulsory keywords"


@pytest.fixture(name="mock_yaml_file")
def fixture_mock_yaml_file(tmp_path):
    yml_path = tmp_path / "config.yml"

    # Mock YAML content
    yml_content = dedent("""
    cif_entry_sets :
      - name: test1
        required : [
          _cell_length_a
        ]
        optional : [
          _atom_site_fract_y
        ]
      - name: test2
        required : [
          _invalid_keyword, _atom_site_fract_x
        ]

    commands :
      - name: process_cif
        merge_cif_su: Yes
        cif_input:
          custom_cif_categories: [custom]
          required_cif_entry_sets: [test1]
          optional_cif_entry_sets: [test2]
          required_cif_entries: [_cell_length_b]
          optional_cif_entries: [_atom_site_label]
        cif_output:
          custom_cif_categories: [custom]
          required_cif_entry_sets: [test1]
          optional_cif_entry_sets: [test2]
          optional_cif_entries: [_atom_site_label]
    """)
    yml_path.write_text(yml_content)

    return yml_path


def test_cif_file_to_specific_by_yml(test_cif_file_unmerged, mock_yaml_file, tmp_path):
    output_cif_path = tmp_path / "output.cif"

    cif_file_to_specific_by_yml(
        input_cif_path=test_cif_file_unmerged,
        output_cif_path=output_cif_path,
        yml_path=mock_yaml_file,
        command="process_cif",
    )

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding="UTF-8")
    search_patterns = (
        r"_cell_length_a\s+10.00\(3\)",
        r"_cell_length_b\s+20.00\(2\)",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
    )
    for pattern in search_patterns:
        assert re.search(pattern, output_content) is not None
    assert "_atom_site_fract_z" not in output_content, "Included _atom_site.fract_z entry unexpectedly"


def test_cif_file_to_specific_by_yml_no_command(test_cif_file_unmerged, mock_yaml_file, tmp_path):
    output_cif_path = tmp_path / "output.cif"

    with pytest.raises(KeyError):
        cif_file_to_specific_by_yml(
            input_cif_path=test_cif_file_unmerged,
            output_cif_path=output_cif_path,
            yml_path=mock_yaml_file,
            command="nonexistent_command",
        )


# CLI tests

CLI_COMMAND = ["python", "-m", "qcrboxtools.cif"]


def test_cli_command_keyword(test_cif_file_unmerged, tmp_path):
    command = "to-specific"
    args = ["--compulsory_entries", "_cell_length_a", "--merge_sus"]
    expected_output_patterns = [
        r"_cell_length_a\s+10.00\(3\)",
    ]
    output_cif_path = tmp_path / "output.cif"
    cli_args = CLI_COMMAND + [command, str(test_cif_file_unmerged), str(output_cif_path)] + args

    # Execute the CLI command
    result = subprocess.run(cli_args, capture_output=True, text=True)

    # Ensure the command executed successfully
    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding="UTF-8")

    # Check for expected patterns in the output content
    for pattern in expected_output_patterns:
        assert re.search(pattern, output_content) is not None, f"Expected pattern not found in output: {pattern}"


def test_cli_command_keywords_yml(test_cif_file_unmerged, mock_yaml_file, tmp_path):
    command = "yml"
    args = [str(mock_yaml_file), "process_cif"]
    expected_output_patterns = [
        r"_cell_length_a\s+10.00\(3\)",
        r"_cell_length_b\s+20.00\(2\)",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
    ]
    output_cif_path = tmp_path / "output.cif"
    cli_args = CLI_COMMAND + [command, str(test_cif_file_unmerged), str(output_cif_path)] + args

    # Execute the CLI command
    result = subprocess.run(cli_args, capture_output=True, text=True)

    # Ensure the command executed successfully
    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding="UTF-8")

    # Check for expected patterns in the output content
    for pattern in expected_output_patterns:
        assert re.search(pattern, output_content) is not None, f"Expected pattern not found in output: {pattern}"
    assert "_atom_site_fract_z" not in output_content, "Included _atom_site.fract_z entry unexpectedly"


def test_cli_command_unify(test_cif_file_merged, tmp_path):
    command = "to-unified"
    args = ["--convert_keywords", "--split_sus"]
    expected_output_patterns = [
        r"_test_value_with_su\s+1\.23",
        r"_test_value_with_su_su\s+0\.04",
        r"_test_value_without_su\s+5\.67",
    ]
    output_cif_path = tmp_path / "output.cif"
    cli_args = CLI_COMMAND + [command, str(test_cif_file_merged), str(output_cif_path)] + args

    # Execute the CLI command
    result = subprocess.run(cli_args, capture_output=True, text=True)

    # Ensure the command executed successfully
    assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding="UTF-8")

    # Check for expected patterns in the output content
    for pattern in expected_output_patterns:
        assert re.search(pattern, output_content) is not None, f"Expected pattern not found in output: {pattern}"


def test_cli_non_existent_command(test_cif_file_merged, tmp_path):
    # Run the script with a non-existent command
    command = "nonexistent_command"
    output_cif_path = tmp_path / "output.cif"
    cli_args = CLI_COMMAND + [command, str(test_cif_file_merged), str(output_cif_path)]
    result = subprocess.run(cli_args, capture_output=True, text=True)

    # Check if the help message is in the output
    assert "usage:" in result.stderr
