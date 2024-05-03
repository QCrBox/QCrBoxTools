# Copyright 2024 Paul Niklas Ruth.
# SPDX-License-Identifier: MPL-2.0

import re
import subprocess
from itertools import product
from pathlib import Path
from textwrap import dedent

import pytest
from iotbx.cif import model

from qcrboxtools.cif.cif2cif import (
    EmptyCommandError,
    EmptyParameterError,
    InvalidEntrySetError,
    NoKeywordsError,
    NonExistentEntrySetError,
    OneOfEntryNotResolvableError,
    UnknownCommandError,
    UnknownParameterError,
    UnnamedCommandError,
    UnnamedParameterError,
    cif_entries_from_entry_set,
    cif_entries_from_parameter_dict,
    cif_entry_sets_from_yml,
    cif_file_merge_to_unified_by_yml,
    cif_file_to_specific,
    cif_file_to_specific_by_yml,
    cif_file_to_unified,
    cif_input_entries_from_yml,
    cif_output_entries_from_yml,
    command_parameter_dict_from_yml,
    resolve_special_entries,
    yml_entries_resolve_special,
    YmlCifInputSettings,
    YmlCifOutputSettings,
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

    # Define required and optional entries for the test
    kwdict = {
        "required_entries": ["_cell_length_a"],
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
        "required_entries": [r"_cell_length_a\s+10.00\(3\)"],
        "optional_entries": [
            r"_cell_length_b\s+20.0",
            "_cell_length_b_su",  # cell_length_b_su is requested as entry and should not be merged
            "_atom_site_fract_x",
            "_atom_site_fract_y",
        ],
        "custom_categories": [r"_custom_test\s+something"],
    }

    # Call the function with merge_su enabled
    for include_required, include_optional, include_custom in product([True, False], repeat=3):
        included_kws = {}
        check_patterns = []
        anti_check_patterns = []
        if include_required:
            included_kws["required_entries"] = kwdict["required_entries"]
            check_patterns.extend(patterns_from_kwdict["required_entries"])
        else:
            anti_check_patterns.extend(patterns_from_kwdict["required_entries"])

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
            merge_su=True,
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
    with open(test_cif_file_unmerged, "a", encoding="UTF-8") as fobj:
        fobj.write("\n_custom_test2  'something else'\n")

    # Define required and optional entries for the test
    required_entries = ["_cell_length_a", "_cell.length_b_su"]
    optional_entries = ["all_unified"]
    custom_categories = ["custom"]

    # Call the function with merge_su enabled
    cif_file_to_specific(
        input_cif_path=test_cif_file_unmerged,
        output_cif_path=output_cif_path,
        required_entries=required_entries,
        optional_entries=optional_entries,
        custom_categories=custom_categories,
        merge_su=True,
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


def test_command_parameter_dict_from_yml():
    """Test extraction of command dictionary from a YAML file."""
    yml_dict = {
        "commands": [
            {
                "name": "not_this_one",
                "parameters": [
                    {
                        "name": "test_parameter",
                        "required_entries": ["_cell_length_c"],
                        "optional_entries": ["_cell_angle_alpha"],
                    },
                ],
            },
            {
                "name": "process_cif",
                "parameters": [
                    {
                        "name": "test_parameter",
                        "required_entries": ["_cell_length_a"],
                        "optional_entries": ["_cell_length_b"],
                    },
                ],
            },
        ]
    }
    parameter_dict = command_parameter_dict_from_yml(yml_dict, "process_cif", "test_parameter")
    assert parameter_dict == {
        "required_entries": ["_cell_length_a"],
        "optional_entries": ["_cell_length_b"],
    }, "Failed to extract command dictionary"

    assert "name" in yml_dict["commands"][1]["parameters"][0], "Name was deleted from yml dictionary"


def test_command_parameter_dict_from_yml_unkown_command():
    """Test that an error is raised if an unknown command is requested."""
    yml_dict = {
        "commands": [
            {
                "name": "test_entry",
            },
        ]
    }
    with pytest.raises(UnknownCommandError):
        command_parameter_dict_from_yml(yml_dict, "unknown_command", "something")


def test_command_parameter_dict_invalid_command():
    """Test that an error is raised if a command is missing a name."""
    yml_dict = {
        "commands": [
            {},
        ]
    }
    with pytest.raises(UnnamedCommandError):
        command_parameter_dict_from_yml(yml_dict, "process_cif", "something")


def test_command_parameter_dict_from_yml_empty_command():
    """Test that an error is raised if a command is empty."""
    yml_dict = {
        "commands": [
            {
                "name": "empty_command",
            },
        ]
    }
    with pytest.raises(EmptyCommandError):
        command_parameter_dict_from_yml(yml_dict, "empty_command", "something")


def test_command_parameter_dict_from_yml_unknown_parameter():
    """Test that an error is raised if an unknown parameter is requested."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "parameters": [
                    {
                        "name": "test_parameter",
                        "required_entries": ["_cell_length_a"],
                        "optional_entries": ["_cell_length_b"],
                    },
                ],
            },
        ]
    }
    with pytest.raises(UnknownParameterError):
        command_parameter_dict_from_yml(yml_dict, "process_cif", "unknown_parameter")


def test_command_parameter_dict_from_yml_invalid_parameter():
    """Test that an error is raised if a parameter is missing a name."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "parameters": [
                    {
                        "required_entries": ["_cell_length_a"],
                        "optional_entries": ["_cell_length_b"],
                    },
                ],
            },
        ]
    }
    with pytest.raises(UnnamedParameterError):
        command_parameter_dict_from_yml(yml_dict, "process_cif", "something")


def test_command_parameter_dict_from_yml_empty_parameter():
    """Test that an error is raised if a parameter is empty."""
    yml_dict = {
        "commands": [
            {
                "name": "empty_command",
                "parameters": [
                    {"name": "empty_parameter"},
                ],
            },
        ]
    }
    with pytest.raises(EmptyParameterError):
        command_parameter_dict_from_yml(yml_dict, "empty_command", "empty_parameter")


def test_cif_entry_sets_from_yml():
    """Test extraction of CIF entry sets from a YAML file."""
    yml_dict = {
        "cif_entry_sets": [
            {
                "name": "test_set",
                "required": ["_cell_length_a", "_cell_length_b"],
                "optional": ["_atom_site_fract_x", "_atom_site_fract_y"],
            }
        ]
    }
    entry_sets = cif_entry_sets_from_yml(yml_dict)
    assert entry_sets == {
        "test_set": {
            "required": ["_cell_length_a", "_cell_length_b"],
            "optional": ["_atom_site_fract_x", "_atom_site_fract_y"],
        }
    }, "Failed to extract CIF entry sets"

    assert "name" in yml_dict["cif_entry_sets"][0], "Name was deleted from yml dictionary"


def test_cif_entry_sets_from_yml_exceptions():
    """
    Test that an error is raised if a CIF entry set is missing a name or has an invalid
    entry.
    """
    yml_dict = {
        "cif_entry_sets": [
            {
                "required": ["_cell_length_a", "_cell_length_b"],
                "optional": ["_atom_site_fract_x", "_atom_site_fract_y"],
            }
        ]
    }
    with pytest.raises(InvalidEntrySetError):
        cif_entry_sets_from_yml(yml_dict)

    yml_dict = {
        "cif_entry_sets": [
            {
                "name": "test_set",
                "required": ["_cell_length_a", "_cell_length_b"],
                "optional": ["_atom_site_fract_x", "_atom_site_fract_y"],
                "typo_entry": ["_cell_length_c"],
            }
        ]
    }
    with pytest.raises(InvalidEntrySetError):
        cif_entry_sets_from_yml(yml_dict)


def test_cif_entries_from_entry_set():
    """Test extraction of CIF entries from an entry set."""
    entry_sets = {
        "test_set1": {
            "required": ["_cell_length_a", "_cell_length_b"],
            "optional": ["_atom_site_fract_x", "_atom_site_fract_y"],
        },
        "test_set2": {
            "required": ["_cell_length_c"],
            "optional": ["_atom_site_fract_z"],
        },
        "non_included_set": {
            "required": ["_cell_angle_alpha"],
            "optional": ["_cell_volume"],
        },
    }
    required, optional = cif_entries_from_entry_set(["test_set1", "test_set2"], entry_sets)
    required_correct = ["_cell_length_a", "_cell_length_b", "_cell_length_c"]
    optional_correct = ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    assert sorted(required) == sorted(required_correct), "Failed to extract required entries"
    assert sorted(optional) == sorted(optional_correct), "Failed to extract optional entries"


def test_cif_entries_from_entry_set_nonexistent_set():
    """Test that an error is raised if a non-existent entry set is requested."""
    entry_sets = {
        "test_set": {
            "required": ["_cell_length_a", "_cell_length_b"],
            "optional": ["_atom_site_fract_x", "_atom_site_fract_y"],
        }
    }
    with pytest.raises(NonExistentEntrySetError):
        cif_entries_from_entry_set(["nonexistent_set"], entry_sets)


def test_cif_entries_from_parameter_dict():
    entry_sets = {
        "test_set1": {
            "required": ["_cell_length_a", "_cell_length_b"],
            "optional": ["_atom_site_fract_x", "_atom_site_fract_y"],
        },
        "test_set2": {
            "required": ["_cell_length_c", "_atom_site_fract_x"],
            "optional": ["_atom_site_fract_z"],
        },
    }

    parameter_dict = {
        "required_entry_sets": ["test_set1"],
        "optional_entry_sets": ["test_set2"],
        "required_entries": ["_cell_angle_alpha"],
        "optional_entries": ["_cell_volume"],
        "custom_categories": ["custom"],
    }

    required, optional, categories = cif_entries_from_parameter_dict(parameter_dict, entry_sets)
    required_correct = ["_cell_length_a", "_cell_length_b", "_cell_angle_alpha"]
    optional_correct = [
        "_atom_site_fract_x",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_cell_length_c",
        "_atom_site_fract_z",
        "_cell_volume",
    ]
    categories_correct = ["custom"]
    assert sorted(required) == sorted(required_correct), "Failed to extract required entries"
    assert sorted(optional) == sorted(optional_correct), "Failed to extract optional entries"
    assert sorted(categories) == sorted(categories_correct), "Failed to extract categories"


def test_cif_entries_from_parameter_dict_no_kws():
    """Test that an error is raised if a non-existent entry set is requested."""
    entry_sets = {}

    parameter_dict = {
        "custom_categories": ["custom"],
    }
    with pytest.raises(NoKeywordsError):
        cif_entries_from_parameter_dict(parameter_dict, entry_sets)


@pytest.mark.parametrize("unified_str", ['_', '.'])
@pytest.mark.parametrize("entries, expected_output", [
    ([{'one_of': ['_cell_length_a', '_cell_volume']}], ['_cell_length_a']),
    ([{'one_of': [['_cell_length_a', '_cell_length_b'], '_cell_volume']}], ['_cell_length_a', '_cell_length_b']),
    (['_cell_length_c'], ['_cell_length_c']),
])
def test_resolve_special_entries(entries, expected_output, unified_str):
    block = model.block()
    block.add_data_item(f"_cell{unified_str}length_a", 10.0)
    block.add_data_item(f"_cell{unified_str}length_b", 20.0)
    block.add_data_item(f"_cell{unified_str}length_c", 30.0)
    assert sorted(resolve_special_entries(entries, block, [])) == sorted(expected_output)

@pytest.mark.parametrize("entries", [
    ([{'one_of': ['_cell_volume']}]),
    ([{'one_of': [['_cell_length_a', '_cell_volume']]}]),
])
def test_resolve_special_entries_error(entries):
    block = model.block()
    block.add_data_item("_cell_length_a", 10.0)
    with pytest.raises(OneOfEntryNotResolvableError):
        resolve_special_entries(entries, block, [])

@pytest.mark.parametrize("unified_str", ['_', '.'])
def test_yml_entries_resolve_special_output_settings(unified_str):
    block = model.block()
    block.add_data_item(f"_cell{unified_str}length_a", 10.0)
    block.add_data_item(f"_cell{unified_str}length_b", 20.0)
    block.add_data_item(f"_cell{unified_str}length_c", 30.0)
    output_settings_mock = YmlCifOutputSettings(
        required_entries=[{'one_of': ['_cell_length_a']}],
        optional_entries=[{'one_of': ['_cell_length_b']}],
        invalidated_entries=[],
        custom_categories=[],
        select_block="0"
    )
    resolved = yml_entries_resolve_special(output_settings_mock, block)
    assert resolved.required_entries == ['_cell_length_a']
    assert resolved.optional_entries == ['_cell_length_b']
    assert resolved.invalidated_entries == []
    assert resolved.custom_categories == []
    assert resolved.select_block == "0"
    assert isinstance(resolved, YmlCifOutputSettings), "The returned object type does not match the expected type."

@pytest.mark.parametrize("unified_str", ['_', '.'])
def test_yml_entries_resolve_special_input_settings(unified_str):
    block = model.block()
    block.add_data_item(f"_cell{unified_str}length_a", 10.0)
    block.add_data_item(f"_cell{unified_str}length_b", 20.0)
    block.add_data_item(f"_cell{unified_str}length_c", 30.0)
    input_settings_mock = YmlCifInputSettings(
        required_entries=[{'one_of': ['_cell_length_a']}],
        optional_entries=[{'one_of': ['_cell_length_b']}],
        custom_categories=[],
        merge_su=True
    )
    resolved = yml_entries_resolve_special(input_settings_mock, block)
    assert resolved.required_entries == ['_cell_length_a']
    assert resolved.optional_entries == ['_cell_length_b']
    assert resolved.custom_categories == []
    assert resolved.merge_su == True
    assert isinstance(resolved, YmlCifInputSettings), "The returned object type does not match the expected type."


def test_cif_input_entries_from_yml():
    """Test extraction of CIF input entries from a YAML file."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "parameters": [
                    {
                        "name": "test_parameter",
                        "required_entries": ["_cell_length_a"],
                        "optional_entries": ["_cell_length_b"],
                    },
                ],
            }
        ]
    }
    yml_input_settings = cif_input_entries_from_yml(yml_dict, "process_cif", "test_parameter")
    # required, optional, custom_categories, merge_su = cif_input_entries_from_yml(yml_dict, "process_cif")
    assert yml_input_settings.required_entries == ["_cell_length_a"], "Failed to extract required entries"
    assert yml_input_settings.optional_entries == ["_cell_length_b"], "Failed to extract optional entries"
    assert yml_input_settings.custom_categories == [], "Failed to extract custom categories"
    assert yml_input_settings.merge_su is False, "Failed to extract merge_su"

    yml_dict["commands"][0]["parameters"][0]["merge_su"] = True
    yml_dict["commands"][0]["parameters"][0]["custom_categories"] = ["custom"]

    yml_input_settings = cif_input_entries_from_yml(yml_dict, "process_cif", "test_parameter")
    assert yml_input_settings.required_entries == ["_cell_length_a"], "Failed to extract required entries"
    assert yml_input_settings.optional_entries == ["_cell_length_b"], "Failed to extract optional entries"
    assert yml_input_settings.custom_categories == ["custom"], "Failed to extract custom categories"
    assert yml_input_settings.merge_su is True, "Failed to extract merge_su"


def test_cif_input_entries_from_yml_exceptions():
    """Tests exceptions of cif_input_entries_from_yml."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "parameters": [
                    {"name": "test_input_parameter", "merge_su": True},
                ],
            }
        ]
    }

    with pytest.raises(NoKeywordsError) as excinfo:
        cif_input_entries_from_yml(yml_dict, "process_cif", "test_input_parameter")
    assert "process_cif" in str(excinfo.value), "Command name not included in error message"


def test_cif_output_entries_from_yml():
    """Test extraction of CIF output entries from a YAML file."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "parameters": [
                    {
                        "name": "test_output_parameter",
                        "required_entries": ["_cell_length_a"],
                        "optional_entries": ["_cell_length_b"],
                        "invalidated_entries": ["_cell_length_c"],
                        "invalidated_entry_sets": ["invalid_test"],
                        "custom_categories": ["custom"],
                    },
                ],
            }
        ],
        "cif_entry_sets": [
            {
                "name": "invalid_test",
                "required": ["_cell_volume", "_cell_length_c"],
                "optional": ["_cell_angle_alpha"],
            }
        ],
    }
    yml_output_settings = cif_output_entries_from_yml(yml_dict, "process_cif", "test_output_parameter")
    assert yml_output_settings.required_entries == ["_cell_length_a"], "Failed to extract required entries"
    assert yml_output_settings.optional_entries == ["_cell_length_b"], "Failed to extract optional entries"
    correct_invalid = ["_cell_length_c", "_cell_volume", "_cell_angle_alpha"]
    assert sorted(yml_output_settings.invalidated_entries) == sorted(
        correct_invalid
    ), "Failed to extract invalid entries"
    assert yml_output_settings.custom_categories == ["custom"], "Failed to extract custom categories"
    assert yml_output_settings.select_block == "0", "Failed to extract default output block value"

    yml_dict["commands"][0]["parameters"][0]["select_block"] = "test_block"
    yml_output_settings = cif_output_entries_from_yml(yml_dict, "process_cif", "test_output_parameter")
    assert yml_output_settings.select_block == "test_block", "Failed to extract output block value"


def test_cif_output_entries_from_yml_exceptions():
    """Tests exceptions of cif_output_entries_from_yml."""
    yml_dict = {
        "commands": [
            {
                "name": "process_cif",
                "parameters": [{"name": "test_output_parameter", "invalidated_entries": ["_cell_length_a"]}],
            }
        ]
    }

    with pytest.raises(NoKeywordsError) as excinfo:
        cif_output_entries_from_yml(yml_dict, "process_cif", "test_output_parameter")
    assert "process_cif" in str(excinfo.value), "Command name not included in error message"


@pytest.fixture(name="mock_yaml_file")
def fixture_mock_yaml_file(tmp_path):
    yml_path = tmp_path / "config.yml"

    # Mock YAML content
    yml_content = dedent("""
    cif_entry_sets :
      - name: input_test1
        required : [ _cell_length_a ]
        optional : [ _atom_site_fract_y ]
      - name: input_test2
        required : [ _invalid_keyword, _atom_site_fract_x ]
      - name: output_test
        required : [ _test_value_with_su ]
        optional : [ _test_loop_id ]

    commands :
      - name: process_cif
        parameters :
          - name: "input_cif_path"
            type: "QCrBox.input_cif"
            merge_su: Yes
            custom_categories: [test_value, test_loop]
            required_entry_sets: [input_test1]
            optional_entry_sets: [input_test2]
            required_entries: [one_of: [_cell_length_b, _invalid_cif_value]]
            optional_entries: [_atom_site_label]
          - name: "output_cif_path"
            type: "QCrBox.output_cif"
            custom_categories: [test_value, test_loop]
            required_entry_sets: [output_test]
            required_entries: [_test_loop_value_without_su]
            optional_entries: [_nonexistent_keyword]
            invalidated_entries: [_test_value_invalidated, _test_loop_value_invalidated]
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
        parameter="input_cif_path",
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


def test_cif_file_to_unified_by_yml(test_cif_file_merged, mock_yaml_file, tmp_path):
    output_cif_path = tmp_path / "output.cif"
    merge_cif = dedent("""
        data_merge_name
        _test_value.with_su 5.67
        _test_value.with_su_su 0.06
        _test_value.without_su 9.99
        _test_value.invalidated 9.01

        loop_
        _test_loop.id
        _test_loop.value_to_merge
        _test_loop.value_invalidated
        1 1.11 6.66
        2 2.22 7.77
    """)

    merge_cif_path = tmp_path / "merge_data.cif"
    merge_cif_path.write_text(merge_cif, encoding="UTF-8")

    cif_file_merge_to_unified_by_yml(
        input_cif_path=test_cif_file_merged,
        output_cif_path=output_cif_path,
        merge_cif_path=merge_cif_path,
        yml_path=mock_yaml_file,
        command="process_cif",
        parameter="output_cif_path",
    )

    # Read the output CIF content
    output_content = output_cif_path.read_text(encoding="UTF-8")
    search_patterns = (
        "data_merge_name",
        r"_test_value\.with_su\s+1\.23",  # in required-> from input
        r"_test_value\.with_su_su\s+0\.04",  # in required-> from input
        r"_test_value\.without_su\s+9\.99",  # not required or optional -> from merge
        # correctly merged loop
        r"_test_loop\.id",
        r"_test_loop\.value_to_merge",
        r"_test_loop\.value_without_su",
    )

    # cif is order agnostic, so we need to check for both possible orders
    if (
        re.search(
            r"\s*_test_loop\.id\n\s*_test_loop\.value_to_merge\n\s*_test_loop\.value_without_su\n", output_content
        )
        is not None
    ):
        search_patterns += (
            r"1\s+1\.11\s+7\.89\s*\n",
            r"2\s+2\.22\s+8\.90\s*\n",
        )
    elif (
        re.search(
            r"\s*_test_loop\.id\n\s*_test_loop\.value_without_su\n\s*_test_loop\.value_to_merge\n", output_content
        )
        is not None
    ):
        search_patterns += (
            r"1\s+7\.89\s+1\.11\s*\n",
            r"2\s+8\.90\s+2\.22\s*\n",
        )
    else:
        raise ValueError("No correctly merged loop found in output file.")

    for pattern in search_patterns:
        assert re.search(pattern, output_content) is not None
    assert "_test_value.invalidated" not in output_content, "Invalidated entry included unexpectedly"
    assert "_test_loop.value_invalidated" not in output_content, "Invalidated loop entry included unexpectedly"


def test_cif_file_to_unified_by_yml_no_merge(test_cif_file_merged, mock_yaml_file, tmp_path):
    output_cif_path = tmp_path / "output_test_data.cif"

    cif_file_merge_to_unified_by_yml(
        input_cif_path=test_cif_file_merged,
        output_cif_path=output_cif_path,
        merge_cif_path=None,
        yml_path=mock_yaml_file,
        command="process_cif",
        parameter="output_cif_path",
    )

    # Read back the output file and verify its content
    output_cif_content = output_cif_path.read_text()

    # Expected content checks
    expected_lines = [
        "data_test",
        r"_test_value\.with_su\s+1\.23",
        r"_test_value\.with_su_su\s+0\.04",
        "loop_",
        r"_test_loop\.id",
        r"_test_loop\.value_without_su",
        r"\s*1\s+7\.89",
        r"\s*2\s+8\.90",
    ]

    for line in expected_lines:
        assert re.search(line, output_cif_content) is not None, f"Expected line not found: {line}"

    unexpected_lines = [
        "_test_value_without_su",
        "_test_loop_value_with_su",
        "_test_loop_value_with_su_su",
    ]

    for line in unexpected_lines:
        assert re.search(line, output_cif_content) is None, f"Unexpected line found: {line}"


def test_cif_file_to_unified_by_yml_missing_entry(test_cif_file_merged, mock_yaml_file, tmp_path):
    output_cif_path = tmp_path / "output_test_data.cif"
    cif_content = test_cif_file_merged.read_text(encoding="UTF-8").splitlines()
    new_cif_content = "\n".join([line for line in cif_content if "_test_value_with_su" not in line])
    test_cif_file_merged.write_text(new_cif_content, encoding="UTF-8")

    with pytest.raises(ValueError):
        cif_file_merge_to_unified_by_yml(
            input_cif_path=test_cif_file_merged,
            output_cif_path=output_cif_path,
            merge_cif_path=None,
            yml_path=mock_yaml_file,
            command="process_cif",
            parameter="output_cif_path",
        )


# CLI tests

CLI_COMMAND = ["python", "-m", "qcrboxtools.cif"]


def test_cli_command_keyword(test_cif_file_unmerged, tmp_path):
    command = "to_specific"
    args = ["--required_entries", "_cell_length_a", "--merge_su"]
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
    command = "specific_by_yml"
    args = [str(mock_yaml_file), "process_cif", "input_cif_path"]
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
    command = "to_unified"
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


def test_cli_command_unify_yml(test_cif_file_merged, mock_yaml_file, tmp_path):
    command = "unified_by_yml"
    args = [str(mock_yaml_file), "process_cif", "output_cif_path"]
    expected_output_patterns = [
        "data_test",
        r"_test_value.with_su\s+1\.23",
        r"_test_value.with_su_su\s+0\.04",
        "loop_",
        r"_test_loop\.id",
        r"_test_loop\.value_without_su",
        r"\s*1\s+7\.89",
        r"\s*2\s+8\.90",
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
