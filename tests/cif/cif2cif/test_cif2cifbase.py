import re
import subprocess
from itertools import product

from qcrboxtools.cif.cif2cif import cif_file_to_specific, cif_file_to_unified


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


def test_cli_non_existent_command(test_cif_file_merged, tmp_path):
    # Run the script with a non-existent command
    command = "nonexistent_command"
    output_cif_path = tmp_path / "output.cif"
    cli_args = CLI_COMMAND + [command, str(test_cif_file_merged), str(output_cif_path)]
    result = subprocess.run(cli_args, capture_output=True, text=True)

    # Check if the help message is in the output
    assert "usage:" in result.stderr
