import re
import subprocess
from itertools import product
from textwrap import dedent

from qcrboxtools.cif.cif2cif import (
    bytes_to_unified_if_cif,
    cif_file_to_specific,
    cif_file_to_unified,
    cif_text_to_unified,
    is_text_cif,
)


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


def test_cif_text_to_unified_basic():
    """Test cif_text_to_unified with basic CIF text input."""
    cif_text = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)

    result = cif_text_to_unified(cif_text, convert_keywords=False, split_sus=False)

    assert "data_test" in result
    assert "_test_value_with_su" in result
    assert "1.23(4)" in result


def test_cif_text_to_unified_with_split_sus():
    """Test cif_text_to_unified with SU splitting enabled."""
    cif_text = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)

    result = cif_text_to_unified(cif_text, convert_keywords=False, split_sus=True)

    assert "data_test" in result
    assert "_test_value_with_su" in result
    assert "_test_value_with_su_su" in result
    # Check that the value is split
    assert re.search(r"_test_value_with_su\s+1\.23", result)
    assert re.search(r"_test_value_with_su_su\s+0\.04", result)


def test_cif_text_to_unified_with_keyword_conversion():
    """Test cif_text_to_unified with keyword conversion enabled."""
    cif_text = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)

    result = cif_text_to_unified(cif_text, convert_keywords=True, split_sus=False, custom_categories=["test"])

    assert "data_test" in result
    assert "_test.value_with_su" in result
    assert re.search(r"_test\.value_with_su\s+1\.23\(4\)", result)


def test_cif_text_to_unified_full_processing():
    """Test cif_text_to_unified with both keyword conversion and SU splitting."""
    cif_text = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)

    result = cif_text_to_unified(cif_text, convert_keywords=True, split_sus=True, custom_categories=["test"])

    assert "data_test" in result
    assert "_test.value_with_su" in result
    assert "_test.value_with_su_su" in result
    # Check that both conversion and splitting occurred
    assert re.search(r"_test\.value_with_su\s+1\.23", result)
    assert re.search(r"_test\.value_with_su_su\s+0\.04", result)


# Tests for is_text_cif function


def test_is_text_cif_valid_cif():
    """Test is_text_cif with a valid CIF text."""
    cif_text = dedent("""
        data_test
        _cell_length_a 10.0
        _cell_length_b 20.0
        """)
    assert is_text_cif(cif_text) is True


def test_is_text_cif_data_block_with_leading_whitespace():
    """Test is_text_cif with data_ preceded by whitespace."""
    cif_text = "    data_test\n_cell_length_a 10.0"
    assert is_text_cif(cif_text) is True


def test_is_text_cif_with_comments_before_data():
    """Test is_text_cif with comments before data_ block."""
    cif_text = dedent("""
        # This is a comment
        # Another comment
        data_test
        _cell_length_a 10.0
        """)
    assert is_text_cif(cif_text) is True


def test_is_text_cif_comment_with_data_keyword():
    """Test is_text_cif with data_ appearing in a comment line."""
    cif_text = dedent("""
        # data_something
        _cell_length_a 10.0
        """)
    # This should return False because the actual data_ line is missing
    assert is_text_cif(cif_text) is False


def test_is_text_cif_empty_string():
    """Test is_text_cif with an empty string."""
    assert is_text_cif("") is False


def test_is_text_cif_only_whitespace():
    """Test is_text_cif with only whitespace."""
    cif_text = "   \n\n\t\n   "
    assert is_text_cif(cif_text) is False


def test_is_text_cif_only_comments():
    """Test is_text_cif with only comment lines."""
    cif_text = dedent("""
        # Comment line 1
        # Comment line 2
        # Comment line 3
        """)
    assert is_text_cif(cif_text) is False


def test_is_text_cif_non_cif_content():
    """Test is_text_cif with non-CIF content."""
    cif_text = dedent("""
        This is not a CIF file
        It has some random text
        """)
    assert is_text_cif(cif_text) is False


def test_is_text_cif_content_before_data():
    """Test is_text_cif with non-comment content before data_ block."""
    cif_text = dedent("""
        _cell_length_a 10.0
        data_test
        """)
    # Should return False because non-comment content appears before data_
    assert is_text_cif(cif_text) is False


def test_is_text_cif_multiple_data_blocks():
    """Test is_text_cif with multiple data blocks."""
    cif_text = dedent("""
        data_first
        _cell_length_a 10.0
        
        data_second
        _cell_length_b 20.0
        """)
    assert is_text_cif(cif_text) is True


def test_is_text_cif_data_underscore_variations():
    """Test is_text_cif with various data_ block names."""
    assert is_text_cif("data_") is True
    assert is_text_cif("data_test123") is True
    assert is_text_cif("data_test_with_underscores") is True
    assert is_text_cif("data_123") is True


def test_is_text_cif_mixed_content():
    """Test is_text_cif with whitespace, comments, then data block."""
    cif_text = dedent("""
        
        
        # Comment after blank lines
        
        data_test
        _cell_length_a 10.0
        """)
    assert is_text_cif(cif_text) is True


# Tests for bytes_to_unified_if_cif function


def test_bytes_to_unified_if_cif_valid_cif():
    """Test bytes_to_unified_if_cif with valid CIF bytes."""
    cif_text = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        _test_value_without_su 5.67
        """)
    cif_bytes = cif_text.encode("utf-8")

    result = bytes_to_unified_if_cif(cif_bytes, convert_keywords=False, split_sus=True)

    assert result is not None
    result_text = result.decode("utf-8")
    assert "data_test" in result_text
    assert "_test_value_with_su_su" in result_text


def test_bytes_to_unified_if_cif_non_cif_bytes():
    """Test bytes_to_unified_if_cif with non-CIF content."""
    non_cif_text = "This is not a CIF file\nJust some random text"
    non_cif_bytes = non_cif_text.encode("utf-8")

    result = bytes_to_unified_if_cif(non_cif_bytes)

    # Should return original bytes unchanged
    assert result == non_cif_bytes


def test_bytes_to_unified_if_cif_invalid_utf8():
    """Test bytes_to_unified_if_cif with invalid UTF-8 bytes."""
    invalid_bytes = b"\x80\x81\x82\x83"  # Invalid UTF-8 sequence

    result = bytes_to_unified_if_cif(invalid_bytes)

    # Should return original bytes unchanged due to decode error
    assert result == invalid_bytes


def test_bytes_to_unified_if_cif_empty_bytes():
    """Test bytes_to_unified_if_cif with empty bytes."""
    empty_bytes = b""

    result = bytes_to_unified_if_cif(empty_bytes)

    # Should return original empty bytes
    assert result == empty_bytes


def test_bytes_to_unified_if_cif_with_keyword_conversion():
    """Test bytes_to_unified_if_cif with keyword conversion enabled."""
    cif_text = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        """)
    cif_bytes = cif_text.encode("utf-8")

    result = bytes_to_unified_if_cif(cif_bytes, convert_keywords=True, custom_categories=["test"], split_sus=True)

    assert result is not None
    result_text = result.decode("utf-8")
    assert "_test.value_with_su" in result_text
    assert "_test.value_with_su_su" in result_text


def test_bytes_to_unified_if_cif_no_processing():
    """Test bytes_to_unified_if_cif with all processing disabled."""
    cif_text = dedent("""
        data_test
        _test_value_with_su 1.23(4)
        """)
    cif_bytes = cif_text.encode("utf-8")

    result = bytes_to_unified_if_cif(cif_bytes, convert_keywords=False, split_sus=False)

    assert result is not None
    result_text = result.decode("utf-8")
    assert "data_test" in result_text
    assert "1.23(4)" in result_text
    # Should not have split SUs
    assert "_test_value_with_su_su" not in result_text


def test_bytes_to_unified_if_cif_comment_only():
    """Test bytes_to_unified_if_cif with only comments (not a valid CIF)."""
    comment_text = "# Just a comment\n# Another comment"
    comment_bytes = comment_text.encode("utf-8")

    result = bytes_to_unified_if_cif(comment_bytes)

    # Should return original bytes unchanged
    assert result == comment_bytes


def test_bytes_to_unified_if_cif_utf8_with_bom():
    """Test bytes_to_unified_if_cif with UTF-8 BOM."""
    cif_text = dedent("""
        data_test
        _cell_length_a 10.0
        """)
    # UTF-8 with BOM
    cif_bytes = b"\xef\xbb\xbf" + cif_text.encode("utf-8")

    result = bytes_to_unified_if_cif(cif_bytes)

    assert result is not None
    result_text = result.decode("utf-8")
    # BOM should be preserved in the decode/encode cycle
    assert "data_test" in result_text


def test_bytes_to_unified_if_cif_multiline_values():
    """Test bytes_to_unified_if_cif with multiline CIF values."""
    cif_text = dedent("""
        data_test
        _test_text
        ;
        This is a multiline value
        in CIF format
        ;
        """)
    cif_bytes = cif_text.encode("utf-8")

    result = bytes_to_unified_if_cif(cif_bytes, convert_keywords=False, split_sus=False)

    assert result is not None
    result_text = result.decode("utf-8")
    assert "data_test" in result_text
    assert "multiline value" in result_text
