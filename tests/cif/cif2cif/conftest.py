from textwrap import dedent

import pytest


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


@pytest.fixture(name="test_cif_file_unmerged")
def fixture_test_cif_file_unmerged(tmp_path):
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
