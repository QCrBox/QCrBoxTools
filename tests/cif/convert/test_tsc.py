"""
Test module for `qcrboxtools.cif.file_converter.tsc`.

This module contains unit tests for TSC and TSCB file reading, parsing,
and conversion functionality.
"""

import struct

import numpy as np
import pytest

from qcrboxtools.cif.file_converter.tsc import (
    TSCBFile,
    TSCFile,
    parse_header,
    parse_tsc_data_line,
    read_tsc_file,
)


@pytest.fixture
def sample_tsc_content():
    """Sample TSC file content for testing."""
    return """TITLE: Test TSC File
SYMM: expanded
SCATTERERS: C1 C2 O1
DATA:
1 0 0 1.23450000e+00,0.00000000e+00 2.34560000e+00,1.00000000e+00 3.45670000e+00,-1.00000000e+00
0 1 0 4.56780000e+00,0.50000000e+00 5.67890000e+00,-0.50000000e+00 6.78900000e+00,0.25000000e+00
"""


@pytest.fixture
def sample_header_string():
    """Sample header string for testing."""
    return """TITLE: Test Header
SYMM: expanded
SCATTERERS: C1 C2 O1
MULTI_LINE: This is a
multi-line entry
that spans several lines"""


@pytest.fixture
def sample_tscb_file(tmp_path):
    """Create a sample TSCB file for testing."""
    tscb_path = tmp_path / "test.tscb"

    # Create minimal TSCB file content
    header_str = "TITLE: Test TSCB\nSYMM: expanded"
    scatterers_str = "C1 C2 O1"

    with open(tscb_path, "wb") as f:
        # Write header size and scatterers size
        f.write(struct.pack("2i", len(header_str), len(scatterers_str)))
        # Write header and scatterers
        f.write(header_str.encode("ASCII"))
        f.write(scatterers_str.encode("ASCII"))
        # Write number of reflections
        f.write(struct.pack("i", 2))

        # Write reflection data: hkl + form factors for 3 atoms
        # Reflection 1: (1,0,0)
        f.write(struct.pack("3i", 1, 0, 0))
        f.write(np.array([1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j], dtype=np.complex128).tobytes())

        # Reflection 2: (0,1,0)
        f.write(struct.pack("3i", 0, 1, 0))
        f.write(np.array([4.0 + 0.5j, 5.0 - 0.5j, 6.0 + 0.25j], dtype=np.complex128).tobytes())

    return tscb_path


def test_parse_header(sample_header_string):
    """Test header parsing functionality."""
    header = parse_header(sample_header_string)

    assert header["TITLE"] == " Test Header"
    assert header["SYMM"] == " expanded"
    assert header["SCATTERERS"] == " C1 C2 O1"
    assert "This is a\nmulti-line entry\nthat spans several lines" in header["MULTI_LINE"]


def test_parse_header_empty():
    """Test header parsing with empty string."""
    header = parse_header("")
    assert header == {None: "\n"}


def test_parse_tsc_data_line():
    """Test parsing of TSC data lines."""
    line = "1 0 0 1.23450000e+00,0.00000000e+00 2.34560000e+00,1.00000000e+00"
    hkl, f0js = parse_tsc_data_line(line)

    assert hkl == (1, 0, 0)
    assert len(f0js) == 2
    assert np.isclose(f0js[0], 1.2345 + 0.0j)
    assert np.isclose(f0js[1], 2.3456 + 1.0j)


def test_parse_tsc_data_line_negative_indices():
    """Test parsing TSC data line with negative Miller indices."""
    line = "-1 2 -3 1.00000000e+00,2.00000000e+00"
    hkl, f0js = parse_tsc_data_line(line)

    assert hkl == (-1, 2, -3)
    assert len(f0js) == 1
    assert np.isclose(f0js[0], 1.0 + 2.0j)


def test_read_tsc_file_tsc_extension(tmp_path, sample_tsc_content):
    """Test reading a .tsc file."""
    tsc_path = tmp_path / "test.tsc"
    tsc_path.write_text(sample_tsc_content)

    result = read_tsc_file(tsc_path)

    assert isinstance(result, TSCFile)
    assert result.scatterers == ["C1", "C2", "O1"]
    assert (1, 0, 0) in result.data
    assert (0, 1, 0) in result.data


def test_read_tsc_file_tscb_extension(sample_tscb_file):
    """Test reading a .tscb file."""
    result = read_tsc_file(sample_tscb_file)

    assert isinstance(result, TSCBFile)
    assert result.scatterers == ["C1", "C2", "O1"]
    assert (1, 0, 0) in result.data
    assert (0, 1, 0) in result.data


def test_read_tsc_file_invalid_tsc(tmp_path):
    """Test reading invalid TSC file raises ValueError."""
    invalid_tsc = tmp_path / "invalid.tsc"
    invalid_tsc.write_text("This is not a valid TSC file")

    with pytest.raises(ValueError, match="Cannot read AFF file"):
        read_tsc_file(invalid_tsc)


def test_read_tsc_file_invalid_tscb(tmp_path):
    """Test reading invalid TSCB file raises ValueError."""
    invalid_tscb = tmp_path / "invalid.tscb"
    invalid_tscb.write_bytes(b"This is not a valid TSCB file")

    with pytest.raises(ValueError, match="Cannot read AFF file"):
        read_tsc_file(invalid_tscb)


def test_tsc_file_scatterers_property():
    """Test TSCFile scatterers property."""
    tsc = TSCFile()
    tsc.header["SCATTERERS"] = "C1 C2 O1 N1"

    assert tsc.scatterers == ["C1", "C2", "O1", "N1"]


def test_tsc_file_scatterers_setter():
    """Test TSCFile scatterers setter."""
    tsc = TSCFile()
    tsc.scatterers = ["C1", "C2", "O1"]

    assert tsc.header["SCATTERERS"] == "C1 C2 O1"


def test_tsc_file_getitem_single_atom():
    """Test TSCFile indexing with single atom label."""
    tsc = TSCFile()
    tsc.scatterers = ["C1", "C2", "O1"]
    tsc.data = {
        (1, 0, 0): np.array([1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j]),
        (0, 1, 0): np.array([4.0 + 0.5j, 5.0 - 0.5j, 6.0 + 0.25j]),
    }

    result = tsc["C1"]
    expected = {(1, 0, 0): 1.0 + 0.0j, (0, 1, 0): 4.0 + 0.5j}

    assert len(result) == 2
    for hkl, value in expected.items():
        assert np.isclose(result[hkl], value)


def test_tsc_file_getitem_multiple_atoms():
    """Test TSCFile indexing with multiple atom labels."""
    tsc = TSCFile()
    tsc.scatterers = ["C1", "C2", "O1"]
    tsc.data = {
        (1, 0, 0): np.array([1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j]),
    }

    result = tsc[["C1", "O1"]]
    expected_values = np.array([1.0 + 0.0j, 3.0 - 1.0j])

    assert len(result) == 1
    assert np.allclose(result[(1, 0, 0)], expected_values)


def test_tsc_file_getitem_unknown_atom():
    """Test TSCFile indexing with unknown atom raises ValueError."""
    tsc = TSCFile()
    tsc.scatterers = ["C1", "C2"]

    with pytest.raises(ValueError, match="Unknown atom label.*O1"):
        tsc["O1"]


def test_tsc_file_from_file(tmp_path, sample_tsc_content):
    """Test TSCFile.from_file method."""
    tsc_path = tmp_path / "test.tsc"
    tsc_path.write_text(sample_tsc_content)

    tsc = TSCFile.from_file(tsc_path)

    assert tsc.header["TITLE"] == " Test TSC File"
    assert tsc.scatterers == ["C1", "C2", "O1"]
    assert len(tsc.data) == 2


def test_tsc_file_to_file(tmp_path):
    """Test TSCFile.to_file method."""
    tsc = TSCFile()
    tsc.header = {"TITLE": "Test", "SYMM": "expanded", "SCATTERERS": "C1 C2"}
    tsc.data = {(1, 0, 0): np.array([1.0 + 0.0j, 2.0 + 1.0j])}

    output_path = tmp_path / "output.tsc"
    tsc.to_file(output_path)

    # Verify file was written correctly
    content = output_path.read_text()
    assert "TITLE: Test" in content
    assert "DATA:" in content
    assert "1 0 0" in content


def test_tscb_file_from_file(sample_tscb_file):
    """Test TSCBFile.from_file method."""
    tscb = TSCBFile.from_file(sample_tscb_file)

    assert tscb.scatterers == ["C1", "C2", "O1"]
    assert len(tscb.data) == 2
    assert (1, 0, 0) in tscb.data
    assert len(tscb.data[(1, 0, 0)]) == 3


def test_tscb_file_to_file(tmp_path):
    """Test TSCBFile.to_file method."""
    tscb = TSCBFile()
    tscb.header = {"TITLE": "Test", "SYMM": "expanded", "SCATTERERS": "C1 C2"}
    tscb.data = {(1, 0, 0): np.array([1.0 + 0.0j, 2.0 + 1.0j], dtype=np.complex128)}

    output_path = tmp_path / "output.tscb"
    tscb.to_file(output_path)

    # Verify file exists and has content
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Try to read it back
    tscb_read = TSCBFile.from_file(output_path)
    assert tscb_read.scatterers == ["C1", "C2"]
    assert (1, 0, 0) in tscb_read.data


def test_tscb_file_empty_header(tmp_path):
    """Test TSCBFile with empty additional header."""
    tscb_path = tmp_path / "test_empty_header.tscb"
    scatterers_str = "C1"

    with open(tscb_path, "wb") as f:
        # Write zero header size
        f.write(struct.pack("2i", 0, len(scatterers_str)))
        f.write(scatterers_str.encode("ASCII"))
        f.write(struct.pack("i", 1))  # One reflection
        f.write(struct.pack("3i", 1, 0, 0))
        f.write(np.array([1.0 + 0.0j], dtype=np.complex128).tobytes())

    tscb = TSCBFile.from_file(tscb_path)
    assert tscb.scatterers == ["C1"]
    assert len(tscb.data) == 1


@pytest.fixture
def sample_structure_cif_content():
    """Sample structure CIF content for testing."""
    return """
data_test
_cell.length_a 10.000
_cell.length_b 12.000
_cell.length_c 8.000
_cell.angle_alpha 90.0
_cell.angle_beta 95.0
_cell.angle_gamma 90.0

loop_
_atom_site.label
_atom_site.type_symbol
_atom_site.fract_x
_atom_site.fract_y
_atom_site.fract_z
C1 C 0.1 0.2 0.3
C2 C 0.4 0.5 0.6
O1 O 0.7 0.8 0.9
"""


@pytest.fixture
def sample_tsc_cif_content():
    """Sample TSC-generated CIF content for testing."""
    return """
data_test
_cell.length_a 10.000
_cell.length_b 12.000
_cell.length_c 8.000
_cell.angle_alpha 90.0
_cell.angle_beta 95.0
_cell.angle_gamma 90.0

loop_
_wfn_moiety.id
_wfn_moiety.atom_id
_wfn_moiety.asu_atom_site_label
_wfn_moiety.atom_type_symbol
_wfn_moiety.symm_code
_wfn_moiety.cartn_x
_wfn_moiety.cartn_y
_wfn_moiety.cartn_z
_wfn_moiety.aff_index
1 1 C1 C 1_555 1.0 2.4 2.4 1
1 2 C2 C 1_555 4.0 6.0 4.8 2
1 3 O1 O 1_555 7.0 9.6 7.2 3

_aspheric_ffs.source 'test_source'
_aspheric_ffs_partitioning.name 'test_partitioning'
_aspheric_ffs_partitioning.software 'test_software'

loop_
_aspheric_ff.index_h
_aspheric_ff.index_k
_aspheric_ff.index_l
_aspheric_ff.form_factor_real
_aspheric_ff.form_factor_imag
1 0 0 '[1.00000000 2.00000000 3.00000000]' '[0.00000000 1.00000000 -1.00000000]'
0 1 0 '[4.00000000 5.00000000 6.00000000]' '[0.50000000 -0.50000000 0.25000000]'
"""


@pytest.fixture
def structure_cif_block(tmp_path, sample_structure_cif_content):
    """Create a structure CIF file and return the first block."""
    cif_path = tmp_path / "structure.cif"
    cif_path.write_text(sample_structure_cif_content)

    from qcrboxtools.cif.read import read_cif_as_unified

    return read_cif_as_unified(cif_path, 0)


@pytest.fixture
def tsc_cif_block(tmp_path, sample_tsc_cif_content):
    """Create a TSC CIF file and return the first block."""
    cif_path = tmp_path / "tsc.cif"
    cif_path.write_text(sample_tsc_cif_content)

    from qcrboxtools.cif.read import read_cif_as_unified

    return read_cif_as_unified(cif_path, 0)


def test_tsc_to_cif_conversion(structure_cif_block):
    """Test converting TSC data to CIF format."""
    # Create a TSC file with test data
    tsc = TSCFile()
    tsc.scatterers = ["C1", "C2", "O1"]
    tsc.data = {
        (1, 0, 0): np.array([1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j]),
        (0, 1, 0): np.array([4.0 + 0.5j, 5.0 - 0.5j, 6.0 + 0.25j]),
    }

    # Convert to CIF
    cif_block = tsc.to_cif(
        structure_cif_block,
        partitioning_source="test_source",
        partitioning_name="test_partitioning",
        partitioning_software="test_software",
    )

    # Verify cell parameters are preserved
    assert cif_block["_cell.length_a"] == "10.000"
    assert cif_block["_cell.length_b"] == "12.000"
    assert cif_block["_cell.angle_beta"] == "95.0"

    # Verify partitioning metadata
    assert cif_block["_aspheric_ffs.source"] == "test_source"
    assert cif_block["_aspheric_ffs_partitioning.name"] == "test_partitioning"
    assert cif_block["_aspheric_ffs_partitioning.software"] == "test_software"

    # Verify moiety loop exists
    moiety_loop = cif_block.get_loop("_wfn_moiety")
    assert moiety_loop is not None
    assert len(moiety_loop["_wfn_moiety.atom_id"]) == 3

    # Verify AFF loop exists
    aff_loop = cif_block.get_loop("_aspheric_ff")
    assert aff_loop is not None
    assert len(aff_loop["_aspheric_ff.index_h"]) == 2


def test_tsc_populate_from_cif_block(tsc_cif_block):
    """Test populating TSC from CIF block."""
    tsc = TSCFile()
    tsc.populate_from_cif_block(tsc_cif_block)

    # Verify scatterers
    assert tsc.scatterers == ["C1", "C2", "O1"]

    # Verify data was loaded correctly
    assert len(tsc.data) == 2
    assert (1, 0, 0) in tsc.data
    assert (0, 1, 0) in tsc.data

    # Verify form factor values
    hkl_100_data = tsc.data[(1, 0, 0)]
    assert np.isclose(hkl_100_data[0], 1.0 + 0.0j)
    assert np.isclose(hkl_100_data[1], 2.0 + 1.0j)
    assert np.isclose(hkl_100_data[2], 3.0 - 1.0j)


def test_tsc_from_cif_file(tmp_path, sample_tsc_cif_content):
    """Test creating TSC from CIF file."""
    cif_path = tmp_path / "test.cif"
    cif_path.write_text(sample_tsc_cif_content)

    tsc = TSCFile.from_cif_file(cif_path)

    assert tsc.scatterers == ["C1", "C2", "O1"]
    assert len(tsc.data) == 2
    assert (1, 0, 0) in tsc.data


def test_tscb_from_cif_file(tmp_path, sample_tsc_cif_content):
    """Test creating TSCB from CIF file."""
    cif_path = tmp_path / "test.cif"
    cif_path.write_text(sample_tsc_cif_content)

    tscb = TSCBFile.from_cif_file(cif_path)

    assert tscb.scatterers == ["C1", "C2", "O1"]
    assert len(tscb.data) == 2
    assert (1, 0, 0) in tscb.data


def test_populate_from_cif_block_missing_entries():
    """Test error handling when CIF block is missing required entries."""
    from iotbx.cif.model import block

    # Create incomplete block missing required entries
    incomplete_block = block()
    incomplete_block.add_data_item("_cell.length_a", "10.0")

    tsc = TSCFile()

    with pytest.raises(ValueError, match="CIF block does not contain required TSC entries"):
        tsc.populate_from_cif_block(incomplete_block)


def test_populate_from_cif_block_missing_aff_loop():
    """Test error handling when AFF loop is missing."""
    from iotbx.cif.model import block

    # Create block with metadata but no AFF loop
    incomplete_block = block()
    incomplete_block.add_data_item("_aspheric_ffs.source", "test")
    incomplete_block.add_data_item("_aspheric_ffs_partitioning.name", "test")
    incomplete_block.add_data_item("_aspheric_ffs_partitioning.software", "test")

    tsc = TSCFile()

    with pytest.raises(KeyError):
        tsc.populate_from_cif_block(incomplete_block)


def test_populate_from_cif_mismatched_atom_count():
    """Test error when AFF values don't match atom count."""
    from iotbx.cif.model import block, loop

    # Create block with mismatched data
    test_block = block()
    test_block.add_data_item("_aspheric_ffs.source", "test")
    test_block.add_data_item("_aspheric_ffs_partitioning.name", "test")
    test_block.add_data_item("_aspheric_ffs_partitioning.software", "test")

    # Moiety loop with 2 atoms
    moiety_data = {"_wfn_moiety.asu_atom_site_label": ["C1", "C2"]}
    test_block.add_loop(loop(data=moiety_data))

    # AFF loop with 3 values (mismatch)
    aff_data = {
        "_aspheric_ff.index_h": [1],
        "_aspheric_ff.index_k": [0],
        "_aspheric_ff.index_l": [0],
        "_aspheric_ff.form_factor_real": ["[1.0 2.0 3.0]"],  # 3 values
        "_aspheric_ff.form_factor_imag": ["[0.0 1.0 -1.0]"],  # 3 values
    }
    test_block.add_loop(loop(data=aff_data))

    tsc = TSCFile()

    with pytest.raises(ValueError, match="Number of AFF values is not a multiple of number of scatterers"):
        tsc.populate_from_cif_block(test_block)


def test_construct_aff_loop_formatting():
    """Test that AFF loop formats form factors correctly."""
    tsc = TSCFile()
    tsc.data = {
        (1, 0, 0): np.array([1.23456789 + 0.0j, 2.345 + 1.0j]),
        (-1, 2, -3): np.array([4.567 - 0.5j, 6.789 + 0.25j]),
    }

    aff_loop = tsc._construct_aff_loop()

    # Verify structure
    assert len(aff_loop["_aspheric_ff.index_h"]) == 2
    assert aff_loop["_aspheric_ff.index_h"][0] == "1"
    assert aff_loop["_aspheric_ff.index_k"][1] == "2"
    assert aff_loop["_aspheric_ff.index_l"][1] == "-3"

    # Verify formatting (should be wrapped in brackets with proper precision)
    real_line = aff_loop["_aspheric_ff.form_factor_real"][0]
    assert real_line.startswith("[")
    assert real_line.endswith("]")
    assert "1.23456789" in real_line


def test_round_trip_tsc_cif_conversion(tmp_path, structure_cif_block):
    """Test round-trip TSC -> CIF -> TSC conversion preserves data."""
    # Create original TSC
    original_tsc = TSCFile()
    original_tsc.scatterers = ["C1", "C2", "O1"]
    original_tsc.data = {
        (1, 0, 0): np.array([1.0 + 0.0j, 2.0 + 1.0j, 3.0 - 1.0j]),
        (0, 1, 0): np.array([4.0 + 0.5j, 5.0 - 0.5j, 6.0 + 0.25j]),
        (-1, -2, 3): np.array([7.0 + 2.0j, 8.0 - 2.0j, 9.0 + 0.1j]),
    }

    # Convert to CIF
    cif_block = original_tsc.to_cif(
        structure_cif_block, partitioning_source="test", partitioning_name="test", partitioning_software="test"
    )

    # Convert back to TSC
    reconstructed_tsc = TSCFile()
    reconstructed_tsc.populate_from_cif_block(cif_block)

    # Verify data is preserved
    assert reconstructed_tsc.scatterers == original_tsc.scatterers
    assert len(reconstructed_tsc.data) == len(original_tsc.data)

    for hkl in original_tsc.data:
        assert hkl in reconstructed_tsc.data
        np.testing.assert_allclose(reconstructed_tsc.data[hkl], original_tsc.data[hkl], rtol=1e-6)
