from io import StringIO
from pathlib import Path

import pytest
from iotbx import cif

from qcrboxtools.cif.file_converter.shelxl import cif2shelx_ins, shelxl_commands
from qcrboxtools.cif.read import cifdata_str_or_index, read_cif_as_unified


@pytest.fixture(name="sample_cif_block")
def fixture_sample_cif_block():
    cif_string = """
    data_test
    _cell.length_a 10.0
    _cell.length_b 12.0
    _cell.length_c 14.0
    _cell.angle_alpha 90.0
    _cell.angle_beta 90.0
    _cell.angle_gamma 90.0
    _cell.length_a_su 0.1
    _cell.length_b_su 0.2
    _cell.length_c_su 0.3
    _cell.formula_units_z 4
    _diffrn_radiation_wavelength.value 0.12345
    _symmetry_space_group_name_H-M 'P 21 21 21'
    loop_
    _atom_site.label
    _atom_site.type_symbol
    C1 C
    N1 N
    O1 O
    H1 H
    C2 C
    C3 C
    N2 N
    """
    cif_file = StringIO(cif_string)
    cif_model = cif.reader(file_object=cif_file).model()
    return cif_model.blocks["test"]


def test_block2ins_symm(sample_cif_block):
    symm_lines = cif2shelx_ins.block2ins_symm(sample_cif_block).split("\n")
    expected = ["LATT -1", "SYMM X+1/2,-Y+1/2,-Z", "SYMM -X,Y+1/2,-Z+1/2", "SYMM -X+1/2,-Y,Z+1/2"]
    for line in symm_lines:
        assert line in expected, f"{line} not in {expected}"
    assert len(symm_lines) == len(expected)


def test_block2cell_zerr(sample_cif_block):
    cell_line, zerr_line = cif2shelx_ins.block2cell_zerr(sample_cif_block)
    assert cell_line == "CELL 0.12345 10.0 12.0 14.0 90.0 90.0 90.0"
    assert zerr_line == "ZERR 4 0.1 0.2 0.3 0.0 0.0 0.0"


def test_create_shelx_sfacs():
    atom_site_atom_types = ["N", "O", "H", "C", "C", "N"]
    result = cif2shelx_ins.create_shelx_sfacs(atom_site_atom_types)
    expected = ["C", "H", "N", "O"]
    assert result == expected


def test_block2sfac_unit(sample_cif_block):
    sfac_line, unit_line = cif2shelx_ins.block2sfac_unit(sample_cif_block)
    expected_sfac = "SFAC C H N O"
    expected_unit = "UNIT 12 4 8 4"
    assert sfac_line == expected_sfac
    assert unit_line == expected_unit


@pytest.mark.parametrize(
    "weight_entry, expected",
    [
        (r"w=1/[\s^2^(Fo^2^)+(0.0173P)^2^+2.2099P] where P=(Fo^2^+2Fc^2^)/3", "WGHT 0.0173 2.2099"),
        (r"w=1/[\s^2^(Fo^2^)] where P=(Fo^2^+2Fc^2^)/3", "WGHT 0.0 0.0"),
        (r"w=1/[\s^2^(Fo^2^)+(0.1235P)^2^] where P=(Fo^2^+2Fc^2^)/3", "WGHT 0.1235 0.0"),
        (r"w=1/[\s^2^(Fo^2^)+0.1235P] where P=(Fo^2^+2Fc^2^)/3", "WGHT 0.0 0.1235"),
    ],
)
def test_block2wght(sample_cif_block, weight_entry, expected):
    sample_cif_block["_refine_ls.weighting_details"] = weight_entry
    wght_line = cif2shelx_ins.block2wght(sample_cif_block)
    assert wght_line == expected


def test_block2wght_no_weighting(sample_cif_block):
    with pytest.raises(cif2shelx_ins.NoWeightingDetailsError):
        cif2shelx_ins.block2wght(sample_cif_block)


@pytest.fixture
def comprehensive_cif_block():
    cif_string = r"""
    data_test
    _cell.length_a 10.5000
    _cell.length_b 20.7000
    _cell.length_c 30.9000
    _cell.angle_alpha 90.0000
    _cell.angle_beta 105.0000
    _cell.angle_gamma 90.0000
    _cell.formula_units_z 4
    _cell.volume 6615.5770
    _diffrn_radiation_wavelength.value 0.71073
    _symmetry_space_group_name_H-M 'P 21/c'
    _refine_ls.weighting_details 'w = 1/[\s^2^(Fo^2^) + (0.0500P)^2^ + 1.5000P]'
    _diffrn.ambient_temperature 293
    _exptl_crystal.size_max 0.5
    _exptl_crystal.size_mid 0.4
    _exptl_crystal.size_min 0.3
    _qcrbox.shelx.scale_factor 0.05
    loop_
    _atom_site.label
    _atom_site.type_symbol
    C1 C
    N1 N
    O1 O
    H1 H
    Fe1 Fe
    """
    cif_file = StringIO(cif_string)
    cif_model = cif.reader(file_object=cif_file).model()
    return cif_model.blocks["test"]


def test_block2header(comprehensive_cif_block):
    result = cif2shelx_ins.block2header(comprehensive_cif_block)
    expected_lines = [
        "TITL  QCrBox generated cif",
        "CELL 0.71073 10.5000 20.7000 30.9000 90.0000 105.0000 90.0000",
        "ZERR 4 0.0 0.0 0.0 0.0 0.0 0.0",
        "LATT 1",
        "SYMM -X,Y+1/2,-Z+1/2",
        "SFAC C H N O Fe",
        "UNIT 4 4 4 4 4",
        "TEMP 20.0",
        "SIZE 0.5 0.4 0.3",
        "CONF",
        "BOND $H",
        "L.S. 10",
        "LIST 4",
        "ACTA",
        "BOND",
        "FMAP 2",
        "MORE -1",
        "WGHT 0.05 1.5",
        "FVAR 0.05",
    ]

    result_lines = result.split("\n")

    for line in result_lines:
        assert line in expected_lines

    assert len(result_lines) == len(expected_lines)


@pytest.fixture(name="atom_site_data")
def fixture_atom_site_data():
    cif_string = """
    data_test
    loop_
    _atom_site.label
    _atom_site.type_symbol
    _atom_site.fract_x
    _atom_site.fract_y
    _atom_site.fract_z
    _atom_site.u_iso_or_equiv
    C1 C 0.25000 0.33333 0.50000 0.05000
    N1 N 0.12500 0.66667 0.25000 0.04000
    O1 O 0.37500 0.50000 0.75000 0.06000
    H1 H 0.50000 0.16667 0.00000 0.07000
    Fe1 Fe 0.00000 0.00000 0.00000 0.03000
    loop_
    _atom_site_aniso.label
    _atom_site_aniso.u_11
    _atom_site_aniso.u_22
    _atom_site_aniso.u_33
    _atom_site_aniso.u_23
    _atom_site_aniso.u_13
    _atom_site_aniso.u_12
    Fe1 0.02000 0.02500 0.03000 0.00100 0.00200 0.00300
    """
    cif_file = StringIO(cif_string)
    cif_model = cif.reader(file_object=cif_file).model()
    return cif_model.blocks["test"]


@pytest.mark.parametrize(
    "index, expected_output",
    [
        (0, "C1 1   0.25000   0.33333   0.50000  11.00000   0.05000"),
        (1, "N1 3   0.12500   0.66667   0.25000  11.00000   0.04000"),
        (2, "O1 4   0.37500   0.50000   0.75000  11.00000   0.06000"),
        (3, "H1 2   0.50000   0.16667   0.00000  11.00000   0.07000"),
        (
            4,
            (
                "Fe1 5   0.00000   0.00000   0.00000  11.00000   0.02000   0.02500 =\n"
                + "  0.03000   0.00100   0.00200   0.00300"
            ),
        ),
    ],
)
def test_create_atom_string(atom_site_data, index, expected_output):
    atom_site_loop = atom_site_data.loops["_atom_site"]
    atom_site_aniso_loop = atom_site_data.loops["_atom_site_aniso"]

    result = cif2shelx_ins.create_atom_string(index, atom_site_loop, atom_site_aniso_loop)

    assert result == expected_output, f"For atom at index {index}, expected:\n{expected_output}\nbut got:\n{result}"


@pytest.mark.parametrize(
    "index, uiso_mult, expected_output",
    [
        (3, 1.2, "H1 2   0.50000   0.16667   0.00000  11.00000 -1.20"),
    ],
)
def test_create_atom_string_with_uiso_mult(atom_site_data, index, uiso_mult, expected_output):
    atom_site_loop = atom_site_data.loops["_atom_site"]
    atom_site_aniso_loop = atom_site_data.loops["_atom_site_aniso"]

    result = cif2shelx_ins.create_atom_string(index, atom_site_loop, atom_site_aniso_loop, uiso_mult)

    assert result == expected_output, (
        f"For atom at index {index} with uiso_mult {uiso_mult}, expected:\n{expected_output}\nbut got:\n{result}"
    )


@pytest.fixture(name="afix_objects")
def fixture_afix_objects():
    cif_path = Path(__file__).parent / "test_files" / "afix_table.cif"
    cif_block, _ = cifdata_str_or_index(read_cif_as_unified(cif_path), 0)

    res_path = cif_path.with_name("from_afix_table.res")
    return cif_block, res_path.read_text()


def test_create_atom_list(afix_objects):
    cif_block, res_text = afix_objects
    lines = cif2shelx_ins.create_atom_list(cif_block).split("\n")
    res_lines = res_text.split("\n")

    for line in lines:
        if len(line.strip()) == 0 or line.startswith(shelxl_commands):
            continue
        assert line in res_lines, f"{line} not in {res_lines}"


def test_qcrbox_cif2ins(afix_objects):
    cif_block, res_text = afix_objects
    ins_text = cif2shelx_ins.qcrbox_cif2ins(cif_block)
    res_lines = res_text.split("\n")
    ins_lines = ins_text.split("\n")

    for line in ins_lines:
        if len(line.strip()) == 0 or line.startswith("REM"):
            continue
        assert line in res_lines, f"{line} not in {res_lines}"

    cif_block["_iucr.refine_instructions_details"] = "TITL Test title\nREM Test comment"
    assert cif2shelx_ins.qcrbox_cif2ins(cif_block) == "TITL Test title\nREM Test comment"

    del cif_block["_iucr.refine_instructions_details"]
    cif_block["_shelx.res_file"] = "TITL Test title2\nREM Test comment2"
    assert cif2shelx_ins.qcrbox_cif2ins(cif_block) == "TITL Test title2\nREM Test comment2"
