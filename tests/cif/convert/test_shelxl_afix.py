from textwrap import dedent

import pytest
from iotbx.cif.model import block, loop

from qcrboxtools.cif.file_converter.shelxl import afix2cif


def test_afix2mn():
    assert afix2cif.afix2mn(6) == (0, 6)
    assert afix2cif.afix2mn(23) == (2, 3)


@pytest.mark.parametrize("afix_line, expected", [("AFIX 6", (0, 6)), ("AFIX 23", (2, 3)), ("AFIX 137", (13, 7))])
def test_afix_line2mn(afix_line, expected):
    assert afix2cif.afix_line2mn(afix_line) == expected


def test_afix_m2cif():
    for afix_m in range(17):
        result = afix2cif.afix_m2cif(afix_m)
        for line in result.split("\n"):
            assert len(line) < 80, "Line length in cif should be limited for compatibility with older software"
    with pytest.raises(KeyError):
        afix2cif.afix_m2cif(17)


def test_afix_n2cif():
    assert afix2cif.afix_n2cif(6) == "RO"
    with pytest.raises(KeyError):
        afix2cif.afix_n2cif(17)


@pytest.fixture(name="minimal_ins")
def fixture_minimal_ins():
    return dedent(
        """\
        TITL Some title
            Title continuation line
        CELL 1.2345 2.3456 3.4567 90 90 90
        ZERR 0.0001 0.0002 0.0003 0 0 0
        SFAC C H Pt
        UNIT 100 200 300 400
        FVAR 0.1234
        PT1 3 0.9876 0.8765 0.7654 11.000 0.4321 0.3210 =
                0.2109 0.1098 0.0987 0.9876
        C1 1 0.1234 0.2345 0.3456 11.0000 0.5678 0.6789 =
              0.7890 0.8901 0.9012 0.1234
        AFIX 137
        H1A 2 -0.1234 -0.2345 -0.3456 11.000 -1.5
        H1B 2 -0.4567 -0.5678 -0.6789 11.000 -1.5
        H1C 2 -0.7890 -0.8901 -0.9012 11.000 -1.5
        AFIX 66
        C1A 1 0.1357 0.2468 0.3579 11.0000 0.5791 0.7912 =
               0.9123 0.1234 0.2345 0.3456
        C2A 1 0.0369 0.1470 0.2581 11.0000 0.3690 0.4701 =
               0.5812 0.6923 0.8234 0.9345
        AFIX 43
        H2A 2 -0.3690 -0.4701 -0.5812 11.000 -1.2
        AFIX 65
        C3A 1 0.2604 0.3715 0.4826 11.0000 0.5934 0.7045 =
               0.8156 0.9267 0.0378 0.1489
        C4A 1 0.0506 0.1617 0.2728 11.0000 0.3845 0.4956 =
               0.5067 0.6178 0.7289 0.8390
        H4A 2 -0.2845 -0.3956 -0.5067 11.000 -1.2
        C5A 1 0.2845 0.3956 0.5067 11.0000 0.6178 0.7289 =
               0.8390 0.9501 0.0612 0.1723
        C6A 1 0.0747 0.1858 0.2969 11.0000 0.4079 0.5180 =
                0.6291 0.7402 0.8513 0.9624
        AFIX 0
        CISO 1 0.1111 0.2222 0.3333 11.0000 0.4444
        HKLF4
        END
        """
    )


def test_ins2atom_site_dicts(minimal_ins):
    atom_site_collect, atom_site_iso_collect, atom_site_aniso_collect = afix2cif.ins2atom_site_dicts(minimal_ins)
    assert all(len(atom_site_collect[key]) == 14 for key in atom_site_collect)
    assert all(len(atom_site_iso_collect[key]) == 1 for key in atom_site_iso_collect)
    assert all(len(atom_site_aniso_collect[key]) == 8 for key in atom_site_aniso_collect)

    assert atom_site_collect["_atom_site.label"][0] == "Pt1"
    assert atom_site_collect["_atom_site.fract_x"][1] == 0.1234  # C1
    assert atom_site_collect["_atom_site.fract_y"][2] == -0.2345  # H1A

    assert atom_site_iso_collect["_atom_site.label"][0] == "CISO"
    assert "H1" not in atom_site_iso_collect["_atom_site.label"], "Only refined isotropic atoms should be included"

    assert atom_site_aniso_collect["_atom_site_aniso.label"][4] == "C3A"
    assert "CISO" not in atom_site_aniso_collect["_atom_site_aniso.label"]
    assert atom_site_aniso_collect["_atom_site_aniso.u_11"][1] == 0.5678  # C1A


def test_create_atom_site_constraints_general(minimal_ins):
    atom_site_collect_dict, constraint_posn_dict = afix2cif.create_atom_site_constraints(minimal_ins)
    for val in atom_site_collect_dict.values():
        assert len(val) == 14
    for val in constraint_posn_dict.values():
        assert len(val) == 3
    for afix in (137, 66, 43):
        assert f"SXL{afix}" in constraint_posn_dict["_qcrbox_constraint_posn.id"]


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, "."),  # Pt1, general case
        (2, "C1"),  # H1A, H-afix bound to last atom before afix
        (5, "."),  # C1A, Rigid group afix, first atom not bound
        (6, "C1A"),  # C2A, Rigid group afix, bound to first atom in group
        (8, "C1A"),  # C3A, Rigid group afix, bound to first atom in group
        (10, "."),  # H4A, H-atom not included in rigid group
        (7, "C2A"),  # H2A, H-afix in rigid group, bound again to atom before afix
        (-1, "."),  # CISO, general case, afix exitted
    ],
)
def test_create_atom_site_constraints_attached_atom(index, expected, minimal_ins):
    atom_site_collect_dict, _ = afix2cif.create_atom_site_constraints(minimal_ins)

    assert atom_site_collect_dict["_atom_site.calc_attached_atom"][index] == expected


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, "."),  # Pt1, general case
        (2, "SXL137"),  # H1A, H-afix bound to last atom before afix
        (5, "SXL66"),  # C1A, Rigid group afix
        (8, "SXL66"),  # C3A, Rigid group afix, recovered after H-afix
        (7, "SXL43"),  # C4A, H-afix in rigid group
        (10, "."),  # H4A, H-atom not included in rigid group
        (-1, "."),  # CISO, general case, afix exitted
    ],
)
def test_create_atom_site_constraints_attached_atom_constraint_id(index, expected, minimal_ins):
    atom_site_collect_dict, _ = afix2cif.create_atom_site_constraints(minimal_ins)
    assert atom_site_collect_dict["_atom_site.qcrbox_constraint_posn_id"][index] == expected


@pytest.fixture(name="minimal_cif_block", params=["_iucr.refine_instructions_details", "_shelx.res_file"])
def fixture_minimal_cif(minimal_ins, request):
    labels = ["Pt1", "C1", "H1A", "H1B", "H1C", "C1A", "C2A", "H2A", "C3A", "C4A", "H4A", "C5A", "C6A", "CISO"]
    atom_site_dict = {
        "_atom_site.label": labels,
        "_atom_site.fract_x": [0.0] * len(labels),
        "_atom_site.fract_y": [0.0] * len(labels),
        "_atom_site.fract_z": [0.0] * len(labels),
        "_atom_site.occupancy": [1.0] * len(labels),
        "_atom_site.u_iso_or_equiv": [2.0] * len(labels),
    }
    aniso_labels = ["Pt1", "C1", "C1A", "C2A", "C3A", "C4A", "C5A", "C6A"]
    atom_site_aniso_dict = {
        "_atom_site_aniso.label": aniso_labels,
        "_atom_site_aniso.u_11": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_22": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_33": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_12": [0.0] * len(aniso_labels),
        "_atom_site_aniso.u_13": [0.0] * len(aniso_labels),
        "_atom_site_aniso.u_23": [0.0] * len(aniso_labels),
    }

    cif_block = block()
    atom_site_loop = loop()
    atom_site_loop.add_columns(atom_site_dict)
    cif_block.add_loop(atom_site_loop)
    atom_site_aniso_loop = loop()
    atom_site_aniso_loop.add_columns(atom_site_aniso_dict)
    cif_block.add_loop(atom_site_aniso_loop)
    cif_block.add_data_item(request.param, minimal_ins)
    return cif_block


def test_afix2cif_update_tables(minimal_cif_block):
    original = minimal_cif_block.deepcopy()
    result = afix2cif.afix_to_cif(minimal_cif_block)
    assert result["_qcrbox.shelx.scale_factor"] == "0.1234"
    changed_columns = [
        "_atom_site.fract_x",
        "_atom_site.fract_y",
        "_atom_site.fract_z",
        "_atom_site_aniso.u_12",
        "_atom_site_aniso.u_13",
        "_atom_site_aniso.u_23",
        "_atom_site_aniso.u_11",
        "_atom_site_aniso.u_22",
        "_atom_site_aniso.u_33",
    ]
    for column in changed_columns:
        assert all(float(new) != float(old) for new, old in zip(result[column], original[column])), (
            f"{column} should be different"
        )


def test_afix2cif_add_columns(minimal_cif_block):
    result = afix2cif.afix_to_cif(minimal_cif_block)
    new_columns = [
        "_atom_site.calc_attached_atom",
        "_atom_site.qcrbox_constraint_posn_id",
        "_atom_site.qcrbox_constraint_posn_index",
        "_atom_site.qcrbox_calc_uiso_multiplier",
        "_qcrbox_constraint_posn.id",
        "_qcrbox_constraint_posn.refined_pars",
        "_qcrbox_constraint_posn.instruction",
    ]
    for column in new_columns:
        assert column in result, f"{column} should be added to the CIF block"


def test_afix2cif_remove_instructions(minimal_cif_block):
    result = afix2cif.afix_to_cif(minimal_cif_block)
    assert "_iucr.refine_instructions_details" not in result, "Refine instructions should be removed from the CIF block"
    assert "_shelx.res_file" not in result, "Shelx res file should be removed from the CIF block"


def test_afix2cif_keep_instructions(minimal_cif_block):
    if "_iucr.refine_instructions_details" in minimal_cif_block:
        key = "_iucr.refine_instructions_details"
    elif "_shelx.res_file" in minimal_cif_block:
        key = "_shelx.res_file"
    else:
        raise NotImplementedError("Test fixture is not working as expected")
    ins_string = minimal_cif_block[key]
    ins_string = ins_string.replace("FVAR", "ABIN\nFVAR")
    minimal_cif_block[key] = ins_string
    result = afix2cif.afix_to_cif(minimal_cif_block)
    assert key in result, "Refine instructions should be kept in the CIF block"


def test_afix_to_cif_no_entry(minimal_cif_block):
    if "_iucr.refine_instructions_details" in minimal_cif_block:
        del minimal_cif_block["_iucr.refine_instructions_details"]
    elif "_shelx.res_file" in minimal_cif_block:
        del minimal_cif_block["_shelx.res_file"]
    else:
        raise NotImplementedError("Test fixture is not working as expected")
    with pytest.raises(afix2cif.MissingRefineInstructionsError):
        afix2cif.afix_to_cif(minimal_cif_block)


def test_afix2cif_atom_site_nonmatch(minimal_ins):
    labels = ["Pt1"]
    atom_site_dict = {
        "_atom_site.label": labels,
        "_atom_site.fract_x": [0.0] * len(labels),
        "_atom_site.fract_y": [0.0] * len(labels),
        "_atom_site.fract_z": [0.0] * len(labels),
        "_atom_site.occupancy": [1.0] * len(labels),
        "_atom_site.u_iso_or_equiv": [2.0] * len(labels),
    }
    atom_site_loop = loop()
    atom_site_loop.add_columns(atom_site_dict)

    aniso_labels = ["Pt1", "C1", "C1A", "C2A", "C3A", "C4A", "C5A", "C6A"]
    atom_site_aniso_dict = {
        "_atom_site_aniso.label": aniso_labels,
        "_atom_site_aniso.u_11": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_22": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_33": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_12": [0.0] * len(aniso_labels),
        "_atom_site_aniso.u_13": [0.0] * len(aniso_labels),
        "_atom_site_aniso.u_23": [0.0] * len(aniso_labels),
    }

    cif_block = block()
    atom_site_loop = loop()
    atom_site_loop.add_columns(atom_site_dict)
    cif_block.add_loop(atom_site_loop)
    atom_site_aniso_loop = loop()
    atom_site_aniso_loop.add_columns(atom_site_aniso_dict)
    cif_block.add_loop(atom_site_aniso_loop)
    cif_block.add_data_item("_iucr.refine_instructions_details", minimal_ins)
    with pytest.raises(afix2cif.CifInsAtomsMismatchError):
        afix2cif.afix_to_cif(cif_block)


def test_afix2cif_atom_site_aniso_nonmatch(minimal_ins):
    labels = ["Pt1", "C1", "H1A", "H1B", "H1C", "C1A", "C2A", "H2A", "C3A", "C4A", "H4A", "C5A", "C6A", "CISO"]
    atom_site_dict = {
        "_atom_site.label": labels,
        "_atom_site.fract_x": [0.0] * len(labels),
        "_atom_site.fract_y": [0.0] * len(labels),
        "_atom_site.fract_z": [0.0] * len(labels),
        "_atom_site.occupancy": [1.0] * len(labels),
        "_atom_site.u_iso_or_equiv": [2.0] * len(labels),
    }
    aniso_labels = ["Pt1"]
    atom_site_aniso_dict = {
        "_atom_site_aniso.label": aniso_labels,
        "_atom_site_aniso.u_11": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_22": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_33": [1.0] * len(aniso_labels),
        "_atom_site_aniso.u_12": [0.0] * len(aniso_labels),
        "_atom_site_aniso.u_13": [0.0] * len(aniso_labels),
        "_atom_site_aniso.u_23": [0.0] * len(aniso_labels),
    }

    cif_block = block()
    atom_site_loop = loop()
    atom_site_loop.add_columns(atom_site_dict)
    cif_block.add_loop(atom_site_loop)
    atom_site_aniso_loop = loop()
    atom_site_aniso_loop.add_columns(atom_site_aniso_dict)
    cif_block.add_loop(atom_site_aniso_loop)
    cif_block.add_data_item("_iucr.refine_instructions_details", minimal_ins)
    with pytest.raises(afix2cif.CifInsAtomsMismatchError):
        afix2cif.afix_to_cif(cif_block)
