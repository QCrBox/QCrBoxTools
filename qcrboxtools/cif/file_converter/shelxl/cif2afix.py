import re
from io import StringIO
from textwrap import wrap
from typing import Optional, Tuple

import numpy as np
from iotbx.cif.builders import crystal_symmetry_builder
from iotbx.cif.model import block, loop
from iotbx.shelx.write_ins import LATT_SYMM

from . import element_list


def block2ins_symm(cif_block: block) -> str:
    """
    Convert CIF block symmetry information to SHELX format.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing symmetry information.

    Returns
    -------
    str
        SHELX formatted symmetry instructions.
    """
    symmetry = crystal_symmetry_builder(cif_block).crystal_symmetry
    with StringIO() as fobj:
        LATT_SYMM(fobj, symmetry.space_group())
        instr = fobj.getvalue()
    return instr.strip()


def block2cell_zerr(cif_block: block) -> Tuple[str, str]:
    """
    Create SHELXL CELL and ZERR lines from CIF block.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing the necessary information.

    Returns
    -------
    tuple of str
        CELL and ZERR lines for SHELX format.
    """
    cell_parameters = ["length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma"]

    cell_line = "CELL " + " ".join(
        [
            str(cif_block["_diffrn_radiation_wavelength.value"]),
            *(str(cif_block[f"_cell.{par}"]) for par in cell_parameters),
        ]
    )

    zerr_line = "ZERR " + " ".join(
        [
            str(cif_block["_cell.formula_units_z"]),
            *(str(cif_block.get(f"_cell.{par}_su", 0.0)) for par in cell_parameters),
        ]
    )

    return cell_line, zerr_line


def create_shelx_sfacs(atom_site_atom_types):
    """
    Create a list of elements in SFAC ordering from atom types in the CIF file.
    """
    atom_types = [x.upper() for x in atom_site_atom_types]
    sfac_elements = [x for x in element_list if x.upper() in atom_types]
    return sfac_elements


def block2sfac_unit(cif_block: block) -> Tuple[str, str]:
    """
    Generate SFAC and UNIT lines from CIF block.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing atom information.

    Returns
    -------
    tuple of str
        SFAC and UNIT lines for SHELX format.
    """
    sfac_els = create_shelx_sfacs(cif_block["_atom_site.type_symbol"])
    upper_els = [el.upper() for el in cif_block["_atom_site.type_symbol"]]

    sfac_dict = {}
    for sfac_el in sfac_els:
        count = upper_els.count(sfac_el.upper())
        if count == 0:
            continue
        sfac_dict[sfac_el] = count
    sfac_line = "SFAC " + " ".join(sfac_dict.keys())
    cell_z = int(cif_block["_cell.formula_units_z"])
    unit_line = "UNIT " + " ".join(str(n * cell_z) for n in sfac_dict.values())
    return sfac_line, unit_line


def block2wght(cif_block: block) -> str:
    """
    Extract weighting scheme parameters and create WGHT line from CIF block.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing weighting details.

    Returns
    -------
    str
        WGHT instruction for SHELX format.
    """
    search_a = re.search(r"\(([-+.\d]+?)P\)\^2\^", cif_block["_refine_ls.weighting_details"])
    if search_a is not None:
        a = float(search_a.group(1))
    else:
        a = 0.0
    search_b = re.search(r"([+-]?\d+\.\d+)P(?!\)\^2\^)", cif_block["_refine_ls.weighting_details"])
    if search_b is not None:
        b = float(search_b.group(1))
    else:
        b = 0.0

    return f"WGHT {a} {b}"


def create_header(cif_block: block) -> str:
    """
    Create the header section of a SHELX instruction file from CIF data.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing structure information.

    Returns
    -------
    str
        Header section of SHELX instruction file.
    """
    ins_lines = ["TITL  QCrBox generated cif"]

    cell_line, zerr_line = block2cell_zerr(cif_block)
    ins_lines.append(cell_line)
    ins_lines.append(zerr_line)
    symm_lines = block2ins_symm(cif_block)
    ins_lines.append(symm_lines)
    sfac_line, unit_line = block2sfac_unit(cif_block)
    ins_lines.append(sfac_line)
    ins_lines.append(unit_line)

    if "_diffrn.ambient_temperature" in cif_block:
        temp_c = float(cif_block["_diffrn.ambient_temperature"]) - 273.0
        ins_lines.append(f"TEMP {temp_c}")
    if all(f"_exptl_crystal.size_{val}" in cif_block for val in ("max", "mid", "min")):
        ins_lines.append("SIZE " + " ".join(cif_block[f"_exptl_crystal.size_{val}"] for val in ("max", "mid", "min")))

    ins_lines.append("CONF\nBOND $H\nL.S. 10\nLIST 4\nACTA\nBOND\nFMAP 2\nMORE -1")
    ins_lines.append(block2wght(cif_block))
    ins_lines.append(f'FVAR {cif_block["_qcrbox.shelx.scale_factor"]}')

    return "\n".join(ins_lines)


def create_atom_string(
    index: int, atom_site_loop: loop, atom_site_aniso_loop: Optional[loop] = None, uiso_mult: Optional[float] = None
) -> str:
    """
    Create a SHELX-formatted atom string from CIF data.

    Parameters
    ----------
    index : int
        Index of the atom in the atom_site_loop.
    atom_site_loop : iotbx.cif.model.loop
        CIF loop object containing atom site information.
    atom_site_aniso_loop : iotbx.cif.model.loop, optional
        CIF loop object containing anisotropic displacement parameters.
    uiso_mult : float, optional
        Uiso is set to a multiple of the calculated value of the bound atom

    Returns
    -------
    str
        SHELX-formatted atom string.
    """
    sfac_list = create_shelx_sfacs(atom_site_loop["_atom_site.type_symbol"])
    uij_indexes = (11, 22, 33, 23, 13, 12)
    label = atom_site_loop["_atom_site.label"][index]
    atom_type = sfac_list.index(atom_site_loop["_atom_site.type_symbol"][index]) + 1
    x = float(atom_site_loop["_atom_site.fract_x"][index])
    y = float(atom_site_loop["_atom_site.fract_y"][index])
    z = float(atom_site_loop["_atom_site.fract_z"][index])
    occ = 11.0
    start = f"{label} {atom_type} {x: 9.5f} {y: 9.5f} {z: 9.5f} {occ: 9.5f}"
    if uiso_mult is not None:
        atom_string = f"{start} {float(-uiso_mult): 4.2f}"
    elif (atom_site_aniso_loop is not None) and (label in atom_site_aniso_loop["_atom_site_aniso.label"]):
        index_aniso = list(atom_site_aniso_loop["_atom_site_aniso.label"]).index(label)
        uijs = [f'{float(atom_site_aniso_loop[f"_atom_site_aniso.u_{ij}"][index_aniso]): 9.5f}' for ij in uij_indexes]
        atom_string = start + " " + " ".join(uijs)
    else:
        atom_string = f'{start} {atom_site_loop["_atom_site.u_iso_or_equiv"][index]: 9.5f}'
    return " =\n  ".join(wrap(atom_string))


def create_atom_table_lines(
    indexes, cif_block, atom_site_loop, atom_site_aniso_loop, attached_collect, parent_afix_m=None
):
    non_h_ns = (0, 1, 2, 5, 6, 9)
    non_h_ms = (0, 5, 6, 7, 10, 11)
    lines = []
    current_m = parent_afix_m
    for index in indexes:
        int_index = int(index)
        label = atom_site_loop["_atom_site.label"][int_index]

        constraint_entry = atom_site_loop["_atom_site.qcrbox_constraint_posn_id"][int_index]
        if constraint_entry == ".":
            m = 0
            n = 0
        else:
            assert constraint_entry.startswith("SXL"), "Cannot create a SHELXL res with non-SHELXL constraints"
            constraint_id = int(atom_site_loop["_atom_site.qcrbox_constraint_posn_id"][int_index][3:])
            m = constraint_id // 10
            n = constraint_id % 10
            if m != parent_afix_m and not (m in non_h_ms or n in non_h_ns) and m != current_m:
                lines.append(f"AFIX {m * 10 + n}")
                current_m = m
        if m != parent_afix_m and (m in non_h_ms or n in non_h_ns) and (m * 10 + n) > 0:
            lines.append(f"AFIX {m * 10 + n}")
        if atom_site_loop["_atom_site.qcrbox_calc_uiso_multiplier"][int_index] != ".":
            uiso_mult = float(atom_site_loop["_atom_site.qcrbox_calc_uiso_multiplier"][int_index])
        else:
            uiso_mult = None
        lines.append(create_atom_string(int_index, atom_site_loop, atom_site_aniso_loop, uiso_mult))
        if label in attached_collect:
            lines.extend(
                create_atom_table_lines(
                    attached_collect[label], cif_block, atom_site_loop, atom_site_aniso_loop, attached_collect, m
                )
            )
            if parent_afix_m is not None:
                lines.append(f"AFIX {parent_afix_m}5")
            else:
                lines.append("AFIX 0")
    return lines


def create_atom_list(cif_block: block) -> str:
    """
    Create the atom list section of a SHELX instruction file from CIF data.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing atom information.

    Returns
    -------
    str
        Atom list section of SHELX instruction file.
    """
    atom_site_loop = cif_block.loops["_atom_site"]
    atom_site_aniso_loop = cif_block.loops["_atom_site_aniso"]
    (non_afix_indexes,) = np.where(np.array(atom_site_loop["_atom_site.calc_attached_atom"]) == ".")
    non_afix_indexes = non_afix_indexes.tolist()

    attached_collect = {}
    psn_id = atom_site_loop["_atom_site.qcrbox_constraint_posn_id"]

    for attached_atom in np.unique(atom_site_loop["_atom_site.calc_attached_atom"]):
        if attached_atom == ".":
            continue
        indexes = np.nonzero(atom_site_loop["_atom_site.calc_attached_atom"] == attached_atom)[0]
        assert all(
            (psn_id[int(i)] == psn_id[int(indexes[0])] for i in indexes[1:])
        ), f"not all constrain posn ids are equal for {attached_atom}"
        attached_collect[attached_atom] = list(
            sorted(indexes, key=lambda x: atom_site_loop["_atom_site.qcrbox_constraint_posn_index"][int(x)])
        )

    body = create_atom_table_lines(non_afix_indexes, cif_block, atom_site_loop, atom_site_aniso_loop, attached_collect)

    return "\n".join(body)


def cif2ins(cif_block: block) -> str:
    """
    Convert a CIF block to a complete SHELX instruction file.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing full structure information.

    Returns
    -------
    str
        Complete SHELX instruction file content.
    """
    ins_lines = [create_header(cif_block), create_atom_list(cif_block), "HKLF 4", "", "END", ""]
    return "\n".join(ins_lines)
