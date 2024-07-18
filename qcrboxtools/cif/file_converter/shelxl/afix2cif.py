import re
from collections import namedtuple
from textwrap import wrap

from iotbx.cif.model import loop, block

from ...merge import merge_cif_loops
from . import shelxl_commands

class CifInsAtomsMismatchError(Exception):
    pass

class MissingRefineInstructionsError(Exception):
    pass

non_implemented_instructions = (
    "ABIN",
    "ANSC",
    "ANSR",
    "BASF",
    "BLOC",
    "BUMP",
    "CHIV",
    "DAMP",
    "DANG",
    "DEFS",
    "DELU",
    "DFIX",
    "DISP",
    "EADP",
    "EXTI",
    "EXYZ",
    "FEND",
    "FLAT",
    "FRAG",
    "HFIX",
    "ISOR",
    "LAUE",
    "MOVE",
    "NCSY",
    "NEUT",
    "OMIT",
    "PART",
    "PRIG",
    "RESI",
    "RIGU",
    "SADI",
    "SAME",
    "SHEL",
    "SIMU",
    "STIR",
    "SUMP",
    "SWAT",
    "TWIN",
    "TWST",
    "WIGL",
    "XNPD",
)
def keep_ins(ins_string):
    lines = ins_string.split("\n")
    return any(line.startswith(command) for command in non_implemented_instructions for line in lines)

def afix2mn(afix_code: int):
    return afix_code // 10, afix_code % 10


def afix_line2mn(afix_line: str):
    afix_code = int(afix_line.split()[1])
    return afix2mn(afix_code)


def afix_m2cif(afix_m: int):
    afix_m_cif_instructions = {
        0: "Relative posistioning of the atoms was kept fixed.",
        1: "Idealized tertiary C-H with all equal X-C-H angles for all three substituents of C.",
        2: "Idealized secondary CH2 with equal X-C-H and Y-C-H angles and H-C-H adapted to X-C-Y",
        3: (
            "Idealized CH3 group with tetrahedral angles, staggered with respect to the shortest bond to the attached "
            + "atom."
        ),
        4: "Aromatic C-H or amide N-H with hydrogen on the external bisector of the X-C-Y or X-N-Y angle.",
        5: "Atoms are fitted to a regular pentagon",
        6: "Atoms are fitted to a regular hexagon",
        7: "Atoms are fitted to a regular hexagon",
        8: (
            "Idealized OH group with tetrahedral X-O-H angle, choosing hydrogen position based on best hydrogen "
            + "bonding."
        ),
        9: "Idealized terminal X=CH2 or X=NH2+ with hydrogens in the plane of the nearest substituent.",
        10: (
            "Atoms are fitted to generate an idealised pentamethylcyclopentadienyl anion. "
            + "Atoms with _atom_site.qcrbox_constraint_posn_index 1 to 5 form the cyclopentadienyl group "
            + "while atoms with _atom_site.qcrbox_constraint_posn_index are the methyl groups."
        ),
        11: (
            "Atoms are fitted to generate an idealised napthalene molecule. The values for "
            + "_atom_site.qcrbox_constraint_posn_index follow a symmetrical figure of eight "
            + "starting with the alpha and then the beta carbon atoms."
        ),
        12: "Idealized disordered methyl group with two positions rotated by 60 degrees.",
        13: (
            "Idealized CH3 group with tetrahedral angles. The atom position with "
            + "_atom_site.qcrbox_constraint_posn_index 1 defines the torsion angle."
        ),
        14: (
            "Idealized OH group with tetrahedral X-O-H angle. The atom position with "
            + "_atom_site.qcrbox_constraint_posn_index 1 defines the torsion angle."
        ),
        15: "BH group with hydrogen placed along the negative sum vector of unit vectors of the other bonds to boron.",
        16: "Acetylenic C-H with linear X-C-H.",
    }
    try:
        return "\n".join(wrap(afix_m_cif_instructions[afix_m]))
    except KeyError as e:
        raise KeyError(f"AFIX with m={afix_m} is not implemented") from e


def afix_n2cif(afix_n: int):
    # unsupported 0 2 (no position constrains) 5 (rigid group continuation),
    # R: Rigid group, D: Distances, O:Orientation, T: Torsion
    afix_n_cif = {1: ".", 3: "R", 4: "RD", 6: "RO", 7: "RT", 8: "RDT", 9: "RDO"}
    try:
        return afix_n_cif[afix_n]
    except KeyError as e:
        raise KeyError(f"AFIX with n={afix_n} is not implemented") from e


def ins2atom_site_dicts(ins_string: str):
    ins_string = ins_string.replace("=\n", " ")
    ins_string = re.sub(r"TITL.*(\n  .*)*", "", ins_string)
    ins_lines = [line.strip() for line in ins_string.split("\n") if len(line) > 0]
    sfac_line = next(line for line in ins_lines if line.startswith("SFAC")).upper().split()

    atom_lines = [line for line in ins_lines[: ins_lines.index("END")] if not line.upper().startswith(shelxl_commands)]
    atom_site_collect = {
        "_atom_site.label": [],
        "_atom_site.fract_x": [],
        "_atom_site.fract_y": [],
        "_atom_site.fract_z": [],
    }

    atom_site_iso_collect = {"_atom_site.label": [], "_atom_site.u_iso_or_equiv": []}

    atom_site_aniso_collect = {
        "_atom_site_aniso.label": [],
        "_atom_site_aniso.u_11": [],
        "_atom_site_aniso.u_22": [],
        "_atom_site_aniso.u_33": [],
        "_atom_site_aniso.u_23": [],
        "_atom_site_aniso.u_13": [],
        "_atom_site_aniso.u_12": [],
    }

    for atom_line in atom_lines:
        content = atom_line.split()
        element = sfac_line[int(content[1])].capitalize()
        if content[0].upper().startswith(element.upper()):
            content[0] = element + content[0][len(element):]
        atom_site_collect["_atom_site.label"].append(content[0])
        atom_site_collect["_atom_site.fract_x"].append(float(content[2]))
        atom_site_collect["_atom_site.fract_y"].append(float(content[3]))
        atom_site_collect["_atom_site.fract_z"].append(float(content[4]))

        if len(content) == 12:
            atom_site_aniso_collect["_atom_site_aniso.label"].append(content[0])
            atom_site_aniso_collect["_atom_site_aniso.u_11"].append(float(content[6]))
            atom_site_aniso_collect["_atom_site_aniso.u_22"].append(float(content[7]))
            atom_site_aniso_collect["_atom_site_aniso.u_33"].append(float(content[8]))
            atom_site_aniso_collect["_atom_site_aniso.u_23"].append(float(content[9]))
            atom_site_aniso_collect["_atom_site_aniso.u_13"].append(float(content[10]))
            atom_site_aniso_collect["_atom_site_aniso.u_12"].append(float(content[11]))

        elif len(content) == 7 and float(content[6]) >= 0:
            atom_site_iso_collect["_atom_site.label"].append(content[0])
            atom_site_iso_collect["_atom_site.u_iso_or_equiv"].append(content[6])

    return atom_site_collect, atom_site_iso_collect, atom_site_aniso_collect


def create_atom_site_constraints(ins_string: str):
    ins_string = ins_string.replace("=\n", " ")
    ins_string = re.sub(r"TITL.*(\n  .*)*", "", ins_string)
    lines = list(line for line in ins_string.split("\n") if len(line.strip()) > 0)
    atom_table_start = max(index for index, line in enumerate(lines) if line.upper().startswith("FVAR ")) + 1
    atom_table_end = next(index for index, line in enumerate(lines) if line.upper().startswith("HKLF"))

    non_afix = list(shelxl_commands)
    non_afix.remove("AFIX")
    non_afix = tuple(non_afix)

    sfac_line = next(line for line in lines if line.startswith("SFAC")).upper().split()
    h_index = sfac_line.index("H")

    only_atoms_afix = [line for line in lines[atom_table_start:atom_table_end] if not line.upper().startswith(non_afix)]

    Afix = namedtuple("Afix", ["m", "n"])
    connect_atoms = []
    afixes = []
    afix_counts = []
    non_h_ns = (0, 1, 2, 5, 6, 9)
    non_h_ms = (5, 6, 7, 10, 11)

    atom_site_collect_dict = {
        "_atom_site.label": [],
        "_atom_site.calc_attached_atom": [],
        "_atom_site.qcrbox_constraint_posn_id": [],
        "_atom_site.qcrbox_constraint_posn_index": [],
        "_atom_site.qcrbox_calc_uiso_multiplier": [],
    }

    constraint_posn_dict = {
        "_qcrbox_constraint_posn.id": [],
        "_qcrbox_constraint_posn.refined_pars": [],
        "_qcrbox_constraint_posn.instruction": [],
    }

    current_name = ""
    for line in only_atoms_afix:
        if line.startswith("AFIX"):
            afix = Afix(*afix_line2mn(line))
            # the non-hydrogen ns also end the previous group
            if afix.n in non_h_ns:
                if len(afixes) > 0:
                    afixes.pop(-1)
                    afix_counts.pop(-1)
                    connect_atoms.pop(-1)
            if afix.n not in (0, 5):
                afixes.append(afix)
                afix_counts.append(1)
        else:
            # atom line everything else was filtered before
            current_name = line.split()[0]
            sfac_index = int(line.split()[1])
            if len(line.split()) == 7 and float(line.split()[6]) < 0:
                multiplier = -float(line.split()[6])
                atom_site_collect_dict["_atom_site.qcrbox_calc_uiso_multiplier"].append(f"{multiplier:.3f}")
            else:
                atom_site_collect_dict["_atom_site.qcrbox_calc_uiso_multiplier"].append(".")
            is_h = sfac_index == h_index
            element = sfac_line[sfac_index].capitalize()
            if current_name.upper().startswith(element.upper()):
                current_name = element + current_name[len(element) :]
            atom_site_collect_dict["_atom_site.label"].append(current_name)
            if len(afixes) == 0:
                atom_site_collect_dict["_atom_site.calc_attached_atom"].append(".")
                atom_site_collect_dict["_atom_site.qcrbox_constraint_posn_id"].append(".")
                atom_site_collect_dict["_atom_site.qcrbox_constraint_posn_index"].append(".")
                last_name = current_name
                continue

            rigid_group = all(
                (
                    afixes[-1].m in non_h_ms,
                    afixes[-1].n in non_h_ns,
                )
            )
            if rigid_group and is_h:
                atom_site_collect_dict["_atom_site.calc_attached_atom"].append(".")
                atom_site_collect_dict["_atom_site.qcrbox_constraint_posn_id"].append(".")
                atom_site_collect_dict["_atom_site.qcrbox_constraint_posn_index"].append(".")
                continue
            if rigid_group and afix_counts[-1] == 1:
                connect_atoms.append(current_name)
                atom_site_collect_dict["_atom_site.calc_attached_atom"].append(".")
            elif not rigid_group and afix_counts[-1] == 1:
                connect_atoms.append(last_name)
                atom_site_collect_dict["_atom_site.calc_attached_atom"].append(connect_atoms[-1])
            else:
                atom_site_collect_dict["_atom_site.calc_attached_atom"].append(connect_atoms[-1])

            constr_id = f"SXL{afixes[-1].m}{afixes[-1].n}"

            atom_site_collect_dict["_atom_site.qcrbox_constraint_posn_id"].append(constr_id)
            atom_site_collect_dict["_atom_site.qcrbox_constraint_posn_index"].append(str(afix_counts[-1]))
            afix_counts[-1] += 1
            if constr_id not in constraint_posn_dict["_qcrbox_constraint_posn.id"]:
                constraint_posn_dict["_qcrbox_constraint_posn.id"].append(constr_id)
                constraint_posn_dict["_qcrbox_constraint_posn.instruction"].append(afix_m2cif(afix.m))
                constraint_posn_dict["_qcrbox_constraint_posn.refined_pars"].append(afix_n2cif(afix.n))
            last_name = current_name
    return atom_site_collect_dict, constraint_posn_dict


def afix_to_cif(cif_block: block):
    ins_string = cif_block.get("_iucr.refine_instructions_details", cif_block.get("_shelx.res_file"))
    if ins_string is None:
        raise MissingRefineInstructionsError(
            "No refine instructions (_iucr.refine_instructions_details, _shelx.res_file) found in CIF block"
        )

    atom_posn_dict, atom_iso_dict, atom_aniso_dict = ins2atom_site_dicts(ins_string)
    atom_site_posn_loop = loop()
    atom_site_posn_loop.add_columns(atom_posn_dict)
    atom_site_uiso_loop = loop()
    atom_site_uiso_loop.add_columns(atom_iso_dict)

    atom_site_const_dict, constr_posn_dict = create_atom_site_constraints(ins_string)
    atom_site_const_loop = loop()
    atom_site_const_loop.add_columns(atom_site_const_dict)

    constr_posn_loop = loop()
    constr_posn_loop.add_columns(constr_posn_dict)

    atom_site_loop = merge_cif_loops(
        cif_block.loops["_atom_site"], atom_site_posn_loop, merge_on=[r"_atom_site\.label"]
    )
    atom_site_loop = merge_cif_loops(atom_site_loop, atom_site_uiso_loop, merge_on=[r"_atom_site\.label"])
    atom_site_loop = merge_cif_loops(atom_site_loop, atom_site_const_loop, merge_on=[r"_atom_site\.label"])

    atom_site_aniso_loop = loop()
    atom_site_aniso_loop.add_columns(atom_aniso_dict)
    atom_site_aniso_loop = merge_cif_loops(
        cif_block.loops["_atom_site_aniso"], atom_site_aniso_loop, merge_on=[r"_atom_site_aniso\.label"]
    )
    try:
        cif_block.loops["_atom_site"].update(atom_site_loop)
    except AssertionError as e:
        raise CifInsAtomsMismatchError(
            "The atoms in the INS file do not match the ones in the atom_site_table"
            ) from e
    try:
        cif_block.loops["_atom_site_aniso"].update(atom_site_aniso_loop)
    except AssertionError as e:
        raise CifInsAtomsMismatchError(
            "The atoms in the INS file do not match the ones in the atom_site_aniso_table"
            ) from e
    cif_block.add_loop(constr_posn_loop)

    scaling = re.search(r"FVAR\s+(\d+\.\d+)", ins_string).group(1)
    cif_block.add_data_item("_qcrbox.shelx.scale_factor", scaling)
    if not keep_ins(ins_string) and "_iucr.refine_instructions_details" in cif_block:
        del(cif_block["_iucr.refine_instructions_details"])
    elif not keep_ins(ins_string) and "_shelx.res_file" in cif_block:
        del(cif_block["_shelx.res_file"])
    return cif_block
