from collections import namedtuple
from typing import List, Tuple

import numpy as np
import trimesh
from cctbx.crystal.distance_based_connectivity import build_simple_two_way_bond_sets
from iotbx import cif
from scitbx.array_family import flex

from ..cif.entries.entry_conversion import block_to_specific_keywords

elements = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
]

radii = [
    0.31,
    0.28,
    1.28,
    0.96,
    0.85,
    0.76,
    0.71,
    0.66,
    0.57,
    0.58,
    1.66,
    1.41,
    1.21,
    1.11,
    1.07,
    1.05,
    1.02,
    1.06,
    2.03,
    1.76,
    1.70,
    1.60,
    1.53,
    1.39,
    1.39,
    1.32,
    1.26,
    1.24,
    1.32,
    1.22,
    1.22,
    1.20,
    1.19,
    1.20,
    1.20,
    1.16,
    2.20,
    1.95,
    1.90,
    1.75,
    1.64,
    1.54,
    1.47,
    1.46,
    1.42,
    1.39,
    1.45,
    1.44,
    1.42,
    1.39,
    1.39,
    1.38,
    1.39,
    1.40,
    2.44,
    2.15,
    2.07,
    2.04,
    2.03,
    2.01,
    1.99,
    1.98,
    1.98,
    1.96,
    1.94,
    1.92,
    1.92,
    1.89,
    1.90,
    1.87,
    1.87,
    1.75,
    1.70,
    1.62,
    1.51,
    1.44,
    1.41,
    1.36,
    1.36,
    1.32,
    1.45,
    1.46,
    1.48,
    1.40,
    1.50,
    1.50,
    2.60,
    2.21,
    2.15,
    2.06,
    2.00,
    1.96,
    1.90,
    1.87,
    1.80,
    1.69,
]

atom_colours = [
    "#ffffff",
    "#d9ffff",
    "#cc80ff",
    "#c2ff00",
    "#ffb5b5",
    "#000000",
    "#3050f8",
    "#ff0d0d",
    "#90e050",
    "#b3e3f5",
    "#ab5cf2",
    "#8aff00",
    "#bfa6a6",
    "#f0c8a0",
    "#ff8000",
    "#ffff30",
    "#1ff01f",
    "#80d1e3",
    "#8f40d4",
    "#3dff00",
    "#e6e6e6",
    "#bfc2c7",
    "#a6a6ab",
    "#8a99c7",
    "#9c7ac7",
    "#e06633",
    "#f090a0",
    "#50d050",
    "#c88033",
    "#7d80b0",
    "#c28f8f",
    "#668f8f",
    "#bd80e3",
    "#ffa100",
    "#a62929",
    "#5cb8d1",
    "#702eb0",
    "#00ff00",
    "#94ffff",
    "#94e0e0",
    "#73c2c9",
    "#54b5b5",
    "#3b9e9e",
    "#248f8f",
    "#0a7d8c",
    "#006985",
    "#c0c0c0",
    "#ffd98f",
    "#a67573",
    "#668080",
    "#9e63b5",
    "#d47a00",
    "#940094",
    "#429eb0",
    "#57178f",
    "#00c900",
    "#70d4ff",
    "#ffffc7",
    "#d9ffc7",
    "#c7ffc7",
    "#a3ffc7",
    "#8fffc7",
    "#61ffc7",
    "#45ffc7",
    "#30ffc7",
    "#1fffc7",
    "#00ff9c",
    "#00e675",
    "#00d452",
    "#00bf38",
    "#00ab24",
    "#4dc2ff",
    "#4da6ff",
    "#2194d6",
    "#267dab",
    "#266696",
    "#175487",
    "#d0d0e0",
    "#ffd123",
    "#b8b8d0",
    "#a6544d",
    "#575961",
    "#9e4fb5",
    "#ab5c00",
    "#754f45",
    "#428296",
    "#420066",
    "#007d00",
    "#70abfa",
    "#00baff",
    "#00a1ff",
    "#008fff",
    "#0080ff",
    "#006bff",
    "#545cf2",
    "#785ce3",
]

ring_colours = [
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#ffffff",
    "#ffffff",
    "#000000",
    "#000000",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#000000",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#000000",
    "#ffffff",
    "#000000",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#ffffff",
    "#ffffff",
    "#ffffff",
    "#ffffff",
    "#000000",
    "#ffffff",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#000000",
    "#ffffff",
    "#ffffff",
    "#ffffff",
]

ElementValues = namedtuple("ElementValues", ["radius", "atom_colour", "ring_colour"])

element_dict = {
    name: ElementValues(radius=radius, atom_colour=atom_colour, ring_colour=ring_colour)
    for name, radius, atom_colour, ring_colour in zip(elements, radii, atom_colours, ring_colours)
}


# def cif_path2dict(cif_path: str, data_block_name: str = None) -> cif.model.block:
#     """Read a CIF file and extract specified data block.

#     Parameters
#     ----------
#     cif_path : str
#         Path to the CIF file.
#     data_block_name : str, optional
#         Name of the data block to extract. If None, extracts first available
#         data block or second if first is 'global'.

#     Returns
#     -------
#     cif.model.block
#         Representation of the extracted data block.

#     Raises
#     ------
#     KeyError
#         If CIF file contains multiple data blocks and data_block_name is not
#         specified.
#     """
#     reader = cif.reader(cif_path)
#     model = reader.model()
#     if data_block_name is not None:
#         return model[data_block_name]
#     else:
#         model_iter = iter(model.items())
#         if len(model) == 1:
#             return next(model_iter)
#         elif len(model) == 2 and next(model_iter)[0] == "global":
#             return next(model_iter)
#         else:
#             raise KeyError("cif file contains multiple data_block entries, but data_block_name is not specified")


def mean_plane2(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean plane normal vector using least squares.

    Parameters
    ----------
    points : np.ndarray
        Nx3 array of point coordinates.

    Returns
    -------
    np.ndarray
        Unit normal vector of the mean plane.
    np.ndarray
        Center point of the plane.
    """
    points = points.T
    centre = points.mean(axis=0)
    centred = points - centre
    A = np.concatenate((centred[:, :2], np.ones((points.shape[0], 1))), axis=1)
    b = centred[:, 2, np.newaxis]
    vector = np.linalg.inv(np.dot(A.T, A)) @ A.T @ b
    return vector[:, 0] / np.linalg.norm(vector), centre[:, None]


def mean_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean plane normal vector using covariance matrix.

    Parameters
    ----------
    points : np.ndarray
        Nx3 array of point coordinates.

    Returns
    -------
    np.ndarray
        Unit normal vector of the mean plane.
    np.ndarray
        Center point of the plane.

    Notes
    -----
    Uses eigenvalue decomposition of the covariance matrix. Falls back to
    mean_plane2 if eigenvalue decomposition fails.
    """
    centre = points.mean(axis=0)
    centred = points - centre
    try:
        _, eigenvectors = np.linalg.eigh(np.einsum("ab, ac -> bc", centred, centred))
    except (np.linalg.LinAlgError, ValueError):
        return mean_plane2(points)
    return eigenvectors[:, 0], centre


def calc_ellipsoid_matrix(uij_cart: List[float]) -> np.ndarray:
    """Calculate transformation matrix for ellipsoid visualization.

    Parameters
    ----------
    uij_cart : list of float
        Uij values in Cartesian coordinates. Order is U11, U22, U33, U12, U13,
        U23.

    Returns
    -------
    np.ndarray
        3x3 transformation matrix for ellipsoid visualization.

    Notes
    -----
    For conversion from CIF convention, see R. W. Grosse-Kunstleve,
    J. Appl. Cryst. (2002). 35, 477-480.
    """
    uij_mat = np.array(uij_cart)[np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])]

    eigenvalues, eigenvectors = np.linalg.eig(uij_mat)
    if np.linalg.det(eigenvectors) != 1:
        eigenvectors /= np.linalg.det(eigenvectors)

    return eigenvectors @ np.diag(np.sqrt(eigenvalues))


def calc_bond_length_rot(position1: np.ndarray, position2: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Calculate cylinder transformation for bond visualization.

    Parameters
    ----------
    position1 : np.ndarray
        3D coordinates of first atom.
    position2 : np.ndarray
        3D coordinates of second atom.

    Returns
    -------
    np.ndarray
        Mean position between atoms.
    float
        Distance between atoms.
    np.ndarray
        Rotation matrix for cylinder transformation.

    Notes
    -----
    Converts a unit length cylinder aligned with y-axis to connect two atomic
    positions.
    """
    mean_position = (position1 + position2) / 2
    length = np.linalg.norm(position1 - position2)
    unit = (position2 - position1) / length
    z = np.array([0.0, 0, 1])
    v = np.cross(unit, z)
    s = np.sqrt(np.sum(v**2))
    c = np.dot(unit, z)
    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.identity(3)
    rot[:3, :3] += v_x + v_x @ v_x * (1 - c) / s**2
    return mean_position, length, rot


def get_unique_bonds(structure, disorder_groups: List[str]) -> Tuple[Tuple[int, int], ...]:
    """Get unique bonds considering disorder groups.

    Parameters
    ----------
    structure : object
        Crystal structure object containing scatterers and sites.
    disorder_groups : list of str
        List of disorder group assignments for each atom.

    Returns
    -------
    tuple of tuple
        Pairs of atom indices representing unique bonds.

    Notes
    -----
    Returns bonds where either one atom is not in a disorder group or both
    atoms are in the same disorder group.
    """
    elements = flex.std_string([scatterer.element_symbol() for scatterer in structure.scatterers()])
    bond_set = build_simple_two_way_bond_sets(structure.sites_cart(), elements)
    unique_bonds = set(tuple(sorted([i, j])) for i, bonds in enumerate(bond_set) for j in bonds)
    one_nondisorder = [
        disorder_groups[index1] == "." or disorder_groups[index2] == "." for index1, index2 in unique_bonds
    ]
    same_group = [disorder_groups[index1] == disorder_groups[index2] for index1, index2 in unique_bonds]
    return tuple(indexes for indexes, cond1, cond2 in zip(unique_bonds, one_nondisorder, same_group) if cond1 or cond2)


def create_scene(
    block: cif.model.block,
) -> trimesh.Scene:
    """Create 3D visualization of crystal structure.

    Parameters
    ----------
    block : cif.model.block
        CIF data block containing structure information.

    Returns
    -------
    trimesh.Scene
        3D scene containing structure visualization.

    Notes
    -----
    Creates a scene with atoms as spheres/ellipsoids and bonds as cylinders.
    Atoms are colored by element and bonds are shown in dark gray.
    """
    block = block_to_specific_keywords(
        compulsory_entries=[
            "_space_group_crystal_system",
            "_space_group_symop_operation_xyz",
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_U_iso_or_equiv",
            "_atom_site_adp_type",
        ],
        optional_entries=[
            "_atom_site_disorder_assembly",
            "_atom_site_disorder_group",
            "_atom_site_aniso_label",
            "_atom_site_aniso_U_11",
            "_atom_site_aniso_U_22",
            "_atom_site_aniso_U_33",
            "_atom_site_aniso_U_23",
            "_atom_site_aniso_U_13",
            "_atom_site_aniso_U_12",
            "_geom_bond_atom_site_label_1",
            "_geom_bond_atom_site_label_2",
            "_geom_bond_distance",
            "_geom_bond_site_symmetry_2",
            "_geom_bond_publ_flag",
        ],
    )

    name = "qcrbox"
    new_model = cif.model.cif()
    new_model[name] = block
    structure = cif.cctbx_data_structures_from_cif(cif_model=new_model).xray_structures[name]

    unit_cell = structure.crystal_symmetry().unit_cell()

    xyz_carts = np.array(structure.sites_cart())
    xyz_carts -= xyz_carts.mean(axis=0)

    scene = trimesh.Scene()

    for xyz_cart, scatterer in zip(xyz_carts, structure.scatterers()):
        uij_cart = scatterer.u_cart_plus_u_iso(unit_cell)
        element_values = element_dict[scatterer.element_symbol()]

        if scatterer.u_star[0] != -1.0:
            ell_rot = calc_ellipsoid_matrix(uij_cart)
            sphere = trimesh.creation.uv_sphere(radius=1)

            # Create a 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = ell_rot
            transform[:3, 3] = xyz_cart

            sphere.apply_transform(transform)
            sphere.visual.face_colors = trimesh.visual.color.hex_to_rgba(element_values.atom_colour)
            scene.add_geometry(sphere, node_name=scatterer.label)

            ring_color = trimesh.visual.color.hex_to_rgba(element_values.ring_colour)

            # Add ring for anisotropic atoms
            ring = trimesh.creation.annulus(r_min=0.95, r_max=1.05, height=0.1)
            ring.apply_transform(transform)
            ring.visual.face_colors = ring_color
            scene.add_geometry(ring, node_name=f"{scatterer.label}_ring")

            # Ring 2 (XZ plane)
            ring2 = trimesh.creation.annulus(r_min=0.95, r_max=1.05, height=0.1)
            rot_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            ring2.apply_transform(rot_x)
            ring2.apply_transform(transform)
            ring2.visual.face_colors = ring_color
            scene.add_geometry(ring2, node_name=f"{scatterer.label}_ring2")

            # Ring 3 (YZ plane)
            ring3 = trimesh.creation.annulus(r_min=0.95, r_max=1.05, height=0.1)
            rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
            ring3.apply_transform(rot_y)
            ring3.apply_transform(transform)
            ring3.visual.face_colors = ring_color
            scene.add_geometry(ring3, node_name=f"{scatterer.label}_ring3")

        elif scatterer.element_symbol() == "H":
            sphere = trimesh.creation.icosphere(radius=0.15)
            sphere.apply_translation(xyz_cart)
            sphere.visual.face_colors = trimesh.visual.color.hex_to_rgba(element_values.atom_colour)
            scene.add_geometry(sphere, node_name=scatterer.label)
        else:
            ell_rot = calc_ellipsoid_matrix(uij_cart)
            sphere = trimesh.creation.uv_sphere(radius=np.linalg.norm(ell_rot, axis=1)[0])
            sphere.apply_translation(xyz_cart)
            sphere.visual.face_colors = trimesh.visual.color.hex_to_rgba(element_values.atom_colour)
            scene.add_geometry(sphere, node_name=scatterer.label)

    labels = [scatterer.label for scatterer in structure.scatterers()]

    # if '_geom_bond_site_symmetry_2' in block:
    #    atom_labels1 = list(block['_geom_bond_atom_site_label_1'])
    #    atom_labels2 = list(block['_geom_bond_atom_site_label_2'])
    #    atom_symms = list(block['_geom_bond_site_symmetry_2'])
    #    bonds = [
    #        (labels.index(l1), labels.index(l2)) for l1, l2, symm in zip(atom_labels1, atom_labels2, atom_symms)
    #        if symm == '.'
    #    ]
    # else:
    #    distances = np.linalg.norm(xyz_carts[np.newaxis,:] - xyz_carts[:,np.newaxis], axis=-1)
    #    rads = np.array([element_dict[scatterer.element_symbol()].radius for scatterer in structure.scatterers()])
    #    sums = rads[:, np.newaxis] + rads[np.newaxis, :]
    #    atoms1, atoms2 = np.where(np.logical_and(1.15*sums > distances, distances > 0))
    #    bonds = set([tuple(sorted(val)) for val in zip(atoms1, atoms2)])

    disorder_groups = block.get("_atom_site_disorder_group", ["."] * len(block["_atom_site_fract_x"]))

    bonds = get_unique_bonds(structure, disorder_groups)

    for index1, index2 in bonds:
        mean_position, length, rot = calc_bond_length_rot(xyz_carts[index1], xyz_carts[index2])
        cylinder = trimesh.creation.cylinder(radius=0.04, height=length)
        transform = np.eye(4)
        transform[:3, :3] = rot.T
        transform[:3, 3] = mean_position
        cylinder.apply_transform(transform)
        cylinder.visual.face_colors = [100, 100, 100, 255]  # Dark gray color
        scene.add_geometry(cylinder, node_name=f"bond_{labels[index1]}_{labels[index2]}")

    return scene
