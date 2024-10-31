from collections import namedtuple
from typing import List, Tuple, Set, Dict

import numpy as np
import trimesh
from cctbx.crystal.distance_based_connectivity import build_simple_two_way_bond_sets
from iotbx import cif
from scitbx.array_family import flex

from ..cif.entries.entry_conversion import block_to_specific_keywords

ElementValues = namedtuple("ElementValues", ["radius", "atom_colour", "ring_colour"])

element_dict = {
    'H': ElementValues(radius=0.31, atom_colour='#ffffff', ring_colour='#000000'),
    'He': ElementValues(radius=0.28, atom_colour='#d9ffff', ring_colour='#000000'),
    'Li': ElementValues(radius=1.28, atom_colour='#cc80ff', ring_colour='#000000'),
    'Be': ElementValues(radius=0.96, atom_colour='#c2ff00', ring_colour='#000000'),
    'B': ElementValues(radius=0.85, atom_colour='#ffb5b5', ring_colour='#000000'),
    'C': ElementValues(radius=0.76, atom_colour='#000000', ring_colour='#ffffff'),
    'N': ElementValues(radius=0.71, atom_colour='#3050f8', ring_colour='#ffffff'),
    'O': ElementValues(radius=0.66, atom_colour='#ff0d0d', ring_colour='#ffffff'),
    'F': ElementValues(radius=0.57, atom_colour='#90e050', ring_colour='#000000'),
    'Ne': ElementValues(radius=0.58, atom_colour='#b3e3f5', ring_colour='#000000'),
    'Na': ElementValues(radius=1.66, atom_colour='#ab5cf2', ring_colour='#ffffff'),
    'Mg': ElementValues(radius=1.41, atom_colour='#8aff00', ring_colour='#000000'),
    'Al': ElementValues(radius=1.21, atom_colour='#bfa6a6', ring_colour='#000000'),
    'Si': ElementValues(radius=1.11, atom_colour='#f0c8a0', ring_colour='#000000'),
    'P': ElementValues(radius=1.07, atom_colour='#ff8000', ring_colour='#000000'),
    'S': ElementValues(radius=1.05, atom_colour='#ffff30', ring_colour='#000000'),
    'Cl': ElementValues(radius=1.02, atom_colour='#1ff01f', ring_colour='#000000'),
    'Ar': ElementValues(radius=1.06, atom_colour='#80d1e3', ring_colour='#000000'),
    'K': ElementValues(radius=2.03, atom_colour='#8f40d4', ring_colour='#ffffff'),
    'Ca': ElementValues(radius=1.76, atom_colour='#3dff00', ring_colour='#000000'),
    'Sc': ElementValues(radius=1.7, atom_colour='#e6e6e6', ring_colour='#000000'),
    'Ti': ElementValues(radius=1.6, atom_colour='#bfc2c7', ring_colour='#000000'),
    'V': ElementValues(radius=1.53, atom_colour='#a6a6ab', ring_colour='#000000'),
    'Cr': ElementValues(radius=1.39, atom_colour='#8a99c7', ring_colour='#000000'),
    'Mn': ElementValues(radius=1.39, atom_colour='#9c7ac7', ring_colour='#000000'),
    'Fe': ElementValues(radius=1.32, atom_colour='#e06633', ring_colour='#ffffff'),
    'Co': ElementValues(radius=1.26, atom_colour='#f090a0', ring_colour='#000000'),
    'Ni': ElementValues(radius=1.24, atom_colour='#50d050', ring_colour='#000000'),
    'Cu': ElementValues(radius=1.32, atom_colour='#c88033', ring_colour='#000000'),
    'Zn': ElementValues(radius=1.22, atom_colour='#7d80b0', ring_colour='#000000'),
    'Ga': ElementValues(radius=1.22, atom_colour='#c28f8f', ring_colour='#000000'),
    'Ge': ElementValues(radius=1.2, atom_colour='#668f8f', ring_colour='#000000'),
    'As': ElementValues(radius=1.19, atom_colour='#bd80e3', ring_colour='#000000'),
    'Se': ElementValues(radius=1.2, atom_colour='#ffa100', ring_colour='#000000'),
    'Br': ElementValues(radius=1.2, atom_colour='#a62929', ring_colour='#ffffff'),
    'Kr': ElementValues(radius=1.16, atom_colour='#5cb8d1', ring_colour='#000000'),
    'Rb': ElementValues(radius=2.2, atom_colour='#702eb0', ring_colour='#ffffff'),
    'Sr': ElementValues(radius=1.95, atom_colour='#00ff00', ring_colour='#000000'),
    'Y': ElementValues(radius=1.9, atom_colour='#94ffff', ring_colour='#000000'),
    'Zr': ElementValues(radius=1.75, atom_colour='#94e0e0', ring_colour='#000000'),
    'Nb': ElementValues(radius=1.64, atom_colour='#73c2c9', ring_colour='#000000'),
    'Mo': ElementValues(radius=1.54, atom_colour='#54b5b5', ring_colour='#000000'),
    'Tc': ElementValues(radius=1.47, atom_colour='#3b9e9e', ring_colour='#000000'),
    'Ru': ElementValues(radius=1.46, atom_colour='#248f8f', ring_colour='#000000'),
    'Rh': ElementValues(radius=1.42, atom_colour='#0a7d8c', ring_colour='#000000'),
    'Pd': ElementValues(radius=1.39, atom_colour='#006985', ring_colour='#ffffff'),
    'Ag': ElementValues(radius=1.45, atom_colour='#c0c0c0', ring_colour='#000000'),
    'Cd': ElementValues(radius=1.44, atom_colour='#ffd98f', ring_colour='#000000'),
    'In': ElementValues(radius=1.42, atom_colour='#a67573', ring_colour='#000000'),
    'Sn': ElementValues(radius=1.39, atom_colour='#668080', ring_colour='#000000'),
    'Sb': ElementValues(radius=1.39, atom_colour='#9e63b5', ring_colour='#ffffff'),
    'Te': ElementValues(radius=1.38, atom_colour='#d47a00', ring_colour='#000000'),
    'I': ElementValues(radius=1.39, atom_colour='#940094', ring_colour='#ffffff'),
    'Xe': ElementValues(radius=1.4, atom_colour='#429eb0', ring_colour='#000000'),
    'Cs': ElementValues(radius=2.44, atom_colour='#57178f', ring_colour='#ffffff'),
    'Ba': ElementValues(radius=2.15, atom_colour='#00c900', ring_colour='#000000'),
    'La': ElementValues(radius=2.07, atom_colour='#70d4ff', ring_colour='#000000'),
    'Ce': ElementValues(radius=2.04, atom_colour='#ffffc7', ring_colour='#000000'),
    'Pr': ElementValues(radius=2.03, atom_colour='#d9ffc7', ring_colour='#000000'),
    'Nd': ElementValues(radius=2.01, atom_colour='#c7ffc7', ring_colour='#000000'),
    'Pm': ElementValues(radius=1.99, atom_colour='#a3ffc7', ring_colour='#000000'),
    'Sm': ElementValues(radius=1.98, atom_colour='#8fffc7', ring_colour='#000000'),
    'Eu': ElementValues(radius=1.98, atom_colour='#61ffc7', ring_colour='#000000'),
    'Gd': ElementValues(radius=1.96, atom_colour='#45ffc7', ring_colour='#000000'),
    'Tb': ElementValues(radius=1.94, atom_colour='#30ffc7', ring_colour='#000000'),
    'Dy': ElementValues(radius=1.92, atom_colour='#1fffc7', ring_colour='#000000'),
    'Ho': ElementValues(radius=1.92, atom_colour='#00ff9c', ring_colour='#000000'),
    'Er': ElementValues(radius=1.89, atom_colour='#00e675', ring_colour='#000000'),
    'Tm': ElementValues(radius=1.9, atom_colour='#00d452', ring_colour='#000000'),
    'Yb': ElementValues(radius=1.87, atom_colour='#00bf38', ring_colour='#000000'),
    'Lu': ElementValues(radius=1.87, atom_colour='#00ab24', ring_colour='#000000'),
    'Hf': ElementValues(radius=1.75, atom_colour='#4dc2ff', ring_colour='#000000'),
    'Ta': ElementValues(radius=1.7, atom_colour='#4da6ff', ring_colour='#000000'),
    'W': ElementValues(radius=1.62, atom_colour='#2194d6', ring_colour='#000000'),
    'Re': ElementValues(radius=1.51, atom_colour='#267dab', ring_colour='#000000'),
    'Os': ElementValues(radius=1.44, atom_colour='#266696', ring_colour='#ffffff'),
    'Ir': ElementValues(radius=1.41, atom_colour='#175487', ring_colour='#ffffff'),
    'Pt': ElementValues(radius=1.36, atom_colour='#d0d0e0', ring_colour='#000000'),
    'Au': ElementValues(radius=1.36, atom_colour='#ffd123', ring_colour='#000000'),
    'Hg': ElementValues(radius=1.32, atom_colour='#b8b8d0', ring_colour='#000000'),
    'Tl': ElementValues(radius=1.45, atom_colour='#a6544d', ring_colour='#ffffff'),
    'Pb': ElementValues(radius=1.46, atom_colour='#575961', ring_colour='#ffffff'),
    'Bi': ElementValues(radius=1.48, atom_colour='#9e4fb5', ring_colour='#ffffff'),
    'Po': ElementValues(radius=1.4, atom_colour='#ab5c00', ring_colour='#ffffff'),
    'At': ElementValues(radius=1.5, atom_colour='#754f45', ring_colour='#ffffff'),
    'Rn': ElementValues(radius=1.5, atom_colour='#428296', ring_colour='#000000'),
    'Fr': ElementValues(radius=2.6, atom_colour='#420066', ring_colour='#ffffff'),
    'Ra': ElementValues(radius=2.21, atom_colour='#007d00', ring_colour='#000000'),
    'Ac': ElementValues(radius=2.15, atom_colour='#70abfa', ring_colour='#000000'),
    'Th': ElementValues(radius=2.06, atom_colour='#00baff', ring_colour='#000000'),
    'Pa': ElementValues(radius=2.0, atom_colour='#00a1ff', ring_colour='#000000'),
    'U': ElementValues(radius=1.96, atom_colour='#008fff', ring_colour='#000000'),
    'Np': ElementValues(radius=1.9, atom_colour='#0080ff', ring_colour='#000000'),
    'Pu': ElementValues(radius=1.87, atom_colour='#006bff', ring_colour='#ffffff'),
    'Am': ElementValues(radius=1.8, atom_colour='#545cf2', ring_colour='#ffffff'),
    'Cm': ElementValues(radius=1.69, atom_colour='#785ce3', ring_colour='#ffffff')
}

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

ElementGeometry = namedtuple("ElementGeometry", ["sphere", "ring"])

def generate_geometries(elements: Set[str]) -> Dict[str, ElementGeometry]:
    """
    Generate the geometries to be reused for every individual ellipsoid, ensures that every object is only generated
    once per element and therefore saves memory in the final output
    
    Parameters
    ----------
    elements: Set[str]
    Set of unique elements that are contained in the structure and therefore have to be created.

    Returns
    -------
    Dict[str, ElementGeometry]
    Element names are the keys, values are ElementGeometry namedtuples with a sphere and ring attribute for every element,
    containing a trimesh.Trimesh object with the geometries respectively
    """
    rings = {}
    element_geometries = {}
    for element in elements:
        element_values = element_dict[element]
        visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=trimesh.visual.color.hex_to_rgba(element_values.atom_colour)
        ))
        sphere = trimesh.creation.icosphere(radius=1, visual=visual)
        if element_values.ring_colour not in rings:
            ring_visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=trimesh.visual.color.hex_to_rgba(element_values.ring_colour)
            ))
            rings[element_values.ring_colour] = trimesh.creation.torus(major_radius=1, minor_radius=0.05, minor_sections=10, visual=ring_visual)
        element_geometries[element] = ElementGeometry(sphere, rings[element_values.ring_colour])
    return element_geometries


def create_scene(
    block: cif.model.block,
    bonds_used: str = 'cctbx',
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

    # This transformation assures unified cif can be used
    block = block_to_specific_keywords(
        block=block,
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
        custom_categories=[]
    )

    name = "qcrbox"
    new_model = cif.model.cif()
    new_model[name] = block
    structure = cif.cctbx_data_structures_from_cif(cif_model=new_model).xray_structures[name]

    unit_cell = structure.crystal_symmetry().unit_cell()

    xyz_carts = np.array(structure.sites_cart())
    xyz_carts -= xyz_carts.mean(axis=0)

    scene = trimesh.Scene()

    elements = set(scatterer.element_symbol() for scatterer in structure.scatterers())
    element_geometries = generate_geometries(elements)

    for xyz_cart, scatterer in zip(xyz_carts, structure.scatterers()):
        uij_cart = scatterer.u_cart_plus_u_iso(unit_cell)
        geometry = element_geometries[scatterer.element_symbol()]
        if scatterer.u_star[0] != -1.0:
            ell_rot = calc_ellipsoid_matrix(uij_cart)

            # Create a 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = ell_rot
            transform[:3, 3] = xyz_cart
            
            scene.add_geometry(geometry.sphere, node_name=scatterer.label, transform=transform)
            # Add ring for anisotropic atoms
            scene.add_geometry(geometry.ring, node_name=f"{scatterer.label}_ring", transform=transform)

            # Ring 2 (XZ plane)
            rot_x = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            scene.add_geometry(geometry.ring, node_name=f"{scatterer.label}_ring2", transform=transform @ rot_x)

            # Ring 3 (YZ plane)
            rot_y = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
            scene.add_geometry(geometry.ring, node_name=f"{scatterer.label}_ring3", transform=transform @ rot_y)

        elif scatterer.element_symbol() == "H":
            transform = np.eye(4)
            transform[:3, :3] *= 0.15
            transform[:3, 3] = xyz_cart
            scene.add_geometry(geometry.sphere, node_name=scatterer.label, transform=transform)
        else:
            ell_rot = calc_ellipsoid_matrix(uij_cart)
            transform = np.eye(4)
            transform[:3, :3] *= np.linalg.norm(ell_rot, axis=1)[0]
            transform[:3, 3] = xyz_cart
            scene.add_geometry(geometry.sphere, node_name=scatterer.label, transform=transform)

    labels = [scatterer.label for scatterer in structure.scatterers()]
    if bonds_used == 'cif':
        atom_labels1 = list(block['_geom_bond_atom_site_label_1'])
        atom_labels2 = list(block['_geom_bond_atom_site_label_2'])
        atom_symms = list(block['_geom_bond_site_symmetry_2'])
        bonds = [
            (labels.index(l1), labels.index(l2)) for l1, l2, symm in zip(atom_labels1, atom_labels2, atom_symms)
            if symm == '.'
        ]
    elif bonds_used == 'cctbx':
        disorder_groups = block.get("_atom_site_disorder_group", ["."] * len(block["_atom_site_fract_x"]))
        bonds = get_unique_bonds(structure, disorder_groups)

    visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[100, 100, 100, 255]
    ))
    bond_geometry = trimesh.creation.cylinder(radius=0.04, height=1, visual=visual)

    for index1, index2 in bonds:
        mean_position, length, rot = calc_bond_length_rot(xyz_carts[index1], xyz_carts[index2])
        stretch_z = np.eye(4)
        stretch_z[2,2] = length
        transform = np.eye(4)
        transform[:3, :3] = rot.T
        transform[:3, 3] = mean_position
        scene.add_geometry(bond_geometry, node_name=f"bond_{labels[index1]}_{labels[index2]}", transform=transform @ stretch_z)

    return scene
