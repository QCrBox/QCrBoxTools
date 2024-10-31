import pathlib
from unittest.mock import Mock, patch

import numpy as np
import pytest
import trimesh
from iotbx.cif.model import block, loop
from scitbx.array_family import flex

from qcrboxtools.analyse.ortep import (
    ElementValues,
    calc_bond_length_rot,
    calc_ellipsoid_matrix,
    cif2ortep_glb,
    create_scene,
    element_dict,
    generate_geometries,
    get_unique_bonds,
    mean_plane,
)


# Fixture for sample points
@pytest.fixture
def sample_points():
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.1]])


# Fixture for sample CIF block
@pytest.fixture
def sample_cif_block():
    sample_block = block()
    sample_block.add_data_item("_cell_length_a", 11.0)
    sample_block.add_data_item("_cell_length_b", 12.0)
    sample_block.add_data_item("_cell_length_c", 13.0)
    sample_block.add_data_item("_cell_angle_alpha", 91.0)
    sample_block.add_data_item("_cell_angle_beta", 92.0)
    sample_block.add_data_item("_cell_angle_gamma", 93.0)
    sample_block.add_data_item("_space_group_crystal_system", "triclinic")

    mock_symm_loop = loop(data={"_space_group_symop_operation_xyz": ["x,y,z", "-x,-y,-z"]})

    sample_block.add_loop(mock_symm_loop)

    mock_atom_loop = loop(
        data={
            "_atom_site_label": ["C1", "H1", "N2"],
            "_atom_site_type_symbol": ["C", "H", "N"],
            "_atom_site_fract_x": ["0.5", "0.6", "0.7"],
            "_atom_site_fract_y": ["0.5", "0.6", "0.7"],
            "_atom_site_fract_z": ["0.5", "0.6", "0.7"],
            "_atom_site_U_iso_or_equiv": ["0.1", "0.1", "0.1"],
            "_atom_site_adp_type": ["Uiso", "Uiso", "Uani"],
            "_atom_site_disorder_group": [".", ".", "1"],
        }
    )
    sample_block.add_loop(mock_atom_loop)

    mock_aniso_loop = loop(
        data={
            "_atom_site_aniso_label": ["N2"],
            "_atom_site_aniso_U_11": ["0.01"],
            "_atom_site_aniso_U_22": ["0.01"],
            "_atom_site_aniso_U_33": ["0.01"],
            "_atom_site_aniso_U_23": ["0.0"],
            "_atom_site_aniso_U_13": ["0.0"],
            "_atom_site_aniso_U_12": ["0.0"],
        }
    )
    sample_block.add_loop(mock_aniso_loop)
    return sample_block


def test_mean_plane(sample_points):
    normal, center = mean_plane(sample_points)
    assert normal.shape == (3,)
    assert center.shape == (3,)
    assert np.allclose(np.linalg.norm(normal), 1.0)


def test_calc_ellipsoid_matrix():
    uij_cart = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]  # Identity matrix case
    matrix = calc_ellipsoid_matrix(uij_cart)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix @ matrix.T, np.eye(3))


def test_calc_bond_length_rot():
    pos1 = np.array([0.0, 0.0, 0.0])
    pos2 = np.array([0.0, 1.0, 0.0])
    mean_pos, length, rot = calc_bond_length_rot(pos1, pos2)

    assert np.allclose(mean_pos, np.array([0.0, 0.5, 0.0]))
    assert np.isclose(length, 1.0)
    assert rot.shape == (3, 3)
    assert np.allclose(np.linalg.det(rot), 1.0)


@patch("cctbx.crystal.distance_based_connectivity.build_simple_two_way_bond_sets")
def test_get_unique_bonds(mock_build_bonds):
    # Mock the bond sets
    mock_build_bonds.return_value = [[1], [0]]

    # Create mock structure
    mock_structure = Mock()
    mock_structure.scatterers.return_value = [Mock(element_symbol=lambda: "C"), Mock(element_symbol=lambda: "O")]
    mock_structure.sites_cart.return_value = flex.vec3_double([[0, 0, 0], [1, 1, 1]])

    disorder_groups = [".", "."]
    bonds = get_unique_bonds(mock_structure, disorder_groups)

    assert len(bonds) == 1
    assert bonds[0] == (0, 1)


def test_generate_geometries():
    elements = {"C", "O"}
    geometries = generate_geometries(elements)

    assert set(geometries.keys()) == elements
    for element in elements:
        assert hasattr(geometries[element], "sphere")
        assert hasattr(geometries[element], "ring")
        assert isinstance(geometries[element].sphere, trimesh.Trimesh)
        assert isinstance(geometries[element].ring, trimesh.Trimesh)


@patch("trimesh.Scene")
def test_create_scene(mock_scene, sample_cif_block):
    with patch("qcrboxtools.analyse.ortep.block_to_specific_keywords") as mock_convert:
        mock_convert.return_value = sample_cif_block

        _ = create_scene(sample_cif_block, bonds_used="cctbx")
        assert mock_scene.called
        # Add more specific assertions based on the expected scene structure


@patch("qcrboxtools.analyse.ortep.create_scene")
def test_cif2ortep_glb(mock_create_scene):
    input_path = pathlib.Path("test.cif")
    output_path = pathlib.Path("test.glb")

    mock_scene = Mock()
    mock_create_scene.return_value = mock_scene
    with patch("qcrboxtools.analyse.ortep.read_cif_as_unified") as mock_read:
        mock_read.return_value = None
        cif2ortep_glb(input_path, output_path)

    assert mock_create_scene.called
    assert mock_scene.export.called
    assert mock_scene.export.call_args[0][0] == output_path


# Test element dictionary
def test_element_dict_validity():
    for element, values in element_dict.items():
        assert isinstance(values, ElementValues)
        assert isinstance(values.radius, (int, float))
        assert isinstance(values.atom_colour, str)
        assert isinstance(values.ring_colour, str)
        assert values.atom_colour.startswith("#")
        assert values.ring_colour.startswith("#")
        assert len(values.atom_colour) == 7  # #RRGGBB format
        assert len(values.ring_colour) == 7  # #RRGGBB format


# Error cases


def test_calc_ellipsoid_matrix_invalid_input():
    with pytest.raises(ValueError):
        calc_ellipsoid_matrix([1, 2, 3])  # Not enough values


def test_generate_geometries_invalid_element():
    with pytest.raises(KeyError):
        generate_geometries({"NotAnElement"})


@pytest.mark.parametrize("bonds_used", ["cctbx", "cif"])
def test_create_scene_bonds_used(bonds_used, sample_cif_block):
    with_bonds_cif_block = sample_cif_block.deepcopy()

    bond_loop = loop(
        data={
            "_geom_bond_atom_site_label_1": ["C1"],
            "_geom_bond_atom_site_label_2": ["N2"],
            "_geom_bond_site_symmetry_2": ["."],
        }
    )

    with_bonds_cif_block.add_loop(bond_loop)
    with patch("qcrboxtools.analyse.ortep.block_to_specific_keywords") as mock_convert:
        mock_convert.return_value = with_bonds_cif_block
        scene = create_scene(with_bonds_cif_block, bonds_used=bonds_used)
        assert isinstance(scene, trimesh.Scene)


def test_create_scene_cif_bonds_missing(sample_cif_block):
    with patch("qcrboxtools.analyse.ortep.block_to_specific_keywords") as mock_convert:
        mock_convert.return_value = sample_cif_block
        with pytest.raises(KeyError):
            create_scene(sample_cif_block, bonds_used="cif")


def test_create_scene_invalid_bonds_used(sample_cif_block):
    with patch("qcrboxtools.analyse.ortep.block_to_specific_keywords") as mock_convert:
        mock_convert.return_value = sample_cif_block
        with pytest.raises(NotImplementedError):
            create_scene(sample_cif_block, bonds_used="invalid")
