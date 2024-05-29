from textwrap import dedent

import numpy as np
import pytest

from qcrboxtools.cif.file_converter.shelxt import (
    ins2symm_cards_and_latt,
    ins2symop_loop,
    symm_cards_and_latt2symm_mat_vecs,
    symm_mat_vec2cifsymop,
    symm_to_matrix_vector,
)


@pytest.mark.parametrize(
    "instruction,expected_matrix,expected_vector",
    [
        # Testing basic symmetry operations
        ("-X, Y, Z", np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0])),
        ("x, y, z", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0])),
        # Testing with translation vectors
        ("X, Y, 0.5+Z", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0.5])),
        ("-x, -y, -0.5+z", np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), np.array([0, 0, -0.5])),
        # Testing with fractional translations
        ("-X, 1/2+Y, Z", np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0.5, 0])),
        ("x, -1/2+y, z", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, -0.5, 0])),
        # Testing with whole number translation
        ("X, Y, 1+Z", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 1])),
        # Testing with mixed fractional coefficients for matrix
        ("1/2*X, -1/3*Y, Z", np.array([[0.5, 0, 0], [0, -1 / 3, 0], [0, 0, 1]]), np.array([0, 0, 0])),
        ("0.25*x, 0.75*y, -1*z", np.array([[0.25, 0, 0], [0, 0.75, 0], [0, 0, -1]]), np.array([0, 0, 0])),
        ("1/2X, -1/3Y, Z", np.array([[0.5, 0, 0], [0, -1 / 3, 0], [0, 0, 1]]), np.array([0, 0, 0])),
        ("0.25x, 0.75y, -z", np.array([[0.25, 0, 0], [0, 0.75, 0], [0, 0, -1]]), np.array([0, 0, 0])),
        # Testing uppercase and lowercase mix
        ("X, y, Z", np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0])),
        ("-x, -Y, -z", np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), np.array([0, 0, 0])),
    ],
)
def test_symm_to_matrix_vector(instruction, expected_matrix, expected_vector):
    matrix, vector = symm_to_matrix_vector(instruction)
    np.testing.assert_array_almost_equal(matrix, expected_matrix)
    np.testing.assert_array_almost_equal(vector, expected_vector)


@pytest.mark.parametrize(
    "symm_cards,latt,expected_matrices,expected_vectors",
    [
        # Basic test with simple identity transformation
        (["x, y, z"], -1, np.array([np.eye(3), np.eye(3)]), np.zeros((2, 3))),
        # Testing centering and inversion due to centrosymmetric structure
        (
            [],
            2,
            np.array([np.eye(3), -np.eye(3), np.eye(3), -np.eye(3)]),
            np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        ),
        # Expanded to test more lattice types
        (
            [],
            3,
            np.array([np.eye(3), -np.eye(3), np.eye(3), -np.eye(3), np.eye(3), -np.eye(3)]),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [-1 / 3, 1 / 3, 1 / 3],
                    [-1 / 3, 1 / 3, 1 / 3],
                    [1 / 3, -1 / 3, -1 / 3],
                    [1 / 3, -1 / 3, -1 / 3],
                ]
            ),
        ),
        # Testing with more symmetry operations
        (
            ["0.5-X,-Y,0.5+Z", "0.5+X,0.5-Y,-Z", " -X,0.5+Y,0.5-Z"],
            -1,
            np.array(
                [
                    np.eye(3),
                    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                ]
            ),
            np.array([[0, 0, 0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]),
        ),
    ],
)
def test_symm_cards_and_latt2symm_mat_vecs(symm_cards, latt, expected_matrices, expected_vectors):
    symm_mats, symm_vecs = symm_cards_and_latt2symm_mat_vecs(symm_cards, latt)
    np.testing.assert_array_almost_equal(symm_mats, expected_matrices)
    np.testing.assert_array_almost_equal(symm_vecs, expected_vectors)


@pytest.mark.parametrize(
    "symm_mat,symm_vec,expected_output",
    [
        # Test identity matrix with no translation
        (np.eye(3), np.zeros(3), "+x,+y,+z"),
        # Test inversion matrix
        (-np.eye(3), np.zeros(3), "-x,-y,-z"),
        # Test half-cell translation along z
        (np.eye(3), [0, 0, 0.5], "+x,+y,1/2+z"),
        # Test rotation and translation
        ([[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0.5, 0, 0], "1/2+y,+x,+z"),
        # Test fraction in the matrix
        ([[0.5, 0, 0], [0, -0.5, 0], [0, 0, 1]], [0, 0, 0], "+1/2*x,-1/2*y,+z"),
        # Test very small non-zero values that should be considered as zero
        ([[1e-11, 1, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0], "+y,+y,+z"),
    ],
)
def test_symm_mat_vec2cifsymop(symm_mat, symm_vec, expected_output):
    result = symm_mat_vec2cifsymop(np.array(symm_mat), np.array(symm_vec))
    assert result == expected_output


@pytest.fixture(name="ins_file")
def fixture_ins_file(tmp_path):
    ins_file = tmp_path / "test.ins"
    ins_file.write_text(
        dedent("""\
        TITL Ylid_Mo_RT_autored in P2(1)2(1)2(1)
        REM P2(1)2(1)2(1) (#19 in standard setting)
        CELL 0.71073   5.958295   9.039438  18.394640  90.0000  90.0000  90.0000
        ZERR    1.00   0.000261   0.000456   0.000923   0.0000   0.0000   0.0000
        LATT -1
        SYMM -x+1/2,-y, z+1/2
        SYMM -x, y+1/2,-z+1/2
        SYMM  x+1/2,-y+1/2,-z
        SFAC C H O S
        UNIT 44.00 40.00 8.00 4.00
        REM  CrysAlisPro recorded range (K): Min=291.9; max=292.0; aver:292.0
        TEMP 19
        TREF
        HKLF 4
        END
    """)
    )
    return ins_file


def test_ins2symm_cards_and_latt(ins_file):
    symm_cards, latt = ins2symm_cards_and_latt(ins_file)

    expected_symm_cards = ["-x+1/2,-y, z+1/2", "-x, y+1/2,-z+1/2", "x+1/2,-y+1/2,-z"]
    expected_latt = -1

    assert symm_cards == expected_symm_cards
    assert latt == expected_latt


def test_ins2symop_loop(ins_file):
    output = ins2symop_loop(ins_file)

    # Expected CIF string, this needs to be adjusted based on actual results from the dependencies
    expected_output = dedent("""\
        loop_
          _space_group_symop.id
          _space_group_symop.operation_xyz
          1  '+x,+y,+z'
          2  '1/2-x,-y,1/2+z'
          3  '-x,1/2+y,1/2-z'
          4  '1/2+x,1/2-y,-z'
    """).strip()

    assert output.strip() == expected_output
