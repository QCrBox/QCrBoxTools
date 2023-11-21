"""
This module contains tests for the `convergence.py` module of the `qcrboxtools` package.
It includes tests to validate the functionality of calculating differences in atomic
positions and anisotropic atomic displacement parameters (ADPs) between CIF datasets.
These tests ensure that the calculations for position differences and ADPs are accurate
and that the results conform to expected values under various scenarios.
"""

from pathlib import Path

from pytest import approx

from qcrboxtools.util.convergence import (
    position_difference, anisotropic_adp_difference, check_converged
)

def test_position_difference_diff():
    """
    Tests the position_difference function for different CIF files.

    Validates the correctness of the calculated maximum and mean absolute positions
    differences, as well as the maximum and mean positions normalized by the estimated
    standard deviation (esd). Ensures these values match the expected results.
    """
    cif1path = Path('tests/util/cif_files/difference_test1.cif')
    cif2path = Path('tests/util/cif_files/difference_test2.cif')

    diff_dict = position_difference(
        cif1path,
        0,
        cif2path,
        0
    )

    esd_from2 = (2 * 0.0003**2)**0.5

    assert diff_dict['max abs position'] == 0.08
    assert diff_dict['mean abs position'] == 0.04
    assert diff_dict['max position/esd'] == approx(0.008/esd_from2)
    assert diff_dict['mean position/esd'] == approx(0.012/9/esd_from2)


def test_position_difference_equal():
    """
    Tests the position_difference function with identical CIF files.

    Ensures that the calculated maximum and mean absolute positions, as well as their
    normalized values by the estimated standard deviation (esd), are zero when comparing
    the same CIF file to itself.
    """
    cif1path = Path('tests/util/cif_files/difference_test1.cif')

    diff_dict = position_difference(cif1path, 0, cif1path, 0)

    assert diff_dict['max abs position'] == 0.0
    assert diff_dict['mean abs position'] == 0.0
    assert diff_dict['max position/esd'] == 0.0
    assert diff_dict['mean position/esd'] == 0.0


def test_uij_difference_diff():
    """
    Tests the anisotropic_adp_difference function for different CIF files.

    Validates the correctness of the calculated maximum and mean absolute differences
    in anisotropic ADPs, as well as their normalized values by the estimated standard
    deviation (esd). Checks these values against expected results.
    """
    cif1path = Path('tests/util/cif_files/difference_test1.cif')
    cif2path = Path('tests/util/cif_files/difference_test2.cif')

    diff_dict = anisotropic_adp_difference(
        cif1path,
        0,
        cif2path,
        0
    )

    esd_from2 = (2 * 0.004**2)**0.5

    assert diff_dict['max abs uij'] == 0.008
    assert diff_dict['mean abs uij'] == 0.016 / 18
    assert diff_dict['max uij/esd'] == approx(0.008 / esd_from2)
    assert diff_dict['mean uij/esd'] == approx(0.016 / 18 / esd_from2)


def test_uij_difference_equal():
    """
    Tests the anisotropic_adp_difference function with identical CIF files.

    Ensures that the calculated maximum and mean absolute anisotropic ADPs, as well as
    their normalized values by the estimated standard deviation (esd), are zero when
    comparing the same CIF file to itself.
    """
    cif1path = Path('tests/util/cif_files/difference_test1.cif')

    diff_dict = anisotropic_adp_difference(cif1path, 0, cif1path, 0)

    assert diff_dict['max abs uij'] == approx(0.00)
    assert diff_dict['mean abs uij'] == approx(0.00)
    assert diff_dict['max uij/esd'] == approx(0.00)
    assert diff_dict['mean uij/esd'] == approx(0.00)

def test_check_convergence():
    """
    Tests the check_converged function with specified criteria.

    Ensures that the function correctly identifies when CIF datasets have converged
    based on predefined criteria for position and ADP differences. Verifies both
    scenarios where datasets are considered converged and not converged.
    """
    cif1path = Path('tests/util/cif_files/difference_test1.cif')
    cif2path = Path('tests/util/cif_files/difference_test2.cif')

    criteria = {
        'max abs position': 0.1,
        'mean abs position': 0.05,
        'max position/esd': 20.0,
        'mean position/esd': 4.0,
        'max abs uij': 0.01,
        'mean abs uij': 0.005,
        'max uij/esd': 2.0,
        'mean uij/esd': 1.0
    }

    assert check_converged(cif1path, 0, cif2path, 0, criteria) is True

    # Adjust criteria to force a non-converged result
    criteria['max abs position'] = 0.001
    assert check_converged(cif1path, 0, cif2path, 0, criteria) is False
