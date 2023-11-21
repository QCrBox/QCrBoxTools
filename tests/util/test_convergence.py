from pathlib import Path

from pytest import approx

from qcrboxtools.util.convergence import position_difference, anisotropic_adp_difference

def test_position_difference_diff():
    """
    This test validates the correctness of the calculated maximum and mean absolute
    positions differences, as well as the  maximum and mean positions normalised by
    the estimated standard deviation (esd).
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
    Validates that all differences are zero if the input cif files are
    the same
    """
    cif1path = Path('tests/util/cif_files/difference_test1.cif')

    diff_dict = position_difference(cif1path, 0, cif1path, 0)

    assert diff_dict['max abs position'] == 0.0
    assert diff_dict['mean abs position'] == 0.0
    assert diff_dict['max position/esd'] == 0.0
    assert diff_dict['mean position/esd'] == 0.0


def test_uij_difference_diff():
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
    cif1path = Path('tests/util/cif_files/difference_test1.cif')

    diff_dict = anisotropic_adp_difference(cif1path, 0, cif1path, 0)

    assert diff_dict['max abs uij'] == approx(0.00)
    assert diff_dict['mean abs uij'] == approx(0.00)
    assert diff_dict['max uij/esd'] == approx(0.00)
    assert diff_dict['mean uij/esd'] == approx(0.00)