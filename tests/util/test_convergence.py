from pathlib import Path

from pytest import approx

from qcrboxtools.util.convergence import position_difference

def test_position_difference_diff():
    """
    This test validates the correctness of the calculated maximum and mean absolute
    positions differences, as well as the  maximum and mean positions normalised by
    the estimated standard deviation (esd).
    """
    cif1path = Path('tests/util/cif_files/position_difference1.cif')
    cif2path = Path('tests/util/cif_files/position_difference2.cif')

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
    cif1path = Path('tests/util/cif_files/position_difference1.cif')

    diff_dict = position_difference(cif1path, 0, cif1path, 0)

    assert diff_dict['max abs position'] == 0.0
    assert diff_dict['mean abs position'] == 0.0
    assert diff_dict['max position/esd'] == 0.0
    assert diff_dict['mean position/esd'] == 0.0