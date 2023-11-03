from pytest import approx

from qcrboxtools.util.cif import split_esds, split_esd_single, add_structure_from_cif

def test_split_esd_single():
    val1, esd1 = split_esd_single('100(1)')
    assert val1 == approx(100.0)
    assert esd1 == approx(1.0)

    val2, esd2 = split_esd_single('-100(20)')
    assert val2 == approx(-100.0)
    assert esd2 == approx(20.0)

    val3, esd3 = split_esd_single('0.021(2)')
    assert val3 == approx(0.021)
    assert esd3 == approx(0.002)

    val4, esd4 = split_esd_single('-0.0213(3)')
    assert val4 == approx(-0.0213)
    assert esd4 == approx(0.0003)

    val5, esd5 = split_esd_single('2.1(1.3)')
    assert val5 == approx(2.1)
    assert esd5 == approx(1.3)

def test_split_esds():
    strings = [
        '0.03527(13)',
        '0.02546(10)',
        '0.02949(11)',
        '0.00307(9)',
        '0.01031(9)',
        '-0.00352(8)'
    ]

    solutions = [
        (0.03527, 0.00013),
        (0.02546, 0.00010),
        (0.02949, 0.00011),
        (0.00307, 0.00009),
        (0.01031, 0.00009),
        (-0.00352, 0.00008)
    ]

    values, esds = split_esds(strings)

    for value, esd, (check_value, check_esd) in zip(values, esds, solutions):
        assert value == approx(check_value)
        assert esd == approx(check_esd)