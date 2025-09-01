import random

import numpy as np
import pytest
from iotbx.cif.model import block, loop

from qcrboxtools.analyse.quality.base import DataQuality
from qcrboxtools.analyse.quality.precision import (
    cif_block2intensity_array,
    diederichs_plot,
    precision_all_data,
    precision_all_data_quality,
    precision_vs_resolution,
)

"""Do not retest the that cctbx works, only implementation"""


@pytest.fixture(name="hkl_cell_cif_block")
def fixture_hkl_cell_cif_block():
    cif_block = block()
    for letter in ("a", "b", "c"):
        entry = f"_cell.length_{letter}"
        value = random.gauss(15.0, 3)
        cif_block.add_data_item(entry, str(value))
    for name in ("alpha", "gamma"):
        entry = f"_cell.angle_{name}"
        cif_block.add_data_item(entry, "90.0")
    cif_block.add_data_item("_cell.angle_beta", str(random.gauss(115.0, 10.0)))
    cif_block.add_data_item("_space_group.it_number", "14")
    cif_block.add_data_item("_space_group.name_h-m_alt", "P 21/c")
    cif_block.add_data_item("_space_group.name_hall", "-P 2ybc")

    size = 500
    diffrn_refln_loop = loop()
    for index in ("h", "k", "l"):
        # prevent (0 0 0) by enforcing l != 0
        random_ints = np.random.randint(-5, 5, size)
        if index == "l":
            random_ints[random_ints == 0] = 1
        diffrn_refln_loop.add_column(f"_diffrn_refln.index_{index}", random_ints)
    diffrn_refln_loop.add_column("_diffrn_refln.intensity_net", np.random.rand(size) * 1000)
    diffrn_refln_loop.add_column("_diffrn_refln.intensity_net_su", np.random.rand(size) * 10)

    cif_block.add_loop(diffrn_refln_loop)
    return cif_block


def test_cifblock2intensity_array(hkl_cell_cif_block):
    intensity_array = cif_block2intensity_array(hkl_cell_cif_block)
    assert intensity_array.is_xray_intensity_array()


def test_precision_all_data(hkl_cell_cif_block):
    possible_indicators = [
        "d_min lower",
        "d_min upper",
        "Mean Redundancy",
        "R_meas",
        "R_pim",
        "R_int",
        "R_sigma",
        "CC1/2",
        "I/sigma(I)",
        "Completeness",
    ]
    # test None -> select all
    precision_dict = precision_all_data(hkl_cell_cif_block)
    assert len(precision_dict) == len(possible_indicators)
    for indicator in possible_indicators:
        assert indicator in precision_dict

    # only one
    test_index = 2
    precision_dict = precision_all_data(hkl_cell_cif_block, indicators=[possible_indicators[test_index]])
    assert len(precision_dict) == 1
    assert possible_indicators[test_index] in precision_dict

    # two
    test_indexes = [3, 4]
    indicators = [possible_indicators[i] for i in test_indexes]
    precision_dict = precision_all_data(hkl_cell_cif_block, indicators=indicators)
    assert len(precision_dict) == len(test_indexes)
    for indicator in indicators:
        assert indicator in precision_dict


def test_precision_all_data_quality(hkl_cell_cif_block):
    precision_dict = precision_all_data(hkl_cell_cif_block)
    data_quality = precision_all_data_quality(precision_dict)
    assert len(data_quality) == len(precision_dict)
    assert data_quality["d_min lower"] is DataQuality.INFORMATION


def test_precision_vs_resolution(hkl_cell_cif_block):
    possible_indicators = [
        "d_min lower",
        "d_min upper",
        "Mean Redundancy",
        "R_meas",
        "R_pim",
        "R_int",
        "R_sigma",
        "CC1/2",
        "I/sigma(I)",
        "Completeness",
    ]
    n_bins = 3
    # test None -> select all
    precision_dict = precision_vs_resolution(hkl_cell_cif_block, n_bins=n_bins)
    assert len(precision_dict) == len(possible_indicators)
    for indicator in possible_indicators:
        assert indicator in precision_dict
        assert len(precision_dict[indicator]) == n_bins

    # only one
    n_bins = 4
    test_index = 2
    indicator = possible_indicators[test_index]
    precision_dict = precision_vs_resolution(hkl_cell_cif_block, n_bins=n_bins, indicators=[indicator])
    assert len(precision_dict) == 1
    assert indicator in precision_dict
    assert len(precision_dict[indicator]) == n_bins

    # two
    n_bins = 2
    test_indexes = [3, 4]
    indicators = [possible_indicators[i] for i in test_indexes]
    precision_dict = precision_vs_resolution(hkl_cell_cif_block, n_bins=n_bins, indicators=indicators)
    assert len(precision_dict) == len(test_indexes)
    for indicator in indicators:
        assert indicator in precision_dict
        assert len(precision_dict[indicator]) == n_bins


def test_diederichs_plot(hkl_cell_cif_block):
    log10i, i_over_sigma = diederichs_plot(hkl_cell_cif_block)
    assert log10i.shape == i_over_sigma.shape
