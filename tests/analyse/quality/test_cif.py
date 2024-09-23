import pytest
from iotbx.cif.model import block

from qcrboxtools.analyse.quality.base import DataQuality
from qcrboxtools.analyse.quality.cif import from_entry


@pytest.mark.parametrize(
    "entry, perfect_value",
    [
        ("_refine_ls.r_factor_all", 0.01),
        ("_refine_ls.wr_factor_gt", 0.01),
        ("_refine.diff_density_max", 0.01),
        ("_refine.diff_density_min", -0.01),
        ("_refine_ls.d_res_high", 0.12),
        ("_refine_ls.goodness_of_fit_ref", 1.0),
    ],
)
def test_from_entry_perfect(entry, perfect_value):
    cif_block = block()
    cif_block.add_data_item(entry, str(perfect_value))
    assert from_entry(cif_block, entry) is DataQuality.GOOD


@pytest.mark.parametrize(
    "entry, bad_value",
    [
        ("_refine_ls.r_factor_all", 100.01),
        ("_refine_ls.wr_factor_gt", 200.01),
        ("_refine.diff_density_max", 100.01),
        ("_refine.diff_density_min", -100.01),
        ("_refine_ls.d_res_high", 9.0),
        ("_refine_ls.goodness_of_fit_ref", 200.0),
    ],
)
def test_from_entry_worst(entry, bad_value):
    cif_block = block()
    cif_block.add_data_item(entry, str(bad_value))
    assert from_entry(cif_block, entry) is DataQuality.BAD
