from typing import Callable, Dict, Union

import numpy as np
from iotbx.cif.model import block

from .base import DataQuality, ascending_levels2func, data_quality_from_level

direct_cif_evaluations: Dict[str, Callable[[float], Union[float, int]]] = {
    "_refine_ls.r_factor_all": lambda x: x / 0.05,
    "_refine_ls.wr_factor_gt": lambda x: x / 0.1,
    "_refine.diff_density_max": ascending_levels2func((0.51, 1.01, 2.01, 3.01, np.inf)),
    "_refine.diff_density_min": lambda x: next(i for i, v in enumerate((0.51, 1.01, 2.01, 3.01, np.inf)) if -x < v),
    "_refine_ls.d_res_high": ascending_levels2func((0.75, 0.841, 0.86, 0.88, np.inf)),
    "_refine_ls.goodness_of_fit_ref": lambda x: next(
        i for i, v in enumerate((0.1, 0.2, 0.3, 0.5, np.inf)) if abs(x - 1) < v
    ),
}


def from_entry(cif_block: block, cif_entry: str) -> DataQuality:
    """
    Evaluate the data quality of a specific CIF entry.

    This function retrieves a value from the CIF block for a given entry,
    applies a corresponding evaluation function, and returns the data quality.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        The CIF block containing the data to be evaluated.
    cif_entry : str
        The specific CIF entry to evaluate. Must be one of:
        - "_refine_ls.r_factor_all"
        - "_refine_ls.wr_factor_gt"
        - "_refine.diff_density_max"
        - "_refine.diff_density_min"
        - "_refine_ls.d_res_high"
        - "_refine_ls.goodness_of_fit_ref"

    Returns
    -------
    DataQuality
        The evaluated data quality for the specified CIF entry.

    Raises
    ------
    KeyError
        If the specified cif_entry is not in the direct_cif_evaluations dictionary.
    """
    value = cif_block[cif_entry]
    operation = direct_cif_evaluations[cif_entry]
    level = operation(float(value))
    return data_quality_from_level(int(level))
