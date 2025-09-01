from typing import Dict, List, Optional, Tuple

import numpy as np
from cctbx import miller
from cctbx.array_family import flex
from iotbx.cif.builders import crystal_symmetry_builder
from iotbx.cif.model import block

from .base import DataQuality, ascending_levels2func, data_quality_from_level, descending_levels2func


def cif_block2intensity_array(cif_block: block) -> miller.array:
    """
    Convert CIF block to intensity array.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing reflection data.

    Returns
    -------
    miller.array
        Intensity array created from the CIF block.
    """
    hs = cif_block["_diffrn_refln.index_h"]
    ks = cif_block["_diffrn_refln.index_k"]
    ls = cif_block["_diffrn_refln.index_l"]
    intensities = cif_block["_diffrn_refln.intensity_net"]
    intensities_esd = cif_block["_diffrn_refln.intensity_net_su"]

    symm = crystal_symmetry_builder(cif_block).crystal_symmetry
    anomalous = not symm.space_group().is_centric
    millerset = miller.set(
        crystal_symmetry=symm,
        indices=flex.miller_index([(int(mil_h), int(mil_k), int(mil_l)) for mil_h, mil_k, mil_l in zip(hs, ks, ls)]),
        anomalous_flag=anomalous,
    )

    intensity_array = millerset.array(
        data=flex.double([float(intensity) for intensity in intensities]),
        sigmas=flex.double([float(intensity_esd) for intensity_esd in intensities_esd]),
    ).set_observation_type_xray_intensity()

    return intensity_array


def precision_all_data(cif_block: block, indicators: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate precision indicators for all data.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing reflection data.
    indicators : Optional[List[str]], optional
        List of indicators to calculate. If None, all indicators are calculated.
        Possible values are:
        - 'd_min lower': Lower limit of resolution
        - 'd_min upper': Upper limit of resolution
        - 'Mean Redundancy': Average number of observations for each reflection
        - 'R_meas': Redundancy-independent merging R-factor
        - 'R_pim': Precision-indicating merging R-factor
        - 'R_int': Internal R-factor
        - 'R_sigma': Ratio of intensity to standard deviation
        - 'CC1/2': Correlation coefficient between random half-datasets
        - 'I/sigma(I)': Signal-to-noise ratio
        - 'Completeness': Percentage of possible reflections that were measured

    Returns
    -------
    Dict[str, float]
        Dictionary of calculated indicators and their values.
    """
    if indicators is None:
        indicators = [
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
    intensity_array = cif_block2intensity_array(cif_block)

    int_nonsym = intensity_array.remove_systematic_absences()

    already_merged = int_nonsym.is_unique_set_under_symmetry()

    results_overall = {}

    if any(val in indicators for val in ("d_min lower", "d_min upper")):
        lowlim, highlim = intensity_array.d_max_min()
        if "d_min lower" in indicators:
            results_overall["d_min lower"] = lowlim
        if "d_min upper" in indicators:
            results_overall["d_min upper"] = highlim
        if "I/sigma(I)" in indicators:
            results_overall["I/sigma(I)"] = int_nonsym.i_over_sig_i()
        if "Completeness" in indicators:
            results_overall["Completeness"] = int_nonsym.completeness()

    if not already_merged:
        int_merged = int_nonsym.merge_equivalents()
        if "Mean Redundancy" in indicators:
            results_overall["Mean Redundancy"] = int_merged.redundancies().as_double().mean()
        if "R_meas" in indicators:
            results_overall["R_meas"] = int_merged.r_meas()  # is equal to R_rim
        if "R_pim" in indicators:
            results_overall["R_pim"] = int_merged.r_pim()
        if "R_int" in indicators:
            results_overall["R_int"] = int_merged.r_int()
        if "R_sigma" in indicators:
            results_overall["R_sigma"] = int_merged.r_sigma()
        if "CC1/2" in indicators:
            results_overall["CC1/2"] = int_nonsym.cc_one_half()
    else:
        non_sensical_entries = ["Mean Redundancy", "R_meas", "R_pim", "R_int", "R_sigma", "CC1/2"]
        for indicator in non_sensical_entries:
            if indicator in indicators:
                results_overall[indicator] = "N/A"

    return results_overall


def precision_all_data_quality(results_overall: Dict[str, float]) -> Dict[str, DataQuality]:
    """
    Calculate data quality for precision indicators.

    Parameters
    ----------
    results_overall : Dict[str, float]
        Dictionary of precision indicators and their values.
        Possible keys are:
        - 'd_min lower', 'd_min upper', 'Mean Redundancy', 'R_meas', 'R_pim',
          'R_int', 'R_sigma', 'CC1/2', 'I/sigma(I)', 'Completeness'

    Returns
    -------
    Dict[str, DataQuality]
        Dictionary of indicators and their corresponding data quality.
    """
    value2level_dict = {
        "Mean Redundancy": descending_levels2func((10, 5, 4, 3, -1)),
        "R_meas": ascending_levels2func((4.0, 6.0, 10.0, 15.0, np.inf)),
        "R_pim": ascending_levels2func((2.0, 3.0, 5.0, 15.0, np.inf)),
        "R_int": ascending_levels2func((4, 6, 10, 15, np.inf)),
        "R_sigma": ascending_levels2func((4, 6, 10, 15, np.inf)),
        "CC1/2": descending_levels2func((0.995, 0.99, 0.98, 0.95, -1)),
        "d_min upper": ascending_levels2func((0.75, 0.841, 0.86, 0.88, np.inf)),
        "I/sigma(I)": descending_levels2func((15, 12, 9, 6, -1)),
        "Completeness": descending_levels2func((0.99, 0.95, 0.90, 0.8, -1.0)),
    }
    quality_values = {}
    for indicator, value in results_overall.items():
        if indicator == "d_min lower" or value == "N/A":
            quality_values[indicator] = DataQuality.INFORMATION
        else:
            operation = value2level_dict[indicator]
            level = operation(float(value))
            quality_values[indicator] = data_quality_from_level(int(level))

    return quality_values


def precision_vs_resolution(
    cif_block: block, n_bins: int, indicators: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate precision indicators versus resolution.

    Parameters
    ----------
    cif_block : iotbx.cif.model.block
        CIF block containing reflection data.
    n_bins : int
        Number of resolution bins.
    indicators : Optional[List[str]], optional
        List of indicators to calculate. If None, all indicators are calculated.
        Possible values are:
        - 'd_min lower': Lower limit of resolution for each bin
        - 'd_min upper': Upper limit of resolution for each bin
        - 'Mean Redundancy': Average number of observations for each reflection in each bin
        - 'R_meas': Redundancy-independent merging R-factor for each bin
        - 'R_pim': Precision-indicating merging R-factor for each bin
        - 'R_int': Internal R-factor for each bin
        - 'R_sigma': Ratio of intensity to standard deviation for each bin
        - 'CC1/2': Correlation coefficient between random half-datasets for each bin
        - 'I/sigma(I)': Signal-to-noise ratio for each bin
        - 'Completeness': Percentage of possible reflections that were measured for each bin

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of calculated indicators and their values for each resolution bin.
    """

    if indicators is None:
        indicators = [
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
    intensity_array = cif_block2intensity_array(cif_block)

    if intensity_array.is_unique_set_under_symmetry():
        non_sensical_entries = ["Mean Redundancy", "R_meas", "R_pim", "R_int", "R_sigma", "CC1/2"]
        indicators = [ind for ind in indicators if ind not in non_sensical_entries]

    intensity_array.setup_binner(n_bins=n_bins)

    binning_range = intensity_array.binner().range_used()
    results_binned = {name: np.zeros(len(binning_range)) for name in indicators}
    for i_bin in intensity_array.binner().range_used():
        array_index = i_bin - binning_range.start
        sel = intensity_array.binner().selection(i_bin)
        bin_array = intensity_array.select(sel)
        bin_merged = bin_array.merge_equivalents()
        lowlim, highlim = intensity_array.binner().bin_d_range(i_bin)

        if "d_min lower" in indicators:
            results_binned["d_min lower"][array_index] = lowlim
        if "d_min upper" in indicators:
            results_binned["d_min upper"][array_index] = highlim
        if "Mean Redundancy" in indicators:
            results_binned["Mean Redundancy"][array_index] = bin_merged.redundancies().as_double().mean()
        if "R_meas" in indicators:
            results_binned["R_meas"][array_index] = bin_merged.r_meas()
        if "R_pim" in indicators:
            results_binned["R_pim"][array_index] = bin_merged.r_pim()
        if "R_int" in indicators:
            results_binned["R_int"][array_index] = bin_merged.r_int()
        if "R_sigma" in indicators:
            try:
                results_binned["R_sigma"][array_index] = bin_merged.r_sigma()
            except ZeroDivisionError:
                results_binned["R_sigma"][array_index] = None
        if "CC1/2" in indicators:
            try:
                results_binned["CC1/2"][array_index] = bin_array.cc_one_half()
            except ZeroDivisionError:
                results_binned["CC1/2"][array_index] = None
        if "I/sigma(I)" in indicators:
            results_binned["I/sigma(I)"][array_index] = bin_merged.array().i_over_sig_i()
        if "Completeness" in indicators:
            results_binned["Completeness"][array_index] = bin_array.completeness(d_max=lowlim)

    return results_binned


def diederichs_plot(cif_block: block) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for Diederichs plot.

    Parameters
    ----------
    input_cif_path : str
        Path to the input CIF file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of log10(intensity) and I/sigma(I) values for plotting.
    """
    intensity = np.array(cif_block["_diffrn_refln.intensity_net"]).astype(np.float16)
    sigma = np.array(cif_block["_diffrn_refln.intensity_net_su"]).astype(np.float64)
    valid = np.logical_and(sigma > 0, intensity > 0)
    sigma = sigma[valid]
    intensity = intensity[valid]
    log10i = np.log10(intensity)
    i_over_sigma = intensity / sigma
    return log10i, i_over_sigma
