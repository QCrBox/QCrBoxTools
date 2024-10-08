from textwrap import dedent, indent
from typing import List, NamedTuple

from ..base import DataQuality
from . import convert_to_mathjax, data_quality_to_css_name


class QualityIndicatorBox(NamedTuple):
    """Represents a data quality indicator box with attributes for display.

    Attributes
    ----------
    name : str
        The name of the data quality indicator.
    value : str
        The value associated with the data quality indicator.
    unit : str
        The unit of the indicator's value (e.g., %, ms).
    quality_level : DataQuality
        The quality level, represented as a `DataQuality` enum.
    """

    name: str
    value: str
    unit: str
    quality_level: DataQuality


def quality_box_div(indicator: QualityIndicatorBox) -> str:
    """Creates an HTML div for a single quality indicator box.

    Parameters
    ----------
    indicator : QualityIndicatorBox
        The quality indicator data to display in the box.

    Returns
    -------
    str
        A formatted HTML string for a single quality indicator box.
    """
    indicator_name = convert_to_mathjax(str(indicator.name))
    indicator_value = convert_to_mathjax(str(indicator.value))
    indicator_unit = convert_to_mathjax(str(indicator.unit))
    return dedent(f"""
        <div class="indicator {data_quality_to_css_name[indicator.quality_level]}">
            <div class="name">{indicator_name}</div>
            <div class="value">{indicator_value}</div>
            <div class="unit">{indicator_unit}</div>
        </div>
    """).strip()


def quality_div_group(indicators: List[QualityIndicatorBox]) -> str:
    """Creates an HTML container div for a group of quality indicator boxes.

    Parameters
    ----------
    indicators : List[QualityIndicatorBox]
        A list of quality indicator boxes to display in the container.

    Returns
    -------
    str
        A formatted HTML string for the container div with multiple indicators.
    """
    strings = (quality_box_div(indicator) for indicator in indicators)
    string = indent("\n".join(strings), prefix="    ")
    return f'<div class="container">\n{string}\n</div>'
