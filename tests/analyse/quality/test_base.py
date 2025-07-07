import numpy as np
import pytest

from qcrboxtools.analyse.quality.base import (
    DataQuality,
    ascending_levels2func,
    data_quality_from_level,
    descending_levels2func,
)


@pytest.mark.parametrize(
    "input_level, result", [(-1, DataQuality.GOOD), (100, DataQuality.BAD), (2, DataQuality.MARGINAL)]
)
def test_data_quality_from_level(input_level, result):
    assert data_quality_from_level(input_level) is result

@pytest.mark.parametrize(
    "input_value, levels, expected_index",
    [
        (1.5, (1.0, 2.0, 3.0, 4.0, np.inf), 1),
        (3.5, (1.0, 2.0, 3.0, 4.0, np.inf), 3),
        (5.0, (1.0, 2.0, 3.0, 4.0, np.inf), 4),  # Should return index of last level
        (0.5, (1.0, 2.0, 3.0, 4.0, np.inf), 0),  # Should return index of first level
    ]
)
def test_ascending_levels2func(input_value, levels, expected_index):
    func = ascending_levels2func(levels)
    assert func(input_value) == expected_index  

@pytest.mark.parametrize(
    "input_value, levels, expected_index",
    [
        (1.5, (4.0, 3.0, 2.0, 1.0, -1.0), 3),
        (3.5, (4.0, 3.0, 2.0, 1.0, -1.0), 1),
        (5.0, (4.0, 3.0, 2.0, 1.0, -1.0), 0),
        (0.5, (4.0, 3.0, 2.0, 1.0, -1.0), 4),
    ]
)
def test_descending_levels2func(input_value, levels, expected_index):
    func = descending_levels2func(levels)
    assert func(input_value) == expected_index
