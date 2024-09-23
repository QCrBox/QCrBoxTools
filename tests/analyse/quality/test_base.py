import pytest

from qcrboxtools.analyse.quality.base import DataQuality, data_quality_from_level


@pytest.mark.parametrize(
    "input_level, result", [(-1, DataQuality.GOOD), (100, DataQuality.BAD), (2, DataQuality.MARGINAL)]
)
def test_data_quality_from_level(input_level, result):
    assert data_quality_from_level(input_level) is result
