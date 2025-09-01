from enum import Enum
from typing import Callable, Tuple


class DataQuality(Enum):
    """
    Enumeration of data quality levels.

    Attributes
    ----------
    INFORMATION : int
        Represents information-only data, not evaluated for quality.
    GOOD : int
        Represents high-quality data.
    GOODISH : int
        Represents reasonably good quality data.
    MARGINAL : int
        Represents questionable quality data.
    BADISH : int
        Represents poor quality data.
    BAD : int
        Represents very poor quality data.
    """

    INFORMATION = -99
    GOOD = 0
    GOODISH = 1
    MARGINAL = 2
    BADISH = 3
    BAD = 4


def data_quality_from_level(level: int) -> DataQuality:
    """
    Convert a numeric level to a DataQuality enum.

    Parameters
    ----------
    level : int
        The numeric level to convert.

    Returns
    -------
    DataQuality
        The corresponding DataQuality enum value.
    """
    if level < 0:
        return DataQuality(0)
    if level >= 4:
        return DataQuality(4)
    return DataQuality(level)


def ascending_levels2func(levels: Tuple[float, ...]) -> Callable[[float], int]:
    """
    Create a function that maps a value to its corresponding level index.

    This function returns a lambda function that takes a single float value
    and returns the index of the first level in the given tuple that the value
    is less than.

    Parameters
    ----------
    levels : Tuple[float, ...]
        A tuple of floats representing the ascending levels.

    Returns
    -------
    Callable[[float], int]
        A function that takes a float and returns the corresponding level index.

    Examples
    --------
    >>> func = ascending_levels2func((1.0, 2.0, 3.0, 4.0))
    >>> func(1.5)
    1
    >>> func(3.5)
    3
    """
    return lambda x: next((i for i, v in enumerate(levels) if x < v))


def descending_levels2func(levels: Tuple[float, ...]) -> Callable[[float], int]:
    """
    Create a function that maps a value to its corresponding level index in descending order.

    This function returns a lambda function that takes a single float value
    and returns the index of the first level in the given tuple that the value
    is greater than.

    Parameters
    ----------
    levels : Tuple[float, ...]
        A tuple of floats representing the descending levels.

    Returns
    -------
    Callable[[float], int]
        A function that takes a float and returns the corresponding level index.

    Examples
    --------
    >>> func = descending_levels2func((4.0, 3.0, 2.0, 1.0))
    >>> func(3.5)
    1
    >>> func(1.5)
    3
    """
    return lambda x: next((i for i, v in enumerate(levels) if x > v))
