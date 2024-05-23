from typing import Union

import numpy as np


def infer_and_cast(s: str, cast_np=True) -> Union[float, int, bool, np.ndarray, str]:
    """
    Attempts to cast a string to an integer, float, boolean, or retains it as
    a string based on its value.

    Parameters
    ----------
    s : str
        The string to be cast.
    cast_np : bool, optional
        Whether to attempt to cast the string to a numpy array if it contains
        space-separated values. Default is True.

    Returns
    -------
    Union[int, float, bool, np.ndarray, str]
        The cast value.
    """
    # Try to cast to integer
    try:
        return int(s)
    except ValueError:
        pass

    # Try to cast to float
    try:
        return float(s)
    except ValueError:
        pass

    # Try to cast to boolean
    if s.lower() in ["true", "false"]:
        return s.lower() == "true"

    # Try to cast to numpy array (if contains space-separated values)
    if " " in s and cast_np:
        return np.array([infer_and_cast(sub_s) for sub_s in s.split()])

    # If all else fails, return as string
    return s
