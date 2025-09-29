import numpy as np


def cell_constants_to_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Convert cell constants to a 3x3 cell matrix.

    Parameters
    ----------
    a : float
        Cell length a in Angstrom.
    b : float
        Cell length b in Angstrom.
    c : float
        Cell length c in Angstrom.
    alpha : float
        Cell angle alpha in degrees.
    beta : float
        Cell angle beta in degrees.
    gamma : float
        Cell angle gamma in degrees.

    Returns
    -------
    np.ndarray
        The 3x3 cell matrix.
    """
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)

    matrix = np.zeros((3, 3))
    matrix[0, 0] = a
    matrix[0, 1] = b * cos_gamma
    matrix[0, 2] = c * cos_beta
    matrix[1, 1] = b * sin_gamma
    matrix[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    matrix[2, 2] = c * np.sqrt(1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma) ** 2)

    return matrix
