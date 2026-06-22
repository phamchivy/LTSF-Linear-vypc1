import numpy as np
from statsmodels.tsa.stattools import acf


def compute_acf(residual, nlags=50):
    """
    residual: numpy array [L]

    return: acf coefficients
    """

    return acf(residual, nlags=nlags)