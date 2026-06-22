import torch


def residual_energy(x, residual):
    """
    x,residual: [B,L,C]

    return scalar
    """

    num = (residual ** 2).sum()
    den = (x ** 2).sum()

    return (num / den).item()