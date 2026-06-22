import numpy as np
from scipy.stats import entropy


def spectral_entropy(signal):

    fft = np.fft.rfft(signal)

    power = np.abs(fft) ** 2

    power /= power.sum()

    return entropy(power)