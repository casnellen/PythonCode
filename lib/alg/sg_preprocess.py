"""
Functions for spectrogram preprocessing methods
"""

__author__ = 'clee'
# Python 3.6.1

# Standard lib
import math

# Additional lib
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def log_mag_norm(sg):
    """
    Log (base 10) magnitude of spectrograms, normalized per spectrogram
    Exponentially small value to avoid divide by 0 errors

    :param sg:
    :return:
    """
    log_mag_sg = np.log10(abs(sg + math.exp(-10)))
    norm_data = MinMaxScaler().fit_transform(log_mag_sg.reshape(-1, 1))

    return norm_data.reshape(sg.shape)