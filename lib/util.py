"""
Functions for plotting various visualizations/graphs
"""

__author__ = 'clee'
# Python 3.6.1

# Standard lib
import itertools
import sys

# Additional lib
import matplotlib.pyplot as plt
import numpy as np

def cnf_matrix(true_labels, pred_labels, num_cls):
    """
    Imitates sklearn.metrics.confusion_matrix(), but with non-variable output shape

    :param true_labels:         array-like of ints          True class values
    :param pred_labels:         array-like of ints          Predicted class values
    :param num_cls:             int                         Total number of classes

    :return:
    """
    # Cast as Numpy arrays for boolean indexing
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    cm = np.zeros((num_cls, num_cls))
    for true_cls in range(num_cls):
        for pred_cls in range(num_cls):
            bool_array = (true_labels == true_cls) & (pred_labels == pred_cls)
            cm[true_cls, pred_cls] = sum(bool_array)

    return cm


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()