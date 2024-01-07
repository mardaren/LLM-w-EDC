import numpy as np


def calculate_likelihood(y_true, y_pred, unc):
    d = (y_true != y_pred) * 1
    return np.mean(d * unc + (1 - d) * (1 - unc))


def get_most_uncertain_sample():
    pass
