import numpy as np


def calculate_likelihood(y_true, y_pred, unc):
    e = (y_true == y_pred) * 1
    return np.mean(e * unc + (1 - e) * (1 - unc))


def get_most_uncertain_sample():
    pass
