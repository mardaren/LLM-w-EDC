import numpy as np


def calculate_likelihood(y_true, y_pred, unc):
    d = (y_true != y_pred) * 1
    return np.mean(d * unc + (1 - d) * (1 - unc))


def get_most_uncertain_sample(text_data, labels, unc):
    unc_idx = np.argmax(unc)
    return text_data[unc_idx], labels[unc_idx]


def get_least_uncertain_sample(text_data, labels, unc):
    unc_idx = np.argmin(unc)
    return text_data[unc_idx], labels[unc_idx]
