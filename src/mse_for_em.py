import numpy as np


def mse(data: np.ndarray, true_data: np.ndarray, mask: np.ndarray):
    """
    Function MSE (mean squared error)
    compare true values with imputed
    """
    masked_diff = true_data[mask] - data[~mask]  
    return np.mean(masked_diff **2)
