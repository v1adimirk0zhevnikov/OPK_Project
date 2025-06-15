import numpy as np


def mse(data: np.ndarray, true_data: np.ndarray):
    """
    Function MSE (mean squared error) 
    """
    return np.sqrt(np.mean((data - true_data) ** 2))
