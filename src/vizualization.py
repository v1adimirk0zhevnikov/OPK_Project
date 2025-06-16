import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mse_vizualization(values: np.ndarray):
    x = np.arange(len(values)) + 1
    plt.errorbar(x, values)

    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.xticks(x)
    plt.title("Mean squared error")

    plt.show()
