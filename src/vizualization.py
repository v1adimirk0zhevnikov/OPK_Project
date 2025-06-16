import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def mse_vizualization(values: np.ndarray):
    x = np.arange(len(values)) + 1
    plt.errorbar(x, values)

    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.xticks(x)
    plt.title("Mean squared error")

    plt.show()


def mse_at_diff_probablity(prob_values: pd.DataFrame):

    X, Y = np.meshgrid(np.arange(20) + 1, [x / 100 for x in range(10, 31)])
    Z = prob_values.T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Creating surface:
    ax.plot_surface(X, Y, Z, cmap="viridis")

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probablity of skip')
    ax.set_zlabel('MSE')

    ax.view_init(elev=30, azim=-30)

    plt.title("Dependence MSE on probablity of skip and quantity of iterations")

    plt.show()
