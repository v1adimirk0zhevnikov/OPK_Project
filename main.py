import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.em_algorithm import em_alg, mu
from src.preaparing_data import create_skips
from src.vizualization import *
from src.mse_for_em import mse


PATH: str = "data\\data_iris.csv"


#=====GET DATA=====#
dataset: pd.DataFrame = pd.read_csv(PATH)
data_with_skips: pd.DataFrame = create_skips(dataset, 0.2)
mask = data_with_skips.isna().values


#====VIZUAL WITH MATPLOTLIB====#
val: np.ndarray = np.zeros(20)
for j in range(20):
    data_with_skips = em_alg(data_with_skips, mask, epochs=1)
    val[j] = mse(data_with_skips, dataset, mask)

mse_vizualization(val)