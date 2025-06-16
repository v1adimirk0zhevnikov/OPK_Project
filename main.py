import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.em_algorithm import em_alg
from src.preaparing_data import create_skips
from src.vizualization import *
from src.mse_for_em import mse


PATH: str = "data\\data_iris.csv"
PATH2: str = "data\\data_iris_with_skips.csv"
PATH3: str = "data\\data_iris_after_em.csv"


#=====GET DATA=====#
dataset: pd.DataFrame = pd.read_csv(PATH)
data_with_skips: pd.DataFrame = create_skips(dataset, 0.15)
mask = data_with_skips.isna().values

#==== ====#
data_with_skips.to_csv(PATH2, index=False)

# Use em:
df_after_em = em_alg(data_with_skips, mask, epochs=20)
df_after_em.to_csv(PATH3, index=False)


#====VIZUAL WITH MATPLOTLIB====#
mse_values: np.ndarray = np.zeros(20)
df: pd.DataFrame = pd.DataFrame()

for probablity in [x / 100 for x in range(1, 31)]:
    data_with_skips: pd.DataFrame = create_skips(dataset, probablity)
    mask = data_with_skips.isna().values
    for j in range(20):
        data_with_skips = em_alg(data_with_skips, mask, epochs=1)
        mse_values[j] = mse(data_with_skips, dataset, mask)
    df[str(probablity)] = mse_values


mse_diff_prob_vizualization(df)
