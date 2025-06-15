# File for downloading iris
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
pd.DataFrame(iris.data, columns=iris.feature_names).to_csv("sandbox\\sand_data\\data_iris.csv", index=False)
