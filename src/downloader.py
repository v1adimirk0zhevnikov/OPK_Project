# File for downloading iris
from sklearn.datasets import load_iris
import pandas as pd

if __name__ == "__main__":
    iris = load_iris()
    pd.DataFrame(iris.data, columns=iris.feature_names).to_csv("data\\data_iris.csv", index=False)
