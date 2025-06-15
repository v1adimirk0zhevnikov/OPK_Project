# File for do missings in dataset
import pandas as pd
import numpy as np


def _create_mask(headers: list, length: int, p: float) -> pd.DataFrame:
    """
    Function for creating mask with define length and columns
    """

    mask: pd.DataFrame = pd.DataFrame()
    for i, header in enumerate(headers):
        np.random.seed(42 + i)
        mask[header] = np.random.rand(length)
        mask[header] = mask[header].apply(lambda x: 0 if x > p else 1)

    return mask


def create_skips(data: pd.DataFrame, probablity: float = 0.1) -> None:
    """
    Function create skips at dataframe
    function get dataframe and probablity of skip    
    """

    columns: list = list(data)
    length: int = len(data[columns[0]])
    mask = _create_mask(columns, length, probablity)

    data[mask == 1] = np.nan
    

def get_true_data(path: str) -> pd.DataFrame:
    """
    Function get true data from file
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    path_clear: str = "sandbox\\sand_data\\data_iris.csv"
    path_with_skips: str = "sandbox\\sand_data\\data_with_skips.csv"

    df = pd.read_csv(path_clear)
    create_skips(df, 0.1)
    df.to_csv(path_with_skips, index=False)    
