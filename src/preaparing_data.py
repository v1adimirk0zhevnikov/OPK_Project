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


def create_skips(data: pd.DataFrame, probablity: float = 0.1) -> pd.DataFrame:
    """
    Function create skips at dataframe
    function get dataframe and probablity of skip    
    """
    df = data.copy()
    columns: list = list(df)
    length: int = len(df[columns[0]])
    mask = _create_mask(columns, length, probablity)

    df[mask == 1] = np.nan
    return df
