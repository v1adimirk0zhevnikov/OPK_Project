import pandas as pd
import numpy as np
from preaparing_data import create_skips


path: str = "sandbox\\sand_data\\data_with_skips.csv"

df: pd.DataFrame = pd.read_csv(path)

current_row: np.ndarray = df.iloc[10].values
print(current_row)
skips_mask = np.isnan(current_row)

print(df.head(11))
