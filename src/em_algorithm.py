import numpy as np
import pandas as pd


def cov_mat(data: pd.DataFrame) -> np.ndarray:
    """
    Function calculate covariation matrix 
    """
    return data.cov().values


def mu(data):
    """
    Function calculate mean
    if data is array function calculate average of array without NaNs
    if data is pandas DataFrame function calculate array of averages
    """

    if type(data) in (np.ndarray, pd.Series):
        return np.nansum(data) / np.sum(~np.isnan(data))
    
    elif isinstance(data, pd.DataFrame):
        return np.mean(data.values, axis=0)


def step_m(df: pd.DataFrame) -> tuple:
    """
    Function calclulate do maximization in EM
    new mean and average value
    """
    return mu(df), cov_mat(df)


def step_e(df: pd.DataFrame,
           mask: np.ndarray,
           mu_now: np.ndarray,
           sigma_now: np.ndarray) -> pd.DataFrame:
    """
    Expectation step in EM:
    """

    df_filled = df.copy()

    for i in range(len(df_filled)):
        skips = mask[i]

        if not any(skips):
            continue

        observed_values = df_filled.iloc[i, ~skips].values
        observed_means = mu_now[~skips]
        skipped_means = mu_now[skips]

        obs_obs_block = sigma_now[np.ix_(~skips, ~skips)]
        obs_skip_block = sigma_now[np.ix_(skips, ~skips)]

        try:
            inv_obs_obs = np.linalg.pinv(obs_obs_block)
            correction = obs_skip_block @ inv_obs_obs @ (observed_values - observed_means)
            df_filled.iloc[i, skips] = skipped_means + correction
        except np.linalg.LinAlgError:
            pass 

    return df_filled
    


def em_alg(df: pd.DataFrame, mask: np.ndarray, epochs: int = 20) ->pd.DataFrame:
    """
    EM algorithm with presumtion of MVN (Multivariative normals)
    """

    # Fill by mean each skip:
    for column in list(df):
        df[column] = df[column].fillna(mu(df[column]))

    for _ in range(epochs):
        
        # Do M-step:
        mu_now, cov_mat_now = step_m(df)

        # Do E-step:
        df = step_e(df, mask, mu_now, cov_mat_now)

    return df
