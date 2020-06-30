import pandas as pd
import numpy as np
from scipy import stats


def remove_outliers(df: pd.DataFrame, numerical_variables: list, strategy: str = 'IQR') -> pd.DataFrame:
    """Remove rows of the input dataframe having at least one variable with outlier

    Args:
        df (pd.DataFrame): Input dataframe
        numerical_variables (list): list of the numerical variables names in the input dataframe
        strategy (str, optional): IQR or z-score outlier strategy. Defaults to 'IQR'.

    Returns:
        pd.DataFrame: transformed variable
    """
    assert strategy in [
        'IQR', 'z-score'], "You must choose IQR or z-score strategy"

    df = df[numerical_variables]

    if strategy == "IQR":
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        return df[~is_outlier.any(axis=1)]

    elif strategy == 'z-score':
        z = np.abs(stats.zscore(df))
        return df[(z < 3).all(axis=1)]
