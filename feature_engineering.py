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
        pd.DataFrame: transformed dataframe
    """
    assert strategy in [
        'IQR', 'z-score'], "You must choose IQR or z-score strategy"

    df_numerical = df[numerical_variables]

    if strategy == "IQR":
        Q1 = df_numerical.quantile(0.25)
        Q3 = df_numerical.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = (df_numerical < (Q1 - 1.5 * IQR)
                      ) | (df_numerical > (Q3 + 1.5 * IQR))
        outliers = df_numerical[is_outlier.any(axis=1)]

    elif strategy == 'z-score':
        z = np.abs(stats.zscore(df))
        outliers = df_numerical[(z >= 3).all(axis=1)]

    return df.drop(outliers.index, axis=0)


def replace_rare_labels(df: pd.DataFrame, categorical_variables: list, percentage_rare: float = 0.01) -> pd.DataFrame:
    """Gather  rare labels under the same label name **Rare**.
       We will call rare labels for a categorical variable, a label that is shared by less than a certain percentage of the instances.

    Args:
        df (pd.DataFrame): Input dataframe
        categorical_variables (list): list of the categorical variables names in the input dataframe
        percentage_rare (float, optional): Threshold percentage of occurency of a label. Defaults to 0.01.

    Returns:
        pd.DataFrame: transformed dataframe
    """

    for var in categorical_variables:
        # Percentage occurency
        per_occ = df[var].value_counts()/len(df)
        # Find rare labels
        rare_labels = per_occ[per_occ < 0.01].index.tolist()
        # Transform the dataframe
        df[var] = np.where(df[var].isin(rare_labels),
                           'Rare',
                           df[var])

    return df
