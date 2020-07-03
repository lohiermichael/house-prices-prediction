import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import feature_engineering


class OutliersRemover(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = feature_engineering.remove_outliers(df=X,
                                                numerical_variables=self.variables)
        return X


class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):

        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.log(X[feature])
        return X


class RareLabelCategoricalEncode(BaseEstimator, TransformerMixin):

    def __init__(self, variables, percentage_rare: int = 0.01):
        self.percentage_rare = percentage_rare
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = feature_engineering.replace_rare_labels(df=X,
                                                    categorical_variables=self.variables,
                                                    percentage_rare=self.percentage_rare)
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = pd.get_dummies(data=X,
                           columns=self.variables,
                           drop_first=True)
        return X


def match_one_hot_encoded_vars(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Make the columns of X_train and X_test similar

    Args:
        X_train (pd.Dataframe): Training set
        X_test (pd.Dataframe): Test set
    """
    _, vars_X_train_not_X_test, vars_X_test_not_X_train = feature_engineering.summarize_common_variables(df1=X_train,
                                                                                                         df2=X_test,
                                                                                                         print_of=False)

    X_train = feature_engineering.complete_one_hot_variables(df=X_train,
                                                             var_names=vars_X_test_not_X_train)
    X_test = feature_engineering.complete_one_hot_variables(df=X_test,
                                                            var_names=vars_X_train_not_X_test)

    assert set(X_train.columns) == set(
        X_test.columns), "The columns don't match"

    return X_train, X_test
