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
