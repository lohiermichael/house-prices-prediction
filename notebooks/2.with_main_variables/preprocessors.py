import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import feature_engineering

"""
Pipeline steps:
1. Numerical: Remove outliers IQR
2. Numerical: Log transform variables
3. Categorical: Add 'Rare' label
4. Categorical: One-hot-encode (no need from sklearn)
5. Categorical: Add additional columns
6. Feature Scaling

"""


class OutliersRemoval(BaseEstimator, TransformerMixin):

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
