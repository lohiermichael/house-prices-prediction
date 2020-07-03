from sklearn.base import BaseEstimator, TransformerMixin

"""
Pipeline steps:
1. Numerical: Remove outliers IQR
2. Numerical: Log transform variables
3. Categorical: Add 'Rare' label
4. Categorical: One-hot-encode
5. Categorical: Add additional columns
6. Feature Scaling

"""
