NUMERICAL_VARIABLES = ['GrLivArea', 'GarageArea', 'TotalBsmtSF']

VARIABLES_TO_LOG_TRANSFORM = ['GrLivArea', 'SalePrice'],

CATEGORICAL_VARIABLES = categorical_variables = [
    'OverallQual', 'FullBath', 'TotRmsAbvGrd']

FEATURES = NUMERICAL_VARIABLES + CATEGORICAL_VARIABLES

TARGET = ['SalePrice']
