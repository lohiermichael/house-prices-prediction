import utils

ROOT = utils.get_project_root()

NUMERICAL_VARIABLES = ['GrLivArea', 'GarageArea', 'TotalBsmtSF']

VARIABLES_TO_LOG_TRANSFORM = ['GrLivArea', 'SalePrice']

CATEGORICAL_VARIABLES = categorical_variables = [
    'OverallQual', 'FullBath', 'TotRmsAbvGrd']

FEATURES = NUMERICAL_VARIABLES + CATEGORICAL_VARIABLES

TARGET = 'SalePrice'

MOST_RELEVANT_VARIABLES = FEATURES + [TARGET]

TRAINING_DATAFILE = f'{ROOT}/datasets/inputs/train.csv'

TESTING_DATAFILE = f'{ROOT}/datasets/inputs/test.csv'
