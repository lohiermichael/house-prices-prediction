# Create structure
mkdir datasets
mkdir datasets/inputs
mkdir datasets/outputs
mkdir datasets/outputs/with_main_variables

# Import data from Kaggle
kaggle competitions download -c house-prices-advanced-regression-techniques -p datasets/inputs
# Unzip the imported folder
unzip 'datasets/inputs/house-prices-advanced-regression-techniques.zip' -d 'datasets/inputs'
# Remove the zipped file
rm 'datasets/inputs/house-prices-advanced-regression-techniques.zip'