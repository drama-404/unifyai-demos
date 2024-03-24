from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile

DATASET = 'danofer/sarcasm'
DATA_DIR = './data'

def get_kaggle_dataset():
    """
    Authenticates with Kaggle, downloads the sarcasm dataset, unzips it, and returns the training
    and test CSV files as Pandas DataFrames.

    This function assumes that the Kaggle API credentials (kaggle.json) are present in the
    current working directory.

    Returns:
        tuple: A tuple containing two Pandas DataFrames:
            - The first element is the training data (train-balanced-sarcasm.csv)
            - The second element is the test data (test-balanced.csv)
    """
    # Authenticate with Kaggle
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    api.dataset_download_files(dataset=DATASET, path=DATA_DIR, quiet=False)
    
    # Declare the specific csv files to be used
    zip_path = f'{DATA_DIR}/sarcasm.zip'
    file_names = ['train-balanced-sarcasm.csv', 'test-balanced.csv']
    datasets = []

    # Read the csv files inside the zip folder
    with zipfile.ZipFile(zip_path, 'r') as zipped_dir:    
        for file in file_names:
            with zipped_dir.open(file) as f:
                datasets.append(pd.read_csv(f))
    
    return datasets[0], datasets[1]