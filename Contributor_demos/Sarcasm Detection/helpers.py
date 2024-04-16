from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile
import numpy as np
import pickle

DATASET = 'danofer/sarcasm'
DATA_DIR = './data'


def save_to_pickle(df, filename):
    """
    Save a DataFrame to a pickle file.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        filename (str): The name of the file to save the DataFrame to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(df, f)

def load_from_pickle(filename):
    """
    Load a DataFrame from a pickle file.

    Args:
        filename (str): The name of the file to load the DataFrame from.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


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


def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)