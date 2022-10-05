from os import path
import pandas as pd


def get_data():
    aug_dir = path.join(path.dirname(__file__), "../../data/augmented")

    # Read in data
    data_path = path.join(path.dirname(__file__), "../../data/db_data_100k.csv")
    data = pd.read_csv(data_path)

    return data, aug_dir
