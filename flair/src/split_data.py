import pandas as pd
from os import path

TRAIN, VAL = 0.7, 0.15


def split(
    data: pd.DataFrame, train_size: float, val_size: float, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert (
        train_size + val_size + test_size == 1.0
    ), "train, val, test sizes must sum to 1.0"

    def get_sample(frac: float):
        return data.sample(frac=frac, replace=True)

    return (
        get_sample(train_size),
        get_sample(val_size),
        get_sample(test_size),
    )


def export_csv(data: pd.DataFrame, csv_path: str):
    data.to_csv(csv_path, index=False)


def main():
    data_folder = path.join(path.dirname(__file__), "../data/")
    splits_folder = data_folder + "splits/"

    data = pd.read_csv(data_folder + "features.csv", sep=";")

    train_data, val_data, test_data = split(data, TRAIN, VAL, 1.0 - TRAIN - VAL)

    export_csv(train_data, splits_folder + "train.csv")
    export_csv(val_data, splits_folder + "val.csv")
    export_csv(test_data, splits_folder + "test.csv")


if __name__ == "__main__":
    main()
