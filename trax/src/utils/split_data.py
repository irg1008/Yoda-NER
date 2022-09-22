from typing import Callable
import pandas as pd


def split(X: list, Y: list, train_size: float, val_size: float):
    assert train_size + val_size <= 1.0, "train, val, test sizes must sum to 1.0"
    test_size = 1 - train_size - val_size

    def get_sample(df: pd.DataFrame, frac: float):
        return df.sample(frac=frac, replace=True) if frac > 0 else pd.DataFrame()

    def get_data_Sample(data: list):
        df = pd.DataFrame(data)
        return (
            get_sample(df, train_size),
            get_sample(df, val_size),
            get_sample(df, test_size),
        )

    return get_data_Sample(X), get_data_Sample(Y)


def export_csv(data: pd.DataFrame, csv_path: str):
    data.to_csv(csv_path, index=False)


def duplicate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Duplicate data with uppercase, capitalize and lowercase title.
    Then concat all those in a single dataframe.
    Args:
        data (pd.DataFrame): Data to duplicate.
    Returns:
        pd.DataFrame: Duplicated data
    """

    def copy(data: pd.DataFrame, str_op: Callable):
        data_copy = data.copy()
        data_copy.title = data_copy.title.apply(str_op)
        return data_copy

    data_upper = copy(data, str_op=str.upper)
    data_capitalize = copy(data, str_op=str.capitalize)
    data_lower = copy(data, str_op=str.lower)

    return pd.concat([data, data_upper, data_capitalize, data_lower])
