import pandas as pd


def split(data: pd.DataFrame, train_size: float, val_size: float):
    assert train_size + val_size <= 1.0, "train, val and test sizes must sum to 1.0"
    test_size = 1 - train_size - val_size

    def get_sample(df: pd.DataFrame, frac: float):
        return df.sample(frac=frac, replace=True) if frac > 0 else pd.DataFrame()

    return (
        get_sample(data, train_size),
        get_sample(data, val_size),
        get_sample(data, test_size),
    )
