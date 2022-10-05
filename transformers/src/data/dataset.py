from datasets.load import load_dataset
from os import path


def get_dataset():
    data_dir = path.join(path.dirname(__file__), "../../data/inputs")
    return load_dataset(
        "json",
        data_dir=data_dir,
        data_files={"train": "train.jsonl", "test": "test.jsonl", "val": "val.jsonl"},
    )
