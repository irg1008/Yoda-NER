import json
from os import path

import pandas as pd
from tqdm import tqdm

from data.corpus import parse_corpus
from data.inputs import get_vectors

from utils.data import split


def create_inputs(corpus):
    X, _, _, Y, _, _ = get_vectors(corpus)
    return X, Y


def export_inputs(data: pd.DataFrame, out_path: str):
    corpus = parse_corpus(data)
    X, Y = create_inputs(corpus)

    with open(out_path, "w") as f:
        print("Exporting file")
        for idx, (x, y) in tqdm(enumerate(zip(X, Y)), total=len(X)):
            f.write(json.dumps({"id": str(idx), "tokens": x, "ner_tags": y}) + "\n")


if __name__ == "__main__":
    data_path = path.join(path.dirname(__file__), "../data")

    in_path = path.join(data_path, "entities.csv")
    out_path = path.join(data_path, "inputs")

    data = pd.read_csv(in_path)
    splits = split(data, train_size=0.7, val_size=0.2)

    for df, name in zip(splits, ["train", "val", "test"]):
        out_file = f"{name}.jsonl"
        print(f"Exporting {name} data")
        export_inputs(df, path.join(out_path, out_file))
        print(f"Exported {name} data successfully \n")
