from unittest.mock import CallableMixin
import typer
import pandas as pd
import json
from pathlib import Path
from random import shuffle
from split_types import Entity, Entities, TitleData, TitlesData
import re
from typing import Union, Callable

# IMPORTANT: First column must be title column. Rest can be any N features wanted.

# Output format:
# ```json
#   [
#     [
#       "Where is Berlin?",
#       { "entities": [[9, 15, "LOC"], ...] }
#     ],
#     [
#       ...
#     ]
#   ]
# ```

UNKNOWN_DATA = "unk"


def find_word_start_end(title: str, word: str) -> Union[tuple[int, int], None]:
    search = re.search(f" {word.lower()} ", f" {title.lower()} ")
    if not search:
        return None

    start, end = search.span()
    return start, end - 2


def get_title_data(title: str, features: pd.Series, tags: pd.Index) -> TitleData:
    entities: Entities = []  # Initial, final and tag

    for tag, feature in zip(tags, features):
        start_end = find_word_start_end(title, feature)
        if feature != UNKNOWN_DATA and start_end:
            start, end = start_end
            entity: Entity = (start, end, tag)
            entities.append(entity)

    return (title, {"entities": entities})


def get_titles_data(data: pd.DataFrame) -> TitlesData:
    tags = data.columns[1:]
    titles_data: TitlesData = []

    for _, row in data.iterrows():
        title, features = row[0], row[1:]
        title_data = get_title_data(title, features, tags)
        titles_data.append(title_data)

    return titles_data


def read_csv(csv_path: Path, sep=",") -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=sep)


def split(titles_data: TitlesData, train_size: float) -> tuple[TitlesData, TitlesData]:
    split_point = int(len(titles_data) * train_size)
    shuffle(titles_data)
    train_data, val_data = titles_data[:split_point], titles_data[split_point:]
    return train_data, val_data


def export_json(titles_data: TitlesData, json_path: Path):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(titles_data))


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


def main(csv_path: Path, train_path: Path, val_path: Path, train_split: float):
    data = read_csv(csv_path, sep=",")

    duplicated_data = duplicate_data(data)
    titles_data = get_titles_data(duplicated_data)
    train_data, val_data = split(titles_data, train_size=train_split)

    export_json(train_data, train_path)
    export_json(val_data, val_path)

    print(
        f"> Successfully exported train and validation data to {train_path} and {val_path} with {len(train_data)} and {len(val_data)} titles respectively."
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
