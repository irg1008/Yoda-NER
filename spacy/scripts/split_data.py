import typer
import pandas as pd
import json
from pathlib import Path
from split_types import Entity, Entities, TitleData, TitlesData
import re

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


Pos = tuple[int, int]
PosData = list[tuple[Pos, str]]


def clean(text: str) -> str:
    """
    Just a helper fuction to add a space before the punctuations for better tokenization
    """
    filters = [
        "!",
        "#",
        "$",
        "%",
        "&",
        "(",
        ")",
        "*",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "_",
        "`",
        "{",
        "}",
        "~",
        "'",
    ]
    double_filters = ["/"]
    for i in text:
        if i in filters:
            text = text.replace(i, " " + i)
        if i in double_filters:
            text = text.replace(i, " " + i + " ")

    return text


def find_match_pos(sentence: str, word: str) -> list[Pos]:
    word = word.strip()
    matches: list[Pos] = [
        m.span()
        for m in re.finditer(
            r"\b" + re.escape(word) + r"\b", sentence, flags=re.IGNORECASE
        )
    ]
    return matches


def get_pos_data(
    sentence: str,
    features: list[str],
    features_name: list[str],
) -> PosData:
    pos_data: PosData = []

    for feat, name in zip(features, features_name):
        if feat == "null":
            continue

        individual_feats = str(feat).split(";")

        for ind_feat in individual_feats:
            ind_feat = clean(ind_feat)
            pos_list = find_match_pos(sentence, ind_feat)

            for pos in pos_list:
                pos_data.append((pos, name))

    return pos_data


def get_title_data(sentence: str, pos_data: PosData) -> TitleData:
    entities: Entities = []  # Initial, final and tag

    for (start, end), name in pos_data:
        entity: Entity = (start, end, name)
        entities.append(entity)

    return (sentence, {"entities": entities})


def get_titles_data(data: pd.DataFrame) -> TitlesData:
    tags = data.columns[1:]
    titles_data: TitlesData = []

    for _, row in data.iterrows():
        title, features = row[0], row[1:]
        pos_data = get_pos_data(title, features.to_list(), tags.to_list())
        title_data = get_title_data(title, pos_data)
        titles_data.append(title_data)

    return titles_data


def read_csv(csv_path: Path, sep=",") -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=sep)


def split_data(
    data: pd.DataFrame, train_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def get_sample(frac: float):
        return data.sample(frac=frac, replace=True)

    val_size = 1 - train_size

    return (
        get_sample(train_size),
        get_sample(val_size),
    )


def export_json(titles_data: TitlesData, json_path: Path):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(titles_data))


def main(csv_path: Path, train_path: Path, val_path: Path, train_split: float):
    data = read_csv(csv_path, sep=",")
    train_data, val_data = split_data(data, train_size=train_split)

    for split, out_path in zip([train_data, val_data], [train_path, val_path]):
        titles_data = get_titles_data(split)
        export_json(titles_data, out_path)

    print(
        f"> Successfully exported train and validation data to {train_path} and {val_path} with {len(train_data)} and {len(val_data)} titles respectively."
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
