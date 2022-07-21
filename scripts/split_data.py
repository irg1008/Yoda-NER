import typer
import pandas as pd
import json
from pathlib import Path


from split_types import Entity, Entities, TitleData, TitlesData


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


def find_word_start_end(title: str, word: str) -> tuple[int, int]:
    words = title.lower().split()
    index = words.index(word.lower())

    start = sum(len(word) + 1 for word in words[:index])
    double_spaces = title[:start].count("  ")
    start += double_spaces
    
    end = start + len(word)
    
    return start, end


def is_word_in_title(title: str, word: str) -> bool:
    title_words = title.lower().split()
    return word.lower() in title_words


def get_title_data(title: str, features: pd.Series, tags: pd.Index) -> TitleData:
    entities: Entities = []  # Initial, final and tag

    for tag, feature in zip(tags, features):
        if feature != UNKNOWN_DATA and is_word_in_title(title, feature):
            start, end = find_word_start_end(title, feature)
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


def read_csv(csv_path: Path, sep=";") -> pd.DataFrame:
    return pd.read_csv(csv_path, sep=sep)


def split(titles_data: TitlesData, train_size: float) -> tuple[TitlesData, TitlesData]:
    split_point = int(len(titles_data) * train_size)
    train_data, val_data = titles_data[:split_point], titles_data[split_point:]
    return train_data, val_data


def export_json(titles_data: TitlesData, json_path: Path):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(titles_data))


def main(csv_path: Path, train_path: Path, val_path: Path):
    data = read_csv(csv_path)
    titles_data = get_titles_data(data)

    train_data, val_data = split(titles_data, train_size=0.6)

    export_json(train_data, train_path)
    export_json(val_data, val_path)

    print(
        f"> Successfully exported train and validation data to {train_path} and {val_path}"
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
