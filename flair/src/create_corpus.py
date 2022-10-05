import re
from os import path

import pandas as pd
from tqdm import tqdm

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


def mark_sentence(sentence: str, pos_data: PosData) -> dict:
    """Mark all words in sentence with BIO scheme

    Args:
        sentence (str): sentence to mark
        pos_data (PosData): list of tuples (word, pos, name)

    Returns:
        str: marked sentence
    """

    word_dict = {}

    # Init on default value.
    for word in sentence.split():
        word_dict[word] = "O"

    for pos, name in pos_data:
        start, end = pos
        match_words = sentence[start:end].split()
        word_dict[match_words[0]] = "B-" + name

        for word in match_words[1:]:
            word_dict[word] = "I-" + name

    return word_dict


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


def parse_corpus(split: pd.DataFrame) -> list[dict]:
    features_name = split.columns[1:]
    markings: list[dict] = []

    print("Parsing corpus")
    for _, row in tqdm(split.iterrows(), total=len(split)):
        sentence, features = row[0], row[1:]
        sentence = clean(sentence)

        pos_data: PosData = get_pos_data(
            sentence, features.to_list(), features_name.to_list()
        )

        marking = mark_sentence(sentence, pos_data)
        markings.append(marking)

    return markings


def export_corpus(corpus: list[dict], filename: str) -> None:
    print("Exporting corpus")
    with open(filename, "w", encoding="utf-8") as f:
        for marking in tqdm(corpus):
            for word, tag in marking.items():
                f.write(word + " " + tag + "\n")
            f.write("\n")


def main():
    data_folder = path.join(path.dirname(__file__), "../data/")
    splits_folder = data_folder + "splits/"
    corpus_folder = data_folder + "corpus/"

    for name in ["train", "val", "test"]:
        data = pd.read_csv(splits_folder + name + ".csv", dtype=str).fillna("null")
        corpus = parse_corpus(data)
        export_corpus(corpus, corpus_folder + name + ".txt")


if __name__ == "__main__":
    main()
