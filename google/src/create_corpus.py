import json
import re
from os import path
from typing import TypedDict, TypeVar

import pandas as pd

Pos = tuple[int, int]
PosData = list[tuple[Pos, str]]

T = TypeVar("T", str, pd.Series)


class Annotation(TypedDict):
    endOffset: int
    startOffset: int
    displayName: str


class CorpusLine(TypedDict):
    text_segment_annotations: list[Annotation]
    textContent: str


def clean(text: T) -> T:
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
    for i in text:
        if i in filters:
            text = text.replace(i, " " + i)

    return text


def find_match_pos(sentence: str, word: str) -> list[Pos]:
    word = word.strip()
    matches: list[Pos] = [
        m.span()
        for m in re.finditer(r"\b" + word + r"\b", sentence, flags=re.IGNORECASE)
    ]
    return matches


def get_pos_data(
    sentence: str,
    features: list[str],
    features_name: list[str],
) -> PosData:
    pos_data: PosData = []

    for feat, name in zip(features, features_name):

        individual_feats = feat.split("/")

        for ind_feat in individual_feats:
            ind_feat = clean(ind_feat)
            pos_list = find_match_pos(sentence, ind_feat)

            for pos in pos_list:
                pos_data.append((pos, name))

    return pos_data


def get_annotations(pos_data: PosData) -> list[Annotation]:
    annotations: list[Annotation] = []

    for pos, name in pos_data:
        start, end = pos
        annotations.append(
            {
                "startOffset": start,
                "endOffset": end,
                "displayName": name,
            }
        )

    return annotations


def get_corpus_line(annotations: list[Annotation], sentence: str) -> CorpusLine:
    return {
        "text_segment_annotations": annotations,
        "textContent": sentence,
    }


def parse_corpus(split: pd.DataFrame) -> list[CorpusLine]:
    features_name = split.columns[1:]
    corpus: list[CorpusLine] = []

    for _, row in split.iterrows():
        sentence, features = row[0], row[1:]
        sentence = clean(sentence)

        pos_data: PosData = get_pos_data(
            sentence, features.to_list(), features_name.to_list()
        )

        annotations = get_annotations(pos_data)
        corpus_line = get_corpus_line(annotations, sentence)

        corpus.append(corpus_line)

    return corpus


def export_corpus(corpus: list[CorpusLine], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(json.dumps(line) + "\n")


def main():
    data_folder = path.join(path.dirname(__file__), "../data/")
    data = pd.read_csv(data_folder + "features.csv")
    corpus = parse_corpus(data)
    export_corpus(corpus, data_folder + "corpus.jsonl")


if __name__ == "__main__":
    main()
