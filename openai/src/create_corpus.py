import json
from os import path
from typing import TypedDict

import pandas as pd


class CorpusLine(TypedDict):
    prompt: str
    completion: str


def parse_corpus(split: pd.DataFrame) -> list[CorpusLine]:
    corpus: list[CorpusLine] = []

    for _, row in split.iterrows():
        sentence, features = row[0], row[1:]

        corpus_line = CorpusLine(prompt=sentence, completion="\n".join(features))
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
