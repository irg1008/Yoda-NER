from flair.data import Corpus
from flair.datasets import ColumnCorpus
from os import path


def get_corpus(folder: str, train_file: str, test_file: str, val_file: str) -> Corpus:
    columns = {0: "text", 1: "ner"}
    return ColumnCorpus(
        folder,
        columns,
        train_file=train_file,
        test_file=test_file,
        dev_file=val_file,
    )


def main():
    corpus_folder = path.abspath(path.join(path.dirname(__file__), "../data/corpus"))

    corpus = get_corpus(corpus_folder, "train.txt", "test.txt", "val.txt")
    return corpus


if __name__ == "__main__":
    main()
