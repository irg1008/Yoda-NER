from os import path
from typing import Literal

from flair.data import Corpus, Dictionary
from flair.embeddings import StackedEmbeddings, TokenEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

import read_corpus

LabelType = Literal["ner", "pos"]


def get_sequence_tagger(
    embeddings: StackedEmbeddings, tag_dictionary: Dictionary, label_type: LabelType
) -> SequenceTagger:
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=label_type,
        use_crf=True,
    )
    return tagger


def get_embedding_stack(embedding="glove") -> StackedEmbeddings:
    embeddings: list[TokenEmbeddings] = [WordEmbeddings(embedding)]
    stack = StackedEmbeddings(embeddings=embeddings)
    return stack


def get_corpus_dict(corpus: Corpus, label_type: LabelType) -> Dictionary:
    return corpus.make_label_dictionary(label_type)


def train(
    tagger: SequenceTagger,
    corpus: Corpus,
    out: str,
    lr=0.1,
    epochs=20,
    batch_size=32,
):
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(out, learning_rate=lr, mini_batch_size=batch_size, max_epochs=epochs)


def main(corpus: Corpus):
    LABEL_TYPE: LabelType = "ner"

    label_dict = get_corpus_dict(corpus, LABEL_TYPE)
    embeddings = get_embedding_stack("es")
    tagger = get_sequence_tagger(embeddings, label_dict, LABEL_TYPE)

    model_path = path.join(path.dirname(__file__), "../models/")

    LEANRING_RATE = 0.1
    MAX_EPOCHS = 50
    BATCH_SIZE = 64

    train(
        tagger,
        corpus,
        out=model_path + "test",
        lr=LEANRING_RATE,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    corpus = read_corpus.main()
    main(corpus)
