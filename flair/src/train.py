from os import path
from typing import Literal

from flair.data import Corpus, Dictionary
from flair.embeddings import (
    StackedEmbeddings,
    TokenEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
    ELMoEmbeddings,
    FlairEmbeddings,
    PooledFlairEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from torch.optim import SGD

import read_corpus

LabelType = Literal["ner", "pos"]


def get_sequence_tagger(
    embeddings: StackedEmbeddings,
    tag_dictionary: Dictionary,
    label_type: LabelType,
    dropout=0.0,
) -> SequenceTagger:
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=label_type,
        use_crf=True,
        dropout=dropout,
    )
    return tagger


def get_embedding_stack(embeddings: list[TokenEmbeddings]) -> StackedEmbeddings:
    stack = StackedEmbeddings(embeddings=embeddings)
    return stack


def get_corpus_dict(corpus: Corpus, label_type: LabelType) -> Dictionary:
    return corpus.make_label_dictionary(label_type)


def train(
    tagger: SequenceTagger,
    corpus: Corpus,
    out: str,
    lr=0.1,
    epochs=15,
    batch_size=32,
):
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(
        out,
        learning_rate=lr,
        mini_batch_size=batch_size,
        max_epochs=epochs,
        use_tensorboard=True,
        tensorboard_log_dir=f"{out}/tensorboard",
        optimizer=SGD,
    )


# List of embeddings: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md
# Examples:
#   - https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner
def get_embeddings() -> list[TokenEmbeddings]:
    flair_forward_embedding = PooledFlairEmbeddings("es-forward")
    flair_backward_embedding = PooledFlairEmbeddings("es-backward")
    transformer_embedding = TransformerWordEmbeddings("dccuchile/bert-base-spanish-wwm-cased")
    # word_embedding = WordEmbeddings("es-crawl")
    return [
        transformer_embedding,
        flair_forward_embedding,
        flair_backward_embedding,
        # word_embedding,
    ]


def main(corpus: Corpus):
    LABEL_TYPE: LabelType = "ner"

    label_dict = get_corpus_dict(corpus, LABEL_TYPE)

    embeddings = get_embeddings()
    stack_embedding = get_embedding_stack(embeddings)
    tagger = get_sequence_tagger(stack_embedding, label_dict, LABEL_TYPE)

    model_path = path.join(path.dirname(__file__), "../models/")

    LEANRING_RATE = 0.1
    MAX_EPOCHS = 10
    BATCH_SIZE = 32

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
