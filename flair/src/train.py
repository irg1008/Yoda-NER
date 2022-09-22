from os import path
from typing import Literal, Union

from flair.data import Corpus, Dictionary
from flair.embeddings import (
    ELMoTransformerEmbeddings,
    FlairEmbeddings,
    PooledFlairEmbeddings,
    StackedEmbeddings,
    TokenEmbeddings,
    TransformerWordEmbeddings,
    WordEmbeddings,
)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.training_utils import AnnealOnPlateau
from torch.optim import SGD, Adam

import read_corpus

LabelType = Literal["ner", "pos"]


def get_sequence_tagger(
    embeddings: Union[StackedEmbeddings, TokenEmbeddings],
    tag_dictionary: Dictionary,
    label_type: LabelType,
    dropout=0.0,
) -> SequenceTagger:
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=label_type,
        use_crf=True,  # https://en.wikipedia.org/wiki/Conditional_random_field
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
        scheduler=AnnealOnPlateau,
        num_workers=6,
    )


# List of embeddings: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_4_ELMO_BERT_FLAIR_EMBEDDING.md
# Examples:
#   - https://huggingface.co/mrm8488/bert-spanish-cased-finetuned-ner
def get_embeddings() -> list[TokenEmbeddings]:
    pool_flair_forward_embedding = PooledFlairEmbeddings("es-forward")
    pool_flair_backward_embedding = PooledFlairEmbeddings("es-backward")
    flair_forward_embedding = FlairEmbeddings("es-forward")
    flair_backward_embedding = FlairEmbeddings("es-backward")
    word_embedding = WordEmbeddings("es-crawl")

    sm_flair_forward_embedding = FlairEmbeddings("es-forward-fast")
    sm_flair_backward_embedding = FlairEmbeddings("es-backward-fast")
    sm_word_embedding = WordEmbeddings("es")
    glove_word_embedding = WordEmbeddings("glove")

    transformer_embedding = TransformerWordEmbeddings("bert-base-multilingual-cased")

    return [
        transformer_embedding,
        # pool_flair_forward_embedding,
        # pool_flair_backward_embedding,
        # flair_forward_embedding,
        # flair_backward_embedding,
        # word_embedding,
        # sm_flair_forward_embedding,
        # sm_flair_backward_embedding,
        # sm_word_embedding,
        # glove_word_embedding,
    ]


def main(corpus: Corpus):
    LABEL_TYPE: LabelType = "ner"
    TRANSFORMER_ONLY = True

    label_dict = get_corpus_dict(corpus, LABEL_TYPE)

    embeddings = get_embeddings()
    stack_embedding = (
        embeddings[0] if TRANSFORMER_ONLY else get_embedding_stack(embeddings)
    )
    tagger = get_sequence_tagger(stack_embedding, label_dict, LABEL_TYPE)

    model_path = path.join(path.dirname(__file__), "../models/")

    LEANRING_RATE = 0.01
    MAX_EPOCHS = 50
    BATCH_SIZE = 4

    train(
        tagger,
        corpus,
        out=model_path + "transformer",
        lr=LEANRING_RATE,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    corpus = read_corpus.main()
    main(corpus)
