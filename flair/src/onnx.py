from os import path

from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from torch import onnx, randn

from predict import load_model


# TODO: https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_13_TRANSFORMERS_PRODUCTION.md

def to_onnx(model: SequenceTagger, out_path: str):

    assert isinstance(
        model.embeddings, (TransformerWordEmbeddings)
    ), "Model must be transformer"

    model.embeddings = model.


def main():
    model_path = path.join(path.dirname(__file__), "../models/transformer")
    best_model_path = path.join(model_path, "best-model.pt")
    onnx_model_path = best_model_path.replace(".pt", ".onnx")

    model = load_model(best_model_path)
    print(model)

    to_onnx(model, onnx_model_path)


if __name__ == "__main__":
    main()
