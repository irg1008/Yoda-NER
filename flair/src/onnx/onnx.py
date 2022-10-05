import sys
from os import path

from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from torch import onnx, randn

sys.path.append(path.join(path.dirname(__file__), ".."))
from predict import load_model


def to_onnx(model: SequenceTagger, out_path: str):
    assert isinstance(
        model.embeddings, (TransformerWordEmbeddings, TransformerDocumentEmbeddings)
    )
    sentences = [
        Sentence("This is a sentence."),
        Sentence(
            "This is a way longer sentence to ensure varying lengths work with LSTM"
        ),
    ]

    model.embeddings = model.embeddings.export_onnx(
        out_path,
        sentences,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def main():
    model_path = path.join(path.dirname(__file__), "../../models/mix_trans_word")
    best_model_path = path.join(model_path, "best-model.pt")

    model = load_model(best_model_path)

    onnx_model_path = best_model_path.replace(".pt", ".onnx")
    to_onnx(model, onnx_model_path)


if __name__ == "__main__":
    main()
