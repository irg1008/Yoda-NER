from os import path

from flair.models import SequenceTagger
from torch import onnx, randn

from predict import load_model


def to_onnx(model: SequenceTagger, out_path: str):
    dummy_input = randn(
        100,
        100,
        3,
    )
    
    model.eval()

    onnx.export(
        model,
        dummy_input,
        out_path,
    )


def main():
    model_path = path.join(path.dirname(__file__), "../models/lite")
    best_model_path = path.join(model_path, "best-model.pt")
    onnx_model_path = best_model_path.replace(".pt", ".onnx")

    model = load_model(best_model_path)
    print(model)

    to_onnx(model, onnx_model_path)


if __name__ == "__main__":
    main()
