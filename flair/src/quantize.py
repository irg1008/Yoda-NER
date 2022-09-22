from os import path

from flair.models import SequenceTagger
from torch import nn, qint8, quantization, save

from predict import load_model

# Keep in mind that flair does not support quantization so this is basically useless.


def quantize_model(model: SequenceTagger):
    cpu_model = model.eval()
    quantized_model = quantization.quantize_dynamic(
        cpu_model,
        {nn.Linear},
        dtype=qint8,
    )
    return quantized_model


def main():
    model_path = path.join(path.dirname(__file__), "../models/transformer")

    model = load_model(f"{model_path}/best-model.pt")
    quantized_model = quantize_model(model)
    save(quantized_model, f"{model_path}/best-model-quantized.pt")


if __name__ == "__main__":
    main()
