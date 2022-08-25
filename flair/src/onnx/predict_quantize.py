import torch
from os import path


def main(text: str):
    model_path = path.join(path.dirname(__file__), "../../models/semi-lite")
    quantized_model = torch.load(f"{model_path}/best-model-quantized.pt")

    sentence = torch.Tensor(text)
    result = quantized_model(sentence)

    return result


if __name__ == "__main__":
    text = "ImseVimse - Bañador-pañal imsevimse con volante L azul marino / gris"
    result = main(text)
    print(result)
