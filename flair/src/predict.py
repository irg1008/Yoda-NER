from flair.data import Sentence
from flair.models import SequenceTagger
from os import path
import torch


def load_model(path: str) -> SequenceTagger:
    return SequenceTagger.load(path)


def get_sentence(text: str) -> Sentence:
    return Sentence(text)


def predict(model: SequenceTagger, sentence: Sentence) -> Sentence:
    model.predict(sentence)
    return sentence


def save_pytorch_model(flair_model: SequenceTagger, path: str):
    flair_model.save(f"{path}/pytorch_model.bin", checkpoint=False)


def main(text: str):
    model_path = path.join(path.dirname(__file__), "../models/s")
    model = load_model(path.join(model_path, "best-model.pt"))

    model.eval()

    sentence = get_sentence(text)
    predict(model, sentence)

    print(sentence.to_tagged_string())

    # Save pytorch_model.bin
    save_pytorch_model(model, model_path)


if __name__ == "__main__":
    text = "Jean Paul Gaultier Classique - 50 ML Eau de Parfum  Damen Parfum"

    main(text)
