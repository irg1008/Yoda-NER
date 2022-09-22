from flair.data import Sentence
from flair.models import SequenceTagger
from os import path


def load_model(path: str) -> SequenceTagger:
    return SequenceTagger.load(path)


def get_sentence(text: str) -> Sentence:
    return Sentence(text)


def predict(model: SequenceTagger, sentence: Sentence) -> Sentence:
    model.predict(sentence)
    return sentence


def main(text: str):
    model_path = path.join(path.dirname(__file__), "../models/transformer")
    model = load_model(path.join(model_path, "best-model.pt"))

    sentence = get_sentence(text)
    predict(model, sentence)

    print(sentence.to_tagged_string())


if __name__ == "__main__":
    text = "Apple Watch Series 7 (GPS) - Caja de Aluminio (Product) Red de 41 mm - Correa Deportiva (Product) Red - Talla única"

    main(text)
