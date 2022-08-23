from flair.data import Sentence
from flair.models import SequenceTagger
from os import path


def load_model(path: str) -> SequenceTagger:
    return SequenceTagger.load(path)


def get_sentence(text: str) -> Sentence:
    return Sentence(text)


def predict(model: SequenceTagger, sentence: Sentence) -> Sentence:
    model.predict(sentence)
    print(sentence.to_tagged_string())
    return sentence


def main(text):
    model_path = path.join(path.dirname(__file__), "../models/")
    model = load_model(model_path + "test/best-model.pt")

    sentence = get_sentence(text)
    predict(model, sentence)


if __name__ == "__main__":
    text = "Cremallera roja/verde talla 45"

    main(text)
