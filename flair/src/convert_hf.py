from transformers.models.bert.modeling_bert import BertPreTrainedModel
from os import path
from flair.data import Sentence
from flair.models import SequenceTagger


def save_pytorch_model(model_path: str, out_path: str):
    model = BertPreTrainedModel.from_pretrained(
        model_path, cache_dir=None, from_tf=False, state_dict=None
    )
    model.save_pretrained(out_path)


def main():
    model_path = path.join(path.dirname(__file__), "../models/mix_trans_word")
    save_pytorch_model(model_path, model_path)


def download():
    model = SequenceTagger.load("lighthousefeed/yoda-ner")
    sentence = Sentence(
        "Jean Paul Gaultier Classique - 50 ML Eau de Parfum  Damen Parfum"
    )
    model.predict(sentence)
    print(sentence.to_tagged_string())


def infer():
    import requests

    API_URL = "https://api-inference.huggingface.co/models/lighthousefeed/yoda-ner"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query(
        {
            "inputs": "My name is Sarah Jessica Parker but you can call me Jessica",
        }
    )

    print(output)


if __name__ == "__main__":
    download()
