from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification

pretrained_model_name = "bert-base-multilingual-cased"


def get_model():
    return AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(pretrained_model_name)
