from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-large-finetuned-conll03-english"
)
classifier = pipeline("ner", model=model, tokenizer=tokenizer)
result = classifier("Hello I'm Omar and I live in Zürich.")

print(result)
