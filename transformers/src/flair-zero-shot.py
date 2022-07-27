from flair.models import TARSTagger
from flair.data import Sentence

# 1. Load zero-shot NER tagger
tars = TARSTagger.load("tars-ner")

# 2. Prepare some test sentences
sentences = [
    Sentence(
        "I bought a xs red coat that would fit me perfectly. I was very happy with the coat."
    ),
    Sentence("Men's trausers made with nylon for better swimming and fitness"),
    Sentence("Pantalón de mujer negro talla 38"),
    Sentence("Green Women cup size B made with whool"),
]

# 3. Define some classes of named entities such as "soccer teams", "TV shows" and "rivers"
labels = [
    "Color",
]
tars.add_and_switch_to_new_task("task 1", labels, label_type="ner")

# 4. Predict for these classes and print results
for sentence in sentences:
    tars.predict(sentence)
    print(sentence.to_tagged_string("ner"))
