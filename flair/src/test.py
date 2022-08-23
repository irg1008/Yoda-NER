from flair.data import Sentence
from flair.models import MultiTagger

# load tagger for POS and NER
tagger = MultiTagger.load(["pos", "ner"])

# make example sentence
sentence = Sentence("George Washington went to Washington on 170mm motorbike.")

# predict with both models
tagger.predict(sentence)

print(sentence)

# iterate over entities and print each
for label in sentence.get_labels("pos"):
    print(label)

for label in sentence.get_labels("ner"):
    print(label)
