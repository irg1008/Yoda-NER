"""Module to convert json spacy data to binary spacy data"""

import warnings
import json
from pathlib import Path
import typer
import spacy
from spacy.tokens import DocBin, Span, Doc
from split_types import TitlesData


def get_json_data(path: Path) -> TitlesData:
    """
    Loads json data from a file.

    Args:
        path (Path): Path to the json file.

    Returns:
        TitlesData: A list of tuples of the form (title, data).
    """
    return json.loads(path.read_text())


def get_no_span_msg(start: int, end: int, label: str, doc: Doc, title: str) -> str:
    """
    Returns a warning message when a span is not found.

    Args:
        start (int): Start index of the span.
        end (int): End index of the span.
        label (str): Label of the span.
        doc (Doc): Spacy document.
        title (str): Title of the document.

    Returns:
        str: Warning message.
    """
    return f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(title)}\n"


def main(lang: str, json_path: Path, spacy_path: Path):
    """
    Main function.

    Args:
        lang (str): Language of the model.
        json_path (Path): Path to the json file.
        spacy_path (Path): Path to the spacy model.
    """
    nlp = spacy.blank(lang)
    db = DocBin()

    titles_data = get_json_data(json_path)
    for title, data in titles_data:
        doc = nlp.make_doc(title)
        entities: list[Span] = []

        for start, end, label in data["entities"]:
            span = doc.char_span(start, end, label=label)

            if span:
                entities.append(span)
            else:
                msg = get_no_span_msg(start, end, label, doc, title)
                warnings.warn(msg)

        doc.ents = tuple(entities)
        db.add(doc)

    db.to_disk(spacy_path)
    print(f"> Successfully exported binary spacy data to {spacy_path}")


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
