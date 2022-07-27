import spacy_streamlit
import typer


def main(models: str, default_text: str):
    models = [name.strip() for name in models.split(",")]
    default_text = default_text[1:-1]
    visualizers = ["ner"]

    spacy_streamlit.visualize(
        models,
        default_text,
        visualizers=visualizers,
        show_visualizer_select=False,
        show_logo=False,
        color="red",
        sidebar_title="Yoda NER"
    )


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
