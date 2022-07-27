<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove): python -m spacy project document --output README.md -->

# 🪐 spaCy Project: Yoda Ner

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `split` | Convert from custom data to train and val data |
| `convert` | Convert the data to spaCy's binary format |
| `create-config` | Create a config for training |
| `train` | Train the NER model |
| `evaluate` | Evaluate the model and export metrics |
| `package` | Package the trained model as a pip package |
| `view` | Visualize the model's output interactively using Streamlit |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `split` &rarr; `convert` &rarr; `create-config` &rarr; `train` &rarr; `evaluate` &rarr; `package` &rarr; `view` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/train.json`](assets/train.json) | Local | Training data converted with convert data script in src folder |
| [`assets/val.json`](assets/val.json) | Local | Validation data converted with convert data script in src folder |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->