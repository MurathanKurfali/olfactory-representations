# Representations of Smells

This repository contains Python code and shareable research artifacts for
reproducing the direct model–human comparisons in **“Representations of smells:
The next frontier for language models?”**

It supports filtered Word2Vec and FastText embeddings, layer-wise BERT
extraction, pairwise prompting of OpenAI and Hugging Face decoder models, and
correlation of model distances with the available human similarity data.

## Repository contents

- `human_data_public/` — the publicly shareable Dravnieks distance matrix.
- `embeddings/` — the filtered Word2Vec and FastText files used in the study.
- `texts/olfactory_contexts/` — English and Swedish olfactory contexts used for
  contextual BERT representations.
- `olfactory/` — data loading, artifact discovery, embedding extraction, and
  analysis code.
- `utils/` — small runnable scripts for analysis and model generation.

Training Word2Vec and FastText models is not part of this repository. The
included files are already-trained embeddings filtered to the study vocabulary.
The COCA sentence-context archive used for some BERT conditions in the paper
cannot be redistributed and is not included.

## Human-data availability

The Dravnieks matrix can be shared publicly and is included as:

```text
human_data_public/DravniekSimilarity_matrix.csv
```

Two other human datasets used in the paper contain non-public participant data
and cannot be distributed in this repository:

- `RatingsSimilarity_matrix.csv` — odor-based similarity judgments.
- `QualtricsSimilarity_matrix.csv` — label-based similarity judgments.

With a public clone, the analysis therefore runs on the Dravnieks dataset only.
The loader automatically uses whichever of the three matrices are available.

Researchers with authorized access to the two private matrices can reproduce
the full three-dataset analysis by creating an ignored `human_data_private/`
folder containing all three files and an ignored `local_config.py` containing:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
HUMAN_DATA_DIR = PROJECT_ROOT / "human_data_private"
```

The private folder is excluded by `.gitignore` and must never be committed.

## Installation

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Analyzing the included static embeddings does not require an API key.
BERT and Hugging Face generation download models from the Hugging Face Hub.

## Analyze the included embeddings

From the repository root, run:

```bash
python -m utils.analyze_models
```

This discovers the included Word2Vec and FastText files, creates deterministic
random baselines, compares every model with the available human dataset(s), and
writes:

```text
outputs/analysis/direct_results.csv
outputs/analysis/artifact_quality.csv
```

`direct_results.csv` contains signed correlations, Fisher intervals, mean
absolute errors, pair counts, coverage, and model metadata. Both Pearson- and
cosine-based embedding distances are evaluated.

## Static embedding layout

The original analysis read the filtered files from `word2vec-filtered/` and
`fasttext-filtered/`. In this repository they are grouped under:

```text
embeddings/
├── word2vec-filtered/
│   ├── blogs/
│   ├── blogs-olf/
│   ├── coca/
│   ├── coca-olf/
│   ├── food_perfume_wine/
│   ├── food_perfume_wine-olf/
│   ├── now/
│   └── now-olf/
└── fasttext-filtered/
    └── ...same corpus structure...
```

Folders ending in `-olf` contain models trained on olfactory contexts only;
the others contain models trained on the corresponding full corpus. The
FastText filenames encode 300 dimensions, but the saved filtered vectors have
100 values; the analyzer reads the actual dimension and reports the discrepancy
in `artifact_quality.csv`.

## Extract BERT embeddings

The repository includes the olfactory contexts used by the extraction code.
Models, contexts, batch size, and revision can be selected in an ignored
`local_config.py`. For a small English smoke test:

```python
BERT_MODELS_TO_RUN = {"en": ("bert-base-cased",), "sv": ()}
BERT_CONTEXTS_TO_RUN = {"en": ("prompt6",), "sv": ()}
BERT_BATCH_SIZE = 8
```

Then run:

```bash
python -m utils.extract_bert_embeddings
python -m utils.analyze_models
```

Generated vectors are saved under `outputs/bert_embeddings/` and are included
automatically in the next analysis. Extraction is batched and resumable. Each
model directory records its model revision and run settings in
`run_metadata.json`.

The public default grid excludes the paper's `coca`, `coca100`, and `coca250`
conditions because their source contexts cannot be shared. Researchers with
authorized local access can configure those paths and conditions through
`local_config.py` without committing the context data.

## Query decoder models

The decoder scripts reproduce the paper's pairwise 0–1 similarity prompt for
all 3,336 descriptor pairs. Runs save incrementally and can be resumed.

For OpenAI, set `OPENAI_API_KEY` in the environment or in an ignored `keys.py`,
then run:

```bash
python -m utils.query_openai
```

For a Hugging Face causal language model, set the model name and optional
revision in `local_config.py`, then run:

```bash
python -m utils.query_huggingface
```

Decoder outputs are written below `outputs/decoder_results/` and are discovered
by the analysis script. A complete decoder run may require substantial compute
or API usage; model IDs and revisions should be recorded for reproducibility.

## Scope

This public repository covers the direct Python model–human comparisons. The
paper's R analyses, static-embedding training pipeline, and non-public human
matrices are not included. Random embeddings are regenerated deterministically
from the seed in `config.py`, so their exact values may differ slightly from the
particular random draw reported in the paper without affecting substantive
comparisons.

## Citation

If you use this repository, please cite:

> Kurfalı, M., Herman, P., Pierzchajlo, S., Olofsson, J., & Hörberg, T. (2025).
> Representations of smells: The next frontier for language models? *Cognition,
> 264*, 106243. https://doi.org/10.1016/j.cognition.2025.106243
