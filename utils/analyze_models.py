"""Run all configured direct model-to-human comparisons."""

from pathlib import Path

from config import (
    ANALYSIS_DISTANCE_METRICS,
    ANALYSIS_FAMILIES,
    BERT_EMBEDDING_DIRS,
    DECODER_DISPLAY_NAMES,
    DECODER_RESULTS_DIRS,
    HUMAN_DATA_DIR,
    OUTPUT_DIR,
    PAPER_ROUNDING,
    RANDOM_DIMENSIONS,
    RANDOM_SEED,
    STATIC_EMBEDDING_DIRS,
)
from olfactory.analysis import analyze_artifacts
from olfactory.artifacts import (
    discover_bert_artifacts,
    discover_decoder_artifacts,
    discover_static_artifacts,
    random_artifacts,
    unique_artifacts,
)
from olfactory.data import load_human_datasets


def configured_artifacts():
    artifacts = []
    for family in ("word2vec", "fasttext"):
        if family not in ANALYSIS_FAMILIES:
            continue
        for root in STATIC_EMBEDDING_DIRS.get(family, ()):
            artifacts.extend(discover_static_artifacts(Path(root), family))
    if "bert" in ANALYSIS_FAMILIES:
        for root in BERT_EMBEDDING_DIRS:
            artifacts.extend(discover_bert_artifacts(Path(root)))
    if "decoder" in ANALYSIS_FAMILIES:
        for root in DECODER_RESULTS_DIRS:
            artifacts.extend(discover_decoder_artifacts(Path(root), DECODER_DISPLAY_NAMES))
    if "random" in ANALYSIS_FAMILIES:
        artifacts.extend(random_artifacts(RANDOM_DIMENSIONS, RANDOM_SEED))
    return unique_artifacts(artifacts)


if __name__ == "__main__":
    datasets = load_human_datasets(HUMAN_DATA_DIR, paper_rounding=PAPER_ROUNDING)
    artifacts = configured_artifacts()
    if not artifacts:
        raise SystemExit("No model artifacts found. Check local_config.py.")
    print(f"Found {len(artifacts)} model artifacts")
    results, quality = analyze_artifacts(
        artifacts,
        datasets,
        metrics=ANALYSIS_DISTANCE_METRICS,
        paper_rounding=PAPER_ROUNDING,
    )
    analysis_dir = OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(analysis_dir / "direct_results.csv", index=False)
    quality.to_csv(analysis_dir / "artifact_quality.csv", index=False)
    print(f"Saved {len(results)} comparisons to {analysis_dir}")
