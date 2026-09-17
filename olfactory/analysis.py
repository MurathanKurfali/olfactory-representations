"""Direct model-to-human comparisons used by the Python analysis pipeline."""

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .artifacts import ModelArtifact
from .data import HumanDataset
from .decoders import read_decoder_scores
from .embeddings import read_text_embeddings
from .vocabulary import (
    BERT_WORDS_EN,
    BERT_WORDS_SV,
    SWEDISH_TO_ENGLISH,
    paper_decoder_pairs,
)


@dataclass(frozen=True)
class LoadedModel:
    pair_distances_by_metric: dict[str, dict[tuple[str, str], float]]
    vector_dimension: int | None
    loaded_words: int
    duplicate_records: int = 0
    decoder_unique_pairs: int | None = None
    decoder_missing_paper_pairs: int | None = None
    decoder_extra_pairs: int | None = None


@dataclass(frozen=True)
class Comparison:
    correlation: float
    fisher_ci_low: float
    fisher_ci_high: float
    pair_count: int
    word_count: int
    expected_pairs: int
    coverage: float
    mean_absolute_error: float


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A zero-variance or zero-length embedding cannot be compared")
    return matrix / norms


def embedding_pair_distances(
    embeddings: dict[str, np.ndarray],
    metric: str,
    paper_rounding: bool = True,
) -> dict[tuple[str, str], float]:
    labels = sorted(embeddings)
    matrix = np.vstack([embeddings[label] for label in labels]).astype(float)
    if metric == "pearson":
        matrix = matrix - matrix.mean(axis=1, keepdims=True)
    elif metric != "cosine":
        raise ValueError(f"Unknown distance metric: {metric!r}")
    normalized = _row_normalize(matrix)
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distances = 0.5 * (1.0 - similarities)
    if paper_rounding:
        distances = np.round(distances, 5)

    return {
        (labels[i], labels[j]): float(distances[i, j])
        for i, j in combinations(range(len(labels)), 2)
    }


def decoder_pair_distances(
    scores: dict[tuple[str, str], float],
    paper_rounding: bool = True,
) -> dict[tuple[str, str], float]:
    distances = {pair: 1.0 - score for pair, score in scores.items()}
    if paper_rounding:
        distances = {pair: round(value, 5) for pair, value in distances.items()}
    return distances


def _fisher_interval(correlation: float, sample_size: int, alpha: float = 0.05) -> tuple[float, float]:
    if sample_size <= 3 or not -1.0 < correlation < 1.0:
        return correlation, correlation
    z = np.arctanh(correlation)
    critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    margin = critical / np.sqrt(sample_size - 3)
    return float(np.tanh(z - margin)), float(np.tanh(z + margin))


def compare_pair_distances(
    model_distances: dict[tuple[str, str], float],
    dataset: HumanDataset,
) -> Comparison:
    shared_pairs = sorted(set(model_distances) & set(dataset.pair_distances))
    if len(shared_pairs) < 4:
        raise ValueError(f"Only {len(shared_pairs)} shared pairs for {dataset.spec.key}")
    human_values = np.asarray([dataset.pair_distances[pair] for pair in shared_pairs])
    model_values = np.asarray([model_distances[pair] for pair in shared_pairs])
    correlation = float(np.corrcoef(human_values, model_values)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError(f"Undefined correlation for {dataset.spec.key}")
    ci_low, ci_high = _fisher_interval(correlation, len(shared_pairs))
    words = {word for pair in shared_pairs for word in pair}
    return Comparison(
        correlation=correlation,
        fisher_ci_low=ci_low,
        fisher_ci_high=ci_high,
        pair_count=len(shared_pairs),
        word_count=len(words),
        expected_pairs=dataset.expected_pair_count,
        coverage=len(shared_pairs) / dataset.expected_pair_count,
        mean_absolute_error=float(np.mean(np.abs(human_values - model_values))),
    )


def absolute_errors(
    model_distances: dict[tuple[str, str], float],
    dataset: HumanDataset,
) -> dict[tuple[str, str], float]:
    shared = set(model_distances) & set(dataset.pair_distances)
    return {
        pair: abs(model_distances[pair] - dataset.pair_distances[pair])
        for pair in shared
    }


def _canonicalize_vectors(
    vectors: dict[str, np.ndarray],
    language: str,
) -> dict[str, np.ndarray]:
    if language == "en":
        return {word.lower(): vector for word, vector in vectors.items()}
    if language != "sv":
        raise ValueError(f"Unsupported artifact language: {language}")
    return {
        SWEDISH_TO_ENGLISH[word.lower()]: vector
        for word, vector in vectors.items()
        if word.lower() in SWEDISH_TO_ENGLISH
    }


def load_model(
    artifact: ModelArtifact,
    metrics: tuple[str, ...],
    paper_rounding: bool,
) -> LoadedModel:
    if artifact.family == "random":
        assert artifact.random_dimension is not None and artifact.random_seed is not None
        generator = np.random.default_rng(artifact.random_seed + artifact.random_dimension)
        vectors = {
            word: generator.uniform(-0.05, 0.05, artifact.random_dimension)
            for word in BERT_WORDS_EN
        }
        return LoadedModel(
            {
                metric: embedding_pair_distances(vectors, metric, paper_rounding)
                for metric in metrics
            },
            artifact.random_dimension,
            len(vectors),
        )

    if artifact.source_path is None:
        raise ValueError(f"No source path for {artifact.artifact_id}")

    if artifact.family == "decoder":
        scores, report = read_decoder_scores(
            artifact.source_path,
            language=artifact.language,
            paper_rounding=paper_rounding,
        )
        distances = decoder_pair_distances(scores, paper_rounding)
        expected_pairs = set(paper_decoder_pairs("en"))
        observed_pairs = set(scores)
        return LoadedModel(
            pair_distances_by_metric={metric: distances for metric in metrics},
            vector_dimension=None,
            loaded_words=len({word for pair in scores for word in pair}),
            duplicate_records=report.identical_duplicates,
            decoder_unique_pairs=len(observed_pairs),
            decoder_missing_paper_pairs=len(expected_pairs - observed_pairs),
            decoder_extra_pairs=len(observed_pairs - expected_pairs),
        )

    wanted = set(BERT_WORDS_SV if artifact.language == "sv" else BERT_WORDS_EN)
    vectors, metadata = read_text_embeddings(artifact.source_path, wanted)
    canonical = _canonicalize_vectors(vectors, artifact.language)
    return LoadedModel(
        {
            metric: embedding_pair_distances(canonical, metric, paper_rounding)
            for metric in metrics
        },
        metadata.actual_dimension,
        len(canonical),
    )


def analyze_artifacts(
    artifacts: list[ModelArtifact],
    datasets: dict[str, HumanDataset],
    metrics: tuple[str, ...],
    paper_rounding: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []

    for index, artifact in enumerate(artifacts, start=1):
        if index == 1 or index % 50 == 0 or index == len(artifacts):
            print(f"[{index}/{len(artifacts)}] {artifact.artifact_id}")
        loaded = load_model(artifact, metrics, paper_rounding)
        quality_rows.append(
            {
                "artifact_id": artifact.artifact_id,
                "family": artifact.family,
                "source_path": str(artifact.source_path or "generated"),
                "loaded_words": loaded.loaded_words,
                "actual_dimension": loaded.vector_dimension,
                "claimed_dimension": artifact.claimed_dimension,
                "dimension_matches_filename": (
                    artifact.claimed_dimension == loaded.vector_dimension
                    if artifact.claimed_dimension and loaded.vector_dimension
                    else ""
                ),
                "exact_duplicate_records": loaded.duplicate_records,
                "decoder_unique_pairs": loaded.decoder_unique_pairs,
                "decoder_missing_paper_pairs": loaded.decoder_missing_paper_pairs,
                "decoder_extra_pairs": loaded.decoder_extra_pairs,
            }
        )
        for metric in metrics:
            for dataset in datasets.values():
                comparison = compare_pair_distances(
                    loaded.pair_distances_by_metric[metric], dataset
                )
                row = artifact.record()
                row.update(asdict(comparison))
                row.update(
                    {
                        "distance_metric": metric,
                        "dataset": dataset.spec.key,
                        "dataset_name": dataset.spec.display_name,
                        "actual_dimension": loaded.vector_dimension,
                    }
                )
                rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["distance_metric", "family", "artifact_id", "dataset"]
    )
    quality = pd.DataFrame(quality_rows).sort_values(["family", "artifact_id"])
    return results.reset_index(drop=True), quality.reset_index(drop=True)


def select_eligible_results(
    results: pd.DataFrame,
    policy: str,
    paper_minimum_pairs: dict[str, int],
) -> pd.DataFrame:
    checked = results.copy()
    if policy == "paper":
        checked["eligible_row"] = checked.apply(
            lambda row: row["pair_count"] >= paper_minimum_pairs[row["dataset"]],
            axis=1,
        )
    elif policy == "complete":
        checked["eligible_row"] = checked["pair_count"] == checked["expected_pairs"]
    else:
        raise ValueError("SELECTION_POLICY must be 'paper' or 'complete'")

    checked["eligible_artifact"] = checked.groupby(
        ["distance_metric", "artifact_id"]
    )["eligible_row"].transform("all")
    return checked[checked["eligible_artifact"]].drop(
        columns=["eligible_row", "eligible_artifact"]
    )


def write_square_pair_matrix(
    path: Path,
    values: dict[tuple[str, str], float],
    diagonal_value: float = -1.0,
    missing_value: float = -1.0,
) -> None:
    words = sorted({word for pair in values for word in pair})
    matrix = pd.DataFrame(missing_value, index=words, columns=words, dtype=float)
    np.fill_diagonal(matrix.values, diagonal_value)
    for (word1, word2), value in values.items():
        matrix.loc[word1, word2] = value
        matrix.loc[word2, word1] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(path)
