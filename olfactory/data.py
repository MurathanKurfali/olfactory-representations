"""Load and validate the three human benchmark matrices."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .vocabulary import DRAVNIEKS_WORDS_EN, ODOR_BASED_WORDS_EN, canonical_word


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    filename: str
    language: str
    values_are: str  # "similarity" or "distance"


DATASET_SPECS = (
    DatasetSpec("odor_based", "Odor-based", "RatingsSimilarity_matrix.csv", "sv", "similarity"),
    DatasetSpec("label_based", "Label-based", "QualtricsSimilarity_matrix.csv", "sv", "similarity"),
    DatasetSpec("dravnieks", "Dravnieks", "DravniekSimilarity_matrix.csv", "en", "distance"),
)


@dataclass
class HumanDataset:
    spec: DatasetSpec
    raw_matrix: pd.DataFrame
    distance_matrix: pd.DataFrame
    pair_distances: dict[tuple[str, str], float]

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(self.distance_matrix.index)

    @property
    def expected_pair_count(self) -> int:
        return len(self.pair_distances)

    def similarity_matrix(self) -> pd.DataFrame:
        if self.spec.values_are == "similarity":
            return self.raw_matrix.copy()
        return 1.0 - self.raw_matrix


def _read_square_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark matrix: {path}."
        )
    frame = pd.read_csv(path, sep=";", index_col=0)
    frame.index = [str(value).strip().lower() for value in frame.index]
    frame.columns = [str(value).strip().lower() for value in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="raise")

    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Expected a square matrix in {path}; found {frame.shape}")
    if set(frame.index) != set(frame.columns):
        raise ValueError(f"Row and column labels differ in {path}")
    frame = frame.loc[frame.index, frame.index]
    return frame


def _canonicalize_labels(frame: pd.DataFrame, language: str) -> pd.DataFrame:
    labels = [canonical_word(label, language) for label in frame.index]
    if len(labels) != len(set(labels)):
        raise ValueError("Translation produced duplicate canonical labels")
    result = frame.copy()
    result.index = labels
    result.columns = labels
    return result


def _validate_matrix(frame: pd.DataFrame, spec: DatasetSpec, path: Path) -> None:
    values = frame.to_numpy(dtype=float)
    valid = values[values != -1]
    if valid.size == 0:
        raise ValueError(f"No usable values in {path}")
    if np.nanmin(valid) < 0 or np.nanmax(valid) > 1:
        raise ValueError(f"Values outside [0, 1] in {path}")

    left = values.copy()
    right = values.T.copy()
    left[left == -1] = np.nan
    right[right == -1] = np.nan
    if not np.allclose(left, right, equal_nan=True, atol=1e-10):
        raise ValueError(f"Matrix is not symmetric: {path}")

    expected_diagonal = 1.0 if spec.values_are == "similarity" else 0.0
    diagonal = np.diag(values)
    if not np.allclose(diagonal, expected_diagonal, atol=1e-10):
        raise ValueError(
            f"Unexpected diagonal in {path}; expected {expected_diagonal:g}"
        )


def load_human_dataset(
    human_data_dir: Path,
    spec: DatasetSpec,
    paper_rounding: bool = True,
) -> HumanDataset:
    path = Path(human_data_dir) / spec.filename
    raw = _canonicalize_labels(_read_square_matrix(path), spec.language)
    _validate_matrix(raw, spec, path)

    prepared = raw.round(3) if paper_rounding else raw.copy()
    distance = 1.0 - prepared if spec.values_are == "similarity" else prepared

    pairs: dict[tuple[str, str], float] = {}
    for row_index, word1 in enumerate(distance.index):
        for column_index in range(row_index + 1, distance.shape[1]):
            word2 = distance.columns[column_index]
            value = float(distance.iloc[row_index, column_index])
            if value == -1 or np.isnan(value):
                continue
            pairs[tuple(sorted((word1, word2)))] = value

    return HumanDataset(spec, raw, distance, pairs)


def load_human_datasets(
    human_data_dir: Path,
    paper_rounding: bool = True,
) -> dict[str, HumanDataset]:
    available_specs = tuple(
        spec for spec in DATASET_SPECS
        if (Path(human_data_dir) / spec.filename).exists()
    )
    if not available_specs:
        expected = ", ".join(spec.filename for spec in DATASET_SPECS)
        raise FileNotFoundError(
            f"No human benchmark matrices found in {human_data_dir}. "
            f"Expected at least one of: {expected}"
        )
    datasets = {
        spec.key: load_human_dataset(human_data_dir, spec, paper_rounding)
        for spec in available_specs
    }
    expected = {"odor_based": 120, "label_based": 120, "dravnieks": 3321}
    expected_words = {
        "odor_based": set(ODOR_BASED_WORDS_EN),
        "label_based": set(ODOR_BASED_WORDS_EN),
        "dravnieks": set(DRAVNIEKS_WORDS_EN),
    }
    for key, dataset in datasets.items():
        pair_count = expected[key]
        observed = dataset.expected_pair_count
        if observed != pair_count:
            raise ValueError(f"{key} has {observed} usable pairs; expected {pair_count}")
        observed_words = set(dataset.words)
        if observed_words != expected_words[key]:
            missing = sorted(expected_words[key] - observed_words)
            extra = sorted(observed_words - expected_words[key])
            raise ValueError(
                f"{key} vocabulary differs from the paper: "
                f"missing={missing}, extra={extra}"
            )
    return datasets
