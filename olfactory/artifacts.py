"""Discover saved model artifacts and attach explicit metadata to each one."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    family: str
    model_name: str
    display_name: str
    source_path: Path | None
    language: str = "en"
    corpus: str = ""
    training: str = ""
    context: str = ""
    layer: int | None = None
    min_count: int | None = None
    epochs: int | None = None
    window: int | None = None
    claimed_dimension: int | None = None
    random_seed: int | None = None
    random_dimension: int | None = None

    def record(self) -> dict[str, object]:
        record = asdict(self)
        record["source_path"] = str(self.source_path) if self.source_path else ""
        return record


_WORD2VEC_PATTERN = re.compile(
    r".+?_(?P<window>\d+)_(?P<dimension>\d+)_(?P<min_count>\d+)_ep(?P<epochs>\d+)\.txt$"
)
_FASTTEXT_PATTERN = re.compile(
    r".+?_(?P<min_count>\d+)_(?P<dimension>\d+)_ep(?P<epochs>\d+)_fasttext\.txt\.vec$"
)
_BERT_PATTERN = re.compile(r"(?P<context>.+)_layer_(?P<layer>\d+)\.vec$")


def _corpus_and_training(folder_name: str) -> tuple[str, str]:
    training = "olfactory" if folder_name.endswith("-olf") else "full"
    corpus = folder_name.removesuffix("-olf")
    corpus = "special" if corpus == "food_perfume_wine" else corpus
    return corpus, training


def discover_static_artifacts(root: Path, family: str) -> list[ModelArtifact]:
    root = Path(root)
    if not root.exists():
        return []
    if family not in {"word2vec", "fasttext"}:
        raise ValueError(f"Unsupported static family: {family}")
    pattern = _WORD2VEC_PATTERN if family == "word2vec" else _FASTTEXT_PATTERN
    display = "Word2Vec" if family == "word2vec" else "FastText"
    artifacts: list[ModelArtifact] = []

    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name.startswith("."):
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        corpus, training = _corpus_and_training(path.parent.name)
        min_count = int(match.group("min_count"))
        epochs = int(match.group("epochs"))
        window = int(match.group("window")) if family == "word2vec" else None
        artifact_id = (
            f"{family}__{corpus}__{training}__mc{min_count}__ep{epochs}"
        )
        artifacts.append(
            ModelArtifact(
                artifact_id=artifact_id,
                family=family,
                model_name=family,
                display_name=display,
                source_path=path.resolve(),
                language="sv" if corpus == "blogs" else "en",
                corpus=corpus,
                training=training,
                min_count=min_count,
                epochs=epochs,
                window=window,
                claimed_dimension=int(match.group("dimension")),
            )
        )
    return artifacts


def discover_bert_artifacts(root: Path) -> list[ModelArtifact]:
    root = Path(root)
    if not root.exists():
        return []
    artifacts: list[ModelArtifact] = []
    for path in sorted(root.rglob("*.vec")):
        match = _BERT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        model_name = path.parent.name
        context = match.group("context")
        layer = int(match.group("layer"))
        artifact_id = f"bert__{model_name}__{context}__layer{layer}"
        artifacts.append(
            ModelArtifact(
                artifact_id=artifact_id,
                family="bert",
                model_name=model_name,
                display_name=model_name,
                source_path=path.resolve(),
                language="sv" if "swedish" in model_name.lower() else "en",
                context=context,
                layer=layer,
            )
        )
    return artifacts


def discover_decoder_artifacts(
    root: Path,
    display_names: dict[str, str],
) -> list[ModelArtifact]:
    root = Path(root)
    if not root.exists():
        return []
    artifacts: list[ModelArtifact] = []
    for path in sorted((*root.glob("*.json"), *root.glob("*.jsonl"))):
        if path.name.endswith("_run_metadata.json"):
            continue
        stem = path.stem
        language = "sv" if stem.endswith("_sv") or "_odor_sv" in stem else "en"
        artifacts.append(
            ModelArtifact(
                artifact_id=f"decoder__{stem}",
                family="decoder",
                model_name=stem,
                display_name=display_names.get(stem, stem),
                source_path=path.resolve(),
                language=language,
                context="paper_similarity_v1",
            )
        )
    return artifacts


def random_artifacts(dimensions: tuple[int, ...], seed: int) -> list[ModelArtifact]:
    return [
        ModelArtifact(
            artifact_id=f"random__dim{dimension}__seed{seed}",
            family="random",
            model_name="random",
            display_name="Random",
            source_path=None,
            random_seed=seed,
            random_dimension=dimension,
        )
        for dimension in dimensions
    ]


def unique_artifacts(artifacts: list[ModelArtifact]) -> list[ModelArtifact]:
    """Keep the first configured copy of an artifact and reject metadata clashes."""
    unique: dict[str, ModelArtifact] = {}
    for artifact in artifacts:
        existing = unique.get(artifact.artifact_id)
        if existing is None:
            unique[artifact.artifact_id] = artifact
            continue
        existing_metadata = existing.record()
        incoming_metadata = artifact.record()
        existing_metadata["source_path"] = ""
        incoming_metadata["source_path"] = ""
        if existing_metadata != incoming_metadata:
            raise ValueError(f"Conflicting metadata for {artifact.artifact_id}")
    return list(unique.values())
