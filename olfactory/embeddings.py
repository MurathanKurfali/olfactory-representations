"""Memory-efficient I/O for text-format static and BERT embeddings."""

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
import shutil
import tempfile

import numpy as np


@dataclass(frozen=True)
class EmbeddingMetadata:
    declared_rows: int | None
    declared_dimension: int | None
    loaded_rows: int
    actual_dimension: int


def _header(parts: list[str]) -> tuple[int, int] | None:
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def read_text_embeddings(
    path: Path,
    wanted_words: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], EmbeddingMetadata]:
    """Read word2vec text format while parsing only requested vectors.

    The source file is streamed line by line. This avoids loading multi-gigabyte
    pretrained embedding files when only the paper's odor vocabulary is needed.
    """
    path = Path(path)
    wanted = {word.lower() for word in wanted_words} if wanted_words else None
    vectors: dict[str, np.ndarray] = {}
    declared_rows: int | None = None
    declared_dimension: int | None = None
    actual_dimension: int | None = None

    def consume(line: str, line_number: int) -> None:
        nonlocal actual_dimension
        parts = line.rstrip().split()
        if len(parts) < 2:
            return
        word = parts[0].lower()
        if wanted is not None and word not in wanted:
            return
        try:
            vector = np.asarray(parts[1:], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Invalid vector at {path}:{line_number}") from exc
        if actual_dimension is None:
            actual_dimension = int(vector.size)
        elif vector.size != actual_dimension:
            raise ValueError(
                f"Inconsistent vector dimension at {path}:{line_number}: "
                f"{vector.size} instead of {actual_dimension}"
            )
        vectors[word] = vector

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
        parsed_header = _header(first_line.split())
        if parsed_header is None:
            consume(first_line, 1)
        else:
            declared_rows, declared_dimension = parsed_header
        for line_number, line in enumerate(handle, start=2):
            consume(line, line_number)

    if not vectors:
        raise ValueError(f"No requested vectors were found in {path}")
    assert actual_dimension is not None
    return vectors, EmbeddingMetadata(
        declared_rows=declared_rows,
        declared_dimension=declared_dimension,
        loaded_rows=len(vectors),
        actual_dimension=actual_dimension,
    )


def write_text_embeddings(path: Path, vectors: dict[str, np.ndarray]) -> None:
    if not vectors:
        raise ValueError("Cannot write an empty embedding file")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(vectors)
    dimension = int(np.asarray(vectors[ordered[0]]).size)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(f"{len(ordered)} {dimension}\n")
            for word in ordered:
                vector = np.asarray(vectors[word], dtype=float)
                if vector.size != dimension:
                    raise ValueError(f"Vector dimension differs for {word!r}")
                values = " ".join(f"{value:.10g}" for value in vector)
                handle.write(f"{word} {values}\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def filter_text_embeddings(
    source: Path,
    destination: Path,
    wanted_words: set[str],
) -> EmbeddingMetadata:
    """Create a compact text embedding file using a single streaming pass."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wanted = {word.lower() for word in wanted_words}
    found: set[str] = set()
    dimension: int | None = None
    declared_rows: int | None = None
    declared_dimension: int | None = None
    body_path: Path | None = None
    output_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False
        ) as temporary:
            body_path = Path(temporary.name)
            with source.open("r", encoding="utf-8", errors="ignore") as handle:
                first_line = handle.readline()
                parsed_header = _header(first_line.rstrip().split())
                if parsed_header is None:
                    lines = chain((first_line,), handle)
                    first_line_number = 1
                else:
                    declared_rows, declared_dimension = parsed_header
                    lines = handle
                    first_line_number = 2
                for line_number, line in enumerate(lines, start=first_line_number):
                    parts = line.rstrip().split()
                    if len(parts) < 2 or parts[0].lower() not in wanted:
                        continue
                    try:
                        current_dimension = len([float(value) for value in parts[1:]])
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid vector at {source}:{line_number}"
                        ) from exc
                    if dimension is None:
                        dimension = current_dimension
                    elif current_dimension != dimension:
                        raise ValueError(
                            f"Inconsistent vector dimension at {source}:{line_number}"
                        )
                    word = parts[0].lower()
                    if word not in found:
                        temporary.write(line if line.endswith("\n") else line + "\n")
                        found.add(word)

        if dimension is None:
            raise ValueError(f"None of the requested words were found in {source}")

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as output:
            output_path = Path(output.name)
            output.write(f"{len(found)} {dimension}\n")
            assert body_path is not None
            with body_path.open("r", encoding="utf-8") as temporary:
                shutil.copyfileobj(temporary, output)
        output_path.replace(destination)
        output_path = None
    finally:
        if body_path is not None:
            body_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)

    return EmbeddingMetadata(
        declared_rows,
        declared_dimension,
        len(found),
        dimension,
    )
