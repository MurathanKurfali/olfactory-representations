"""BERT context preparation and layer-wise target embedding extraction."""

from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import pickle
import random
import re
import tempfile
import warnings
from typing import Any

import numpy as np

from .embeddings import write_text_embeddings
from .prompts import BERT_PROMPTS
from .vocabulary import BERT_WORDS_EN, BERT_WORDS_SV


def _load_pickled_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as handle:
        payload = pickle.load(handle)
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict("records")
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of records in {path}")
    return payload


def _stable_rng(seed: int, context_name: str, word: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{context_name}:{word}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _context_source(context_name: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(.+?)(100|250)?", context_name)
    assert match is not None
    base = match.group(1)
    limit = int(match.group(2)) if match.group(2) else None
    return base, limit


def build_bert_contexts(
    context_name: str,
    language: str,
    context_dirs: dict[str, tuple[Path, ...]],
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    """Build ``local word -> [(surface form, sentence), ...]`` contexts."""
    words = BERT_WORDS_EN if language == "en" else BERT_WORDS_SV
    if context_name.startswith("prompt"):
        try:
            template = BERT_PROMPTS[language][context_name]
        except KeyError as exc:
            raise ValueError(f"Unknown BERT prompt context: {language}/{context_name}") from exc
        return {word: [(word, template.format(word=word))] for word in words}

    source_name, limit = _context_source(context_name)
    try:
        directories = tuple(Path(path) for path in context_dirs[source_name])
    except KeyError as exc:
        raise ValueError(f"No context directories configured for {source_name!r}") from exc
    missing_directories = [str(path) for path in directories if not path.exists()]
    if missing_directories:
        raise FileNotFoundError(
            "Missing BERT context directories: " + ", ".join(missing_directories)
        )

    alternatives = "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))
    word_pattern = re.compile(rf"(?<!\w)({alternatives})(?!\w)", flags=re.IGNORECASE)
    target_contexts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    canonical_local = {word.casefold(): word for word in words}

    allowed_suffixes = {".gz", ".gzip", ".pickle", ".pkl"}
    for directory in directories:
        paths = sorted(
            candidate
            for candidate in directory.iterdir()
            if candidate.is_file()
            and not candidate.name.startswith(".")
            and candidate.suffix.lower() in allowed_suffixes
        )
        for path in paths:
            for record in _load_pickled_records(path):
                sentence = record.get("sentence")
                if isinstance(sentence, list):
                    sentence = " ".join(str(token) for token in sentence)
                if not isinstance(sentence, str):
                    continue
                seen_in_sentence: set[str] = set()
                for match in word_pattern.finditer(sentence):
                    surface = match.group(0)
                    local_word = canonical_local[surface.casefold()]
                    if local_word in seen_in_sentence:
                        continue
                    target_contexts[local_word].append((surface, sentence))
                    seen_in_sentence.add(local_word)

    missing_words = sorted(set(words) - set(target_contexts))
    if missing_words:
        raise ValueError(
            f"No {context_name} contexts found for {len(missing_words)} words: "
            + ", ".join(missing_words)
        )

    if limit is not None:
        for word, contexts in target_contexts.items():
            contexts = list(contexts)
            _stable_rng(seed, context_name, word).shuffle(contexts)
            target_contexts[word] = contexts[:limit]
    return dict(target_contexts)


def _split_sentence_pair(sentence: str) -> tuple[str, str | None]:
    if " [SEP] " not in sentence:
        return sentence, None
    first, second = sentence.split(" [SEP] ", 1)
    return first, second


def _ensure_run_manifest(
    model_folder: Path,
    *,
    model_name: str,
    language: str,
    requested_revision: str | None,
    resolved_revision: str | None,
    layer_count: int,
    context_sampling_seed: int,
    transformers_version: str,
) -> None:
    """Create or validate metadata before reusing layer files."""
    model_folder = Path(model_folder)
    manifest_path = model_folder / "run_metadata.json"
    identity = {
        "format_version": 1,
        "model_name": model_name,
        "language": language,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "layer_count": layer_count,
        "context_sampling_seed": context_sampling_seed,
        "transformers_version": transformers_version,
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid BERT run manifest: {manifest_path}") from exc
        for field, expected in identity.items():
            if field == "transformers_version":
                continue
            if existing.get(field) != expected:
                raise ValueError(
                    f"Cannot safely resume {model_folder}: manifest has "
                    f"{field}={existing.get(field)!r}, expected {expected!r}. "
                    "Use a new output directory."
                )
        recorded_version = existing.get("transformers_version")
        if recorded_version != transformers_version:
            warnings.warn(
                "Resuming BERT extraction with Transformers "
                f"{transformers_version}; the run started with {recorded_version}.",
                stacklevel=2,
            )
        return

    if model_folder.exists() and any(model_folder.glob("*.vec")):
        raise ValueError(
            f"Cannot safely resume {model_folder}: layer files exist without "
            "run_metadata.json. Move them to the read-only artifact directory "
            "or use a new output directory."
        )

    model_folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(identity)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=model_folder,
            prefix=".run_metadata.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(manifest_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _character_spans(text: str, surface: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(surface), text, flags=re.IGNORECASE)
    ]


def extract_target_embeddings(
    contexts: dict[str, list[tuple[str, str]]],
    tokenizer: Any,
    model: Any,
    device: Any,
    batch_size: int = 16,
) -> dict[int, dict[str, np.ndarray]]:
    """Average subword embeddings over occurrences and contexts for each layer."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before extracting BERT") from exc

    layer_vectors: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for target_word in sorted(contexts):
        examples = contexts[target_word]
        sums: dict[int, np.ndarray] = {}
        counts: dict[int, int] = defaultdict(int)

        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            split = [_split_sentence_pair(sentence) for _, sentence in batch]
            texts = [item[0] for item in split]
            text_pairs = [item[1] for item in split]
            tokenize_kwargs: dict[str, Any] = {
                "padding": True,
                "truncation": True,
                "max_length": 512,
                "return_offsets_mapping": True,
                "return_tensors": "pt",
            }
            if any(pair is not None for pair in text_pairs):
                encoded = tokenizer(texts, text_pair=[pair or "" for pair in text_pairs], **tokenize_kwargs)
            else:
                encoded = tokenizer(texts, **tokenize_kwargs)

            offsets = encoded.pop("offset_mapping").cpu().numpy()
            sequence_ids = [encoding.sequence_ids for encoding in encoded.encodings]
            model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.no_grad():
                output = model(**model_inputs, output_hidden_states=True, return_dict=True)

            for example_index, (surface, _) in enumerate(batch):
                spans = _character_spans(texts[example_index], surface)
                for span_start, span_end in spans:
                    token_indices = [
                        token_index
                        for token_index, ((offset_start, offset_end), sequence_id) in enumerate(
                            zip(offsets[example_index], sequence_ids[example_index])
                        )
                        if sequence_id == 0
                        and offset_end > offset_start
                        and offset_start < span_end
                        and offset_end > span_start
                    ]
                    if not token_indices:
                        continue
                    for layer, hidden_state in enumerate(output.hidden_states[1:], start=1):
                        vector = (
                            hidden_state[example_index, token_indices]
                            .mean(dim=0)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(float)
                        )
                        sums[layer] = sums.get(layer, np.zeros_like(vector)) + vector
                        counts[layer] += 1

        if not counts:
            raise ValueError(f"No token occurrence extracted for {target_word!r}")
        for layer in sorted(sums):
            layer_vectors[layer][target_word] = sums[layer] / counts[layer]
    return dict(layer_vectors)


def extract_bert_grid(
    models_by_language: dict[str, tuple[str, ...]],
    contexts_by_language: dict[str, tuple[str, ...]],
    context_dirs: dict[str, tuple[Path, ...]],
    output_dir: Path,
    batch_size: int = 16,
    seed: int = 42,
    revision: str | None = None,
    skip_existing: bool = True,
) -> None:
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before extracting BERT") from exc

    output_dir = Path(output_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    for language, model_names in models_by_language.items():
        context_names = contexts_by_language.get(language, ())
        if not context_names:
            continue
        for model_name in model_names:
            model_folder = output_dir / model_name.split("/")[-1]
            model_config = AutoConfig.from_pretrained(model_name, revision=revision)
            layer_count = int(model_config.num_hidden_layers)
            resolved_revision = getattr(model_config, "_commit_hash", None)
            _ensure_run_manifest(
                model_folder,
                model_name=model_name,
                language=language,
                requested_revision=revision,
                resolved_revision=resolved_revision,
                layer_count=layer_count,
                context_sampling_seed=seed,
                transformers_version=transformers.__version__,
            )
            pending_contexts = []
            for context_name in context_names:
                expected_paths = [
                    model_folder / f"{context_name}_layer_{layer}.vec"
                    for layer in range(1, layer_count + 1)
                ]
                if skip_existing and all(
                    path.is_file() and path.stat().st_size > 0 for path in expected_paths
                ):
                    print(f"Skipping complete output: {model_name} / {context_name}")
                else:
                    pending_contexts.append(context_name)
            if not pending_contexts:
                continue

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                use_fast=True,
            )
            if not tokenizer.is_fast:
                raise ValueError(f"A fast tokenizer is required for {model_name}")
            model = AutoModel.from_pretrained(
                model_name,
                revision=revision,
                config=model_config,
            )
            model.to(device)
            model.eval()
            for context_name in pending_contexts:
                print(f"Extracting {model_name} / {language} / {context_name} on {device}")
                contexts = build_bert_contexts(
                    context_name, language, context_dirs, seed
                )
                extracted = extract_target_embeddings(
                    contexts, tokenizer, model, device, batch_size=batch_size
                )
                expected_layers = set(range(1, layer_count + 1))
                if set(extracted) != expected_layers:
                    raise ValueError(
                        f"{model_name}/{context_name} returned layers "
                        f"{sorted(extracted)}; expected 1..{layer_count}"
                    )
                for layer, vectors in extracted.items():
                    write_text_embeddings(
                        model_folder / f"{context_name}_layer_{layer}.vec",
                        vectors,
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
