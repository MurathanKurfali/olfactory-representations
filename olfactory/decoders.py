"""Read and generate pairwise similarity judgments from decoder models."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json
import re
from typing import Any

from .prompts import DECODER_PROMPT_ID, format_decoder_prompt
from .vocabulary import canonical_word, paper_decoder_pairs


_SCORE_PATTERN = re.compile(
    r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])"
)
_LABELED_SCORE_PATTERN = re.compile(
    r"(?:similarity\s+)?score(?:\s+is)?\s*[:=]?\s*"
    r"(?P<score>0(?:\.\d+)?|1(?:\.0+)?|\.\d+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DecoderReadReport:
    records: int
    unique_pairs: int
    identical_duplicates: int


def parse_similarity_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("A Boolean value is not a similarity score")
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(value).strip()
        if "[/INST]" in text:
            text = text.split("[/INST]", 1)[1]
        if "<|CHATBOT_TOKEN|>" in text:
            text = text.split("<|CHATBOT_TOKEN|>", 1)[1]
        labeled = list(_LABELED_SCORE_PATTERN.finditer(text))
        matches = list(_SCORE_PATTERN.finditer(text))
        if labeled:
            score = float(labeled[-1].group("score"))
        elif len(matches) == 1:
            score = float(matches[0].group(0))
        elif not matches:
            raise ValueError(f"Could not parse a similarity score from {value!r}")
        else:
            raise ValueError(
                f"Ambiguous response contains multiple possible scores: {value!r}"
            )
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Similarity score outside [0, 1]: {score}")
    return score


def _pair_from_record(record: dict[str, Any], language: str) -> tuple[str, str]:
    if "word1" in record and "word2" in record:
        words = (str(record["word1"]), str(record["word2"]))
    else:
        try:
            words = tuple(str(record["pair"]).split("_", 1))
        except KeyError as exc:
            raise ValueError("Decoder record has no pair information") from exc
    if len(words) != 2:
        raise ValueError(f"Invalid pair record: {record!r}")
    canonical = tuple(canonical_word(word, language) for word in words)
    if canonical[0] == canonical[1]:
        raise ValueError(f"Decoder record contains an identical-word pair: {record!r}")
    return tuple(sorted(canonical))


def read_decoder_scores(
    path: Path,
    language: str = "en",
    paper_rounding: bool = True,
) -> tuple[dict[tuple[str, str], float], DecoderReadReport]:
    scores: dict[tuple[str, str], float] = {}
    records = 0
    duplicate_count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "prompt" in record and "pair" not in record and "word1" not in record:
                continue
            pair = _pair_from_record(record, language)
            score = parse_similarity_score(record.get("score", record.get("raw_response")))
            if paper_rounding:
                score = round(score, 3)
            records += 1
            if pair in scores:
                if scores[pair] != score:
                    raise ValueError(
                        f"Conflicting duplicate for {pair} at {path}:{line_number}"
                    )
                duplicate_count += 1
            scores[pair] = score
    return scores, DecoderReadReport(records, len(scores), duplicate_count)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _prepare_run_metadata(output_path: Path, settings: dict[str, Any]) -> None:
    """Create or check the single metadata file used when resuming a run."""
    output_path = Path(output_path)
    metadata_path = output_path.with_name(f"{output_path.stem}_run_metadata.json")
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid run metadata: {metadata_path}") from exc
        if existing != settings:
            raise ValueError(
                f"Run settings differ from {metadata_path}. "
                "Use a new output filename."
            )
        return
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError(
            f"Cannot resume {output_path} without {metadata_path.name}. "
            "Use a new output filename."
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _completed_local_pairs(path: Path, language: str) -> set[tuple[str, str]]:
    if not Path(path).exists():
        return set()
    scores, _ = read_decoder_scores(path, language=language, paper_rounding=False)
    if language == "en":
        return set(scores)
    # The query functions use local-language pairs, while the reader returns
    # canonical English. Resume from raw records to avoid reverse ambiguity.
    completed: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "word1" in record and "word2" in record:
                completed.add(tuple(sorted((record["word1"], record["word2"]))))
            elif "pair" in record:
                completed.add(tuple(sorted(str(record["pair"]).split("_", 1))))
    return completed


async def query_openai_pairs(
    model_name: str,
    language: str,
    output_path: Path,
    api_key: str,
    max_concurrency: int = 5,
    max_retries: int = 4,
) -> None:
    """Query the Chat Completions endpoint with resumable, validated batches."""
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the ignored keys.py file")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    output_path = Path(output_path)
    _prepare_run_metadata(
        output_path,
        {
            "model": model_name,
            "prompt_id": DECODER_PROMPT_ID,
            "language": language,
            "temperature": 0,
            "max_tokens": 10,
        },
    )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before querying OpenAI") from exc

    completed = _completed_local_pairs(output_path, language)
    pending = [pair for pair in paper_decoder_pairs(language) if pair not in completed]
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def query_one(pair: tuple[str, str]) -> dict[str, Any]:
        prompt = format_decoder_prompt(*pair, language=language)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=10,
                    )
                raw = (response.choices[0].message.content or "").strip()
                score = parse_similarity_score(raw)
                return {
                    "pair": f"{pair[0]}_{pair[1]}",
                    "word1": pair[0],
                    "word2": pair[1],
                    "score": score,
                    "raw_response": raw,
                    "response_id": response.id,
                    "response_model": response.model,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:  # SDK errors vary by installed version.
                last_error = exc
                if attempt + 1 < max_retries:
                    await asyncio.sleep(min(2 ** attempt, 20))
        assert last_error is not None
        raise RuntimeError(f"Failed to query {pair} after {max_retries} attempts") from last_error

    for start in range(0, len(pending), max_concurrency):
        batch = pending[start:start + max_concurrency]
        outcomes = await asyncio.gather(
            *(query_one(pair) for pair in batch),
            return_exceptions=True,
        )
        records = [
            outcome for outcome in outcomes if not isinstance(outcome, BaseException)
        ]
        if records:
            append_jsonl(output_path, records)
        failures = [
            (pair, outcome)
            for pair, outcome in zip(batch, outcomes)
            if isinstance(outcome, BaseException)
        ]
        if failures:
            failed_pairs = ", ".join(
                f"{word1}/{word2}" for (word1, word2), _ in failures
            )
            raise RuntimeError(
                f"Saved {len(records)} successful responses, but failed on: "
                f"{failed_pairs}. Rerun to resume."
            ) from failures[0][1]
        print(f"Saved {min(start + len(batch), len(pending))}/{len(pending)} pending pairs")


def query_huggingface_pairs(
    model_name: str,
    language: str,
    output_path: Path,
    token: str = "",
    batch_size: int = 4,
    use_chat_template: bool = False,
    quantization: str | None = None,
    revision: str | None = None,
) -> None:
    """Query a local Hugging Face causal LM and save resumable JSONL output."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    output_path = Path(output_path)
    _prepare_run_metadata(
        output_path,
        {
            "model": model_name,
            "model_revision": revision,
            "prompt_id": DECODER_PROMPT_ID,
            "language": language,
            "used_chat_template": use_chat_template,
            "quantization": quantization,
            "do_sample": False,
            "max_new_tokens": 10,
        },
    )
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before querying Hugging Face") from exc

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    if token:
        model_kwargs["token"] = token
    if revision:
        model_kwargs["revision"] = revision
    if quantization:
        if quantization not in {"8bit", "4bit"}:
            raise ValueError("quantization must be None, '8bit', or '4bit'")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quantization == "8bit",
            load_in_4bit=quantization == "4bit",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token or None,
        revision=revision,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{model_name} tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    completed = _completed_local_pairs(output_path, language)
    pending = [pair for pair in paper_decoder_pairs(language) if pair not in completed]

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        prompts = [format_decoder_prompt(*pair, language=language) for pair in batch]
        if use_chat_template:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
        generated = generator(
            prompts,
            batch_size=batch_size,
            do_sample=False,
            max_new_tokens=10,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        if len(generated) != len(batch):
            raise RuntimeError(
                f"Model returned {len(generated)} outputs for a batch of {len(batch)}"
            )
        records: list[dict[str, Any]] = []
        for pair, output in zip(batch, generated):
            item = output[0] if isinstance(output, list) else output
            raw = str(item["generated_text"]).strip()
            records.append(
                {
                    "pair": f"{pair[0]}_{pair[1]}",
                    "word1": pair[0],
                    "word2": pair[1],
                    "score": parse_similarity_score(raw),
                    "raw_response": raw,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        append_jsonl(output_path, records)
        print(f"Saved {min(start + len(batch), len(pending))}/{len(pending)} pending pairs")
