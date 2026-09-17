"""Query one configured OpenAI model for the paper's 3,336 odor pairs."""

import asyncio
import os

from config import (
    DECODER_OUTPUT_DIR,
    OPENAI_LANGUAGE,
    OPENAI_MAX_CONCURRENCY,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL_NAME,
)
from olfactory.decoders import query_openai_pairs

try:
    from keys import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = ""


if __name__ == "__main__":
    key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    output = DECODER_OUTPUT_DIR / f"{OPENAI_MODEL_NAME}_odor_{OPENAI_LANGUAGE}.jsonl"
    asyncio.run(
        query_openai_pairs(
            model_name=OPENAI_MODEL_NAME,
            language=OPENAI_LANGUAGE,
            output_path=output,
            api_key=key,
            max_concurrency=OPENAI_MAX_CONCURRENCY,
            max_retries=OPENAI_MAX_RETRIES,
        )
    )
