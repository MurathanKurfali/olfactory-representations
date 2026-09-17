"""Query one configured open-source Hugging Face model for all odor pairs."""

import os

from config import (
    DECODER_OUTPUT_DIR,
    HUGGINGFACE_BATCH_SIZE,
    HUGGINGFACE_LANGUAGE,
    HUGGINGFACE_MODEL_NAME,
    HUGGINGFACE_MODEL_REVISION,
    HUGGINGFACE_QUANTIZATION,
    HUGGINGFACE_USE_CHAT_TEMPLATE,
)
from olfactory.decoders import query_huggingface_pairs

try:
    from keys import HUGGINGFACE_TOKEN
except ImportError:
    HUGGINGFACE_TOKEN = ""


if __name__ == "__main__":
    token = HUGGINGFACE_TOKEN or os.environ.get("HUGGINGFACE_TOKEN", "")
    basename = HUGGINGFACE_MODEL_NAME.split("/")[-1]
    output = DECODER_OUTPUT_DIR / f"{basename}_odor_{HUGGINGFACE_LANGUAGE}.jsonl"
    query_huggingface_pairs(
        model_name=HUGGINGFACE_MODEL_NAME,
        revision=HUGGINGFACE_MODEL_REVISION,
        language=HUGGINGFACE_LANGUAGE,
        output_path=output,
        token=token,
        batch_size=HUGGINGFACE_BATCH_SIZE,
        use_chat_template=HUGGINGFACE_USE_CHAT_TEMPLATE,
        quantization=HUGGINGFACE_QUANTIZATION,
    )
