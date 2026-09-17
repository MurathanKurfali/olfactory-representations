"""Extract all configured BERT representations used by the paper.

Edit ``BERT_MODELS_TO_RUN`` and ``BERT_CONTEXTS_TO_RUN`` in local_config.py
when you want a smaller run. No command-line options are required.
"""

from config import (
    BERT_BATCH_SIZE,
    BERT_CONTEXT_DIRS,
    BERT_CONTEXTS_TO_RUN,
    BERT_MODEL_REVISION,
    BERT_MODELS_TO_RUN,
    BERT_OUTPUT_DIR,
    BERT_RANDOM_SEED,
    BERT_SKIP_EXISTING,
)
from olfactory.bert import extract_bert_grid


if __name__ == "__main__":
    extract_bert_grid(
        models_by_language=BERT_MODELS_TO_RUN,
        contexts_by_language=BERT_CONTEXTS_TO_RUN,
        context_dirs=BERT_CONTEXT_DIRS,
        output_dir=BERT_OUTPUT_DIR,
        batch_size=BERT_BATCH_SIZE,
        seed=BERT_RANDOM_SEED,
        revision=BERT_MODEL_REVISION,
        skip_existing=BERT_SKIP_EXISTING,
    )
