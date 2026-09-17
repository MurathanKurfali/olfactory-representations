"""Project-wide settings.

The defaults use only paths inside this repository. Put machine-specific path
changes in ``local_config.py`` so they cannot be committed accidentally.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
HUMAN_DATA_DIR = PROJECT_ROOT / "human_data_public"
LOCAL_DATA_DIR = PROJECT_ROOT / "local_data"
EMBEDDING_DIR = PROJECT_ROOT / "embeddings"
OLFACTORY_CONTEXT_DIR = PROJECT_ROOT / "texts" / "olfactory_contexts"


STATIC_EMBEDDING_DIRS = {
    "word2vec": (EMBEDDING_DIR / "word2vec-filtered",),
    "fasttext": (EMBEDDING_DIR / "fasttext-filtered",),
}
BERT_OUTPUT_DIR = OUTPUT_DIR / "bert_embeddings"
DECODER_OUTPUT_DIR = OUTPUT_DIR / "decoder_results"
BERT_EMBEDDING_DIRS = (LOCAL_DATA_DIR / "bert_embeddings", BERT_OUTPUT_DIR)
DECODER_RESULTS_DIRS = (LOCAL_DATA_DIR / "decoder_results", DECODER_OUTPUT_DIR)

BERT_CONTEXT_DIRS = {
    "blogs": (
        OLFACTORY_CONTEXT_DIR / "blogs_olf",
    ),
    "specialized": (
        OLFACTORY_CONTEXT_DIR / "food_perfume_wine_olf",
    ),
}

ANALYSIS_DISTANCE_METRICS = ("pearson", "cosine")
ANALYSIS_FAMILIES = ("word2vec", "fasttext", "bert", "decoder", "random")
PAPER_ROUNDING = True
RANDOM_DIMENSIONS = (50, 300, 1000)
RANDOM_SEED = 20250924
RANDOM_TABLE_DIMENSION = 300

# The original plotting script retained artifacts with >91 odor/label pairs
# and >1000 Dravnieks pairs. This named policy reproduces that selection while
# keeping pair coverage visible in every output table.
SELECTION_POLICY = "paper"
PAPER_MINIMUM_PAIRS = {
    "odor_based": 92,
    "label_based": 92,
    "dravnieks": 1001,
}

STATIC_FILTER_JOBS = ()

BERT_MODELS_TO_RUN = {
    "en": (
        "bert-base-cased",
        "bert-base-uncased",
        "bert-large-cased",
        "bert-large-uncased",
    ),
    "sv": ("KBLab/bert-base-swedish-cased",),
}
BERT_CONTEXTS_TO_RUN = {
    "en": (
        "prompt0", "prompt1", "prompt2", "prompt3", "prompt4", "prompt5", "prompt6",
        "specialized", "specialized100", "specialized250",
    ),
    "sv": (
        "prompt0", "prompt1", "prompt2", "prompt3", "prompt4", "prompt5", "prompt6",
        "blogs", "blogs100", "blogs250",
    ),
}
BERT_BATCH_SIZE = 16
BERT_RANDOM_SEED = 42
BERT_MODEL_REVISION = None
BERT_SKIP_EXISTING = True

OPENAI_MODEL_NAME = "gpt-4o-2024-08-06"
OPENAI_LANGUAGE = "en"
OPENAI_MAX_CONCURRENCY = 5
OPENAI_MAX_RETRIES = 4

HUGGINGFACE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HUGGINGFACE_MODEL_REVISION = None
HUGGINGFACE_LANGUAGE = "en"
HUGGINGFACE_BATCH_SIZE = 4
HUGGINGFACE_USE_CHAT_TEMPLATE = False
HUGGINGFACE_QUANTIZATION = None  # Set to "8bit" or "4bit" when supported.

DECODER_DISPLAY_NAMES = {
    "Qwen2.5-7B-Instruct_odor_en": "Qwen 2.5 (7B)",
    "Qwen2.5-32B-Instruct_odor_en": "Qwen 2.5 (32B)",
    "Llama-3.1-8B-Instruct_odor_en": "LLaMA 3.1 (8B)",
    "gemma-2-9b-it_odor_en": "Gemma 2 (9B)",
    "Mistral-Small-24B-Instruct-2501_odor_en": "Mistral Small (24B, 2501)",
    "gpt-3.5-turbo-0613_odor_en": "GPT-3.5 Turbo (06-13)",
    "gpt-3.5-turbo-0301_odor_en": "GPT-3.5 Turbo (03-01)",
    "gpt-3.5-turbo-0125_odor_en": "GPT-3.5 Turbo (01-25)",
    "gpt-4o-mini-2024-07-18_odor_en": "GPT-4o Mini (2024-07-18)",
    "gpt-4-0613_odor_en": "GPT-4 (06-13)",
    "gpt-4o-2024-08-06_odor_en": "GPT-4o (2024-08-06)",
}

# Display order used in Supplementary Figure 15 (smallest open model through
# the proprietary GPT models). Unknown future models are appended by name.
DECODER_PLOT_ORDER = (
    "Qwen 2.5 (7B)",
    "LLaMA 3.1 (8B)",
    "Gemma 2 (9B)",
    "Mistral Small (24B, 2501)",
    "Qwen 2.5 (32B)",
    "GPT-3.5 Turbo (06-13)",
    "GPT-3.5 Turbo (03-01)",
    "GPT-3.5 Turbo (01-25)",
    "GPT-4o Mini (2024-07-18)",
    "GPT-4 (06-13)",
    "GPT-4o (2024-08-06)",
)


def _apply_local_overrides() -> None:
    try:
        import local_config
    except ImportError:
        return

    for name in tuple(globals()):
        if name.isupper() and hasattr(local_config, name):
            globals()[name] = getattr(local_config, name)


_apply_local_overrides()
