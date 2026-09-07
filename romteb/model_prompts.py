"""Declared encode prefixes / instruction templates per embedder.

Mismatched prefixes do not crash: they silently drop retrieval scores by
5–15 points. Every model that needs a prefix is listed here and applied
in ``run_benchmark._load_model``. Prefer the official MTEB loader for
models whose wrapper does more than a string prefix (Qwen3, jina-v3).

See docs/evaluation_protocol.md (MODEL_PROMPTS).
"""

from __future__ import annotations

from fnmatch import fnmatch

# Keys understood by SentenceTransformerEncoderWrapper / PromptType.
QUERY = "query"
DOCUMENT = "document"

E5_PROMPTS = {QUERY: "query: ", DOCUMENT: "passage: "}

# BGE multilingual gemma2 retrieval instruction (see model card).
# MTEB registry has use_instructions=False for this model, but the model
# card explicitly recommends a query instruction for retrieval; empirically
# missing it costs ~10-15 nDCG@10 on Romanian retrieval.
BGE_GEMMA2_QUERY_PROMPT = (
    "<instruct>Given a query, retrieve relevant passages that answer it.\n<query>"
)

# model id (or glob) -> prompt dict. Empty dict = no prefix.
MODEL_PROMPTS: dict[str, dict[str, str]] = {
    "intfloat/multilingual-e5-small": dict(E5_PROMPTS),
    "intfloat/multilingual-e5-base": dict(E5_PROMPTS),
    "intfloat/multilingual-e5-large": dict(E5_PROMPTS),
    "intfloat/multilingual-e5-large-instruct": dict(E5_PROMPTS),
    "BAAI/bge-m3": {},
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {},
    "sentence-transformers/LaBSE": {},
    "BlackKakapo/stsb-xlm-r-multilingual-ro": {},
    "iliemihai/romanian-sentence-bert-base-uncased-v1": {},
    "ibm-granite/granite-embedding-97m-multilingual-r2": {},
    "ibm-granite/granite-embedding-107m-multilingual": {},
    # Qwen3 uses Instruct: …\\nQuery: via the MTEB instruct wrapper, not a
    # static prefix. Listed so the table is complete; loader must be mteb.
    "Qwen/Qwen3-Embedding-4B": {},
    "Qwen/Qwen3-Embedding-0.6B": {},
    "Qwen/Qwen3-Embedding-8B": {},
    # google/embeddinggemma-300m has per-task-type prompts baked into its
    # sentence_transformers config (task: search result | query: ...). MTEB
    # picks them up when loaded via the official wrapper; listed so the
    # table is complete and loader is forced to `mteb`.
    "google/embeddinggemma-300m": {},
    # nvidia/llama-embed-nemotron-8b requires the LlamaEmbedNemotron wrapper
    # (Instruct: {task}\nQuery: template). Loader must be `mteb`.
    "nvidia/llama-embed-nemotron-8b": {},
    # BAAI/bge-multilingual-gemma2: MTEB registry has use_instructions=False,
    # so the wrapper skips instructions unless we pass a query prompt.
    "BAAI/bge-multilingual-gemma2": {QUERY: BGE_GEMMA2_QUERY_PROMPT},
    "romteb/bm25s-ro": {},
}

# Globs applied after exact matches.
MODEL_PROMPT_GLOBS: list[tuple[str, dict[str, str]]] = [
    ("intfloat/multilingual-e5-*", dict(E5_PROMPTS)),
    ("intfloat/e5-*", dict(E5_PROMPTS)),
]

# Official MTEB wrappers apply prompts / LoRA adapters / instruct templates.
PREFER_MTEB_LOADER: frozenset[str] = frozenset(
    {
        "intfloat/multilingual-e5-small",
        "intfloat/multilingual-e5-base",
        "intfloat/multilingual-e5-large",
        "intfloat/multilingual-e5-large-instruct",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-8B",
        "jinaai/jina-embeddings-v3",
        "jinaai/jina-embeddings-v4",
        # embeddinggemma: MTEB wrapper picks up native task-type prompts.
        "google/embeddinggemma-300m",
        # nemotron: LlamaEmbedNemotron wrapper applies Instruct: template.
        "nvidia/llama-embed-nemotron-8b",
    }
)

PROMPT_NOTES: dict[str, str] = {
    "intfloat/multilingual-e5-large": "query: / passage: (required)",
    "BAAI/bge-m3": "no prefix (MTEB default)",
    "BAAI/bge-multilingual-gemma2": "<instruct>...\\n<query> (query only; MTEB registry has use_instructions=False)",
    "Qwen/Qwen3-Embedding-4B": "Instruct: {task}\\nQuery: via mteb instruct wrapper",
    "google/embeddinggemma-300m": "task: search result | query: (native ST prompts, via mteb wrapper)",
    "nvidia/llama-embed-nemotron-8b": "Instruct: {task}\\nQuery: via LlamaEmbedNemotron wrapper (loader=mteb)",
    "ibm-granite/granite-embedding-97m-multilingual-r2": "no prefix",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "no prefix",
    "sentence-transformers/LaBSE": "no prefix",
    "BlackKakapo/stsb-xlm-r-multilingual-ro": "no prefix",
    "iliemihai/romanian-sentence-bert-base-uncased-v1": "no prefix",
    "romteb/bm25s-ro": "lexical; no embedding prefixes",
}


def prompts_for(model_name: str) -> dict[str, str]:
    """Return the declared prompt map for ``model_name`` (possibly empty)."""
    if model_name in MODEL_PROMPTS:
        return dict(MODEL_PROMPTS[model_name])
    for pattern, prompts in MODEL_PROMPT_GLOBS:
        if fnmatch(model_name, pattern):
            return dict(prompts)
    return {}


def prefer_mteb_loader(model_name: str) -> bool:
    return model_name in PREFER_MTEB_LOADER
