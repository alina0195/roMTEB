"""Romanian BM25 lexical baseline (bm25s + PyStemmer + diacritic fold).

Used as a row in every retrieval / reranking table. Same ``top_ranked``
restriction as dense rerankers, so MCQ pools stay closed-set.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mteb.models.models_protocols import SearchProtocol

logger = logging.getLogger(__name__)

_DIACRITICS = str.maketrans(
    {
        "ă": "a",
        "â": "a",
        "î": "i",
        "ș": "s",
        "ş": "s",
        "ț": "t",
        "ţ": "t",
        "Ă": "a",
        "Â": "a",
        "Î": "i",
        "Ș": "s",
        "Ş": "s",
        "Ț": "t",
        "Ţ": "t",
    }
)

# Fallback if bm25s has no built-in Romanian stopword list.
_RO_STOPWORDS = [
    "a",
    "acea",
    "aceasta",
    "aceea",
    "acei",
    "aceia",
    "acel",
    "acela",
    "acele",
    "acelea",
    "acest",
    "acesta",
    "aceste",
    "acestea",
    "aceşti",
    "aceştia",
    "acolo",
    "acum",
    "ai",
    "aia",
    "aibă",
    "am",
    "ar",
    "are",
    "as",
    "asă",
    "așa",
    "asa",
    "au",
    "avea",
    "avem",
    "aveți",
    "azi",
    "aş",
    "aşadar",
    "aţi",
    "ca",
    "că",
    "căci",
    "când",
    "care",
    "căreia",
    "cărora",
    "cărui",
    "căror",
    "ce",
    "cel",
    "ceva",
    "cu",
    "cum",
    "cumva",
    "da",
    "dacă",
    "dar",
    "de",
    "deci",
    "deja",
    "din",
    "doar",
    "după",
    "ea",
    "ei",
    "el",
    "ele",
    "era",
    "este",
    "eu",
    "face",
    "fără",
    "fi",
    "fie",
    "fiecare",
    "fost",
    "i",
    "ia",
    "iar",
    "ii",
    "îi",
    "îl",
    "îmi",
    "în",
    "înainte",
    "încât",
    "încă",
    "însă",
    "între",
    "într",
    "la",
    "le",
    "li",
    "lor",
    "lui",
    "m",
    "mă",
    "mai",
    "mea",
    "mei",
    "mele",
    "meu",
    "mi",
    "mie",
    "mine",
    "mult",
    "ne",
    "ni",
    "noi",
    "nostru",
    "noastră",
    "nu",
    "o",
    "or",
    "ori",
    "pe",
    "pentru",
    "peste",
    "poate",
    "prea",
    "prin",
    "sa",
    "să",
    "sai",
    "sale",
    "sau",
    "se",
    "și",
    "si",
    "sunt",
    "suntem",
    "sunteți",
    "ta",
    "tăi",
    "tale",
    "te",
    "ți",
    "ți",
    "toată",
    "toate",
    "tot",
    "totuși",
    "tu",
    "un",
    "una",
    "unde",
    "unei",
    "unele",
    "unei",
    "unii",
    "unor",
    "unui",
    "va",
    "vă",
    "vi",
    "voastre",
    "vostru",
    "vouă",
    "vreo",
    "vreun",
]


def fold_romanian(text: str) -> str:
    """Lowercase and fold Romanian diacritics for lexical matching."""
    folded = (text or "").lower().translate(_DIACRITICS)
    return unicodedata.normalize("NFC", folded)


def _romanian_stopwords():
    try:
        import bm25s

        stopwords = getattr(bm25s, "stopwords", None)
        if stopwords is None:
            return _RO_STOPWORDS
        for key in ("romanian", "ro", "ron"):
            if hasattr(stopwords, key):
                return getattr(stopwords, key)
            if isinstance(stopwords, dict) and key in stopwords:
                return stopwords[key]
        get = getattr(stopwords, "get", None)
        if callable(get):
            hit = get("romanian") or get("ro")
            if hit:
                return hit
    except Exception as exc:
        logger.info("bm25s Romanian stopwords unavailable (%s); using fallback list", exc)
    return _RO_STOPWORDS


def load_bm25_ro(**kwargs) -> SearchProtocol:
    """BM25s with Romanian stemmer, stopwords, and diacritic folding."""
    from mteb.models.model_implementations.bm25 import bm25_loader

    stopwords = kwargs.pop("stopwords", None) or _romanian_stopwords()
    stemmer_language = kwargs.pop("stemmer_language", "romanian")
    inner = bm25_loader(
        "romteb/bm25s-ro",
        stopwords=stopwords,
        stemmer_language=stemmer_language,
        **kwargs,
    )
    orig_encode = inner._encode

    def _encode(texts: list[str]):
        folded = [fold_romanian(t) for t in texts]
        return orig_encode(folded)

    inner._encode = _encode  # type: ignore[method-assign]
    # MTEB 2.12 RetrievalEvaluator dispatches via runtime-checked SearchProtocol
    # which requires a `mteb_model_meta` property. bm25_loader from upstream
    # returns a BM25Search instance without one, so we attach it here.
    inner.mteb_model_meta = bm25_model_meta()  # type: ignore[attr-defined]
    logger.info(
        "BM25-RO: stemmer=%s stopwords=%s (diacritics folded)",
        stemmer_language,
        "list" if isinstance(stopwords, list) else type(stopwords).__name__,
    )
    return inner


def bm25_model_meta() -> Any:
    from mteb.models.model_meta import ModelMeta

    return ModelMeta(
        loader=load_bm25_ro,
        extra_requirements_groups=["bm25s"],
        name="romteb/bm25s-ro",
        model_type=["dense"],
        languages=["ron-Latn"],
        open_weights=True,
        revision="0_1_10",
        release_date="2024-07-10",
        n_parameters=None,
        n_embedding_parameters=None,
        memory_usage_mb=None,
        embed_dim=None,
        license=None,
        max_tokens=None,
        reference="https://github.com/xhluca/bm25s",
        similarity_fn_name=None,
        framework=[],
        use_instructions=False,
        public_training_code="https://github.com/xhluca/bm25s",
        public_training_data=None,
        training_datasets=None,
    )


BM25_MODEL_NAME = "romteb/bm25s-ro"
LEXICAL_BASELINE_NAMES = frozenset({BM25_MODEL_NAME, "bm25s", "BM25", "mteb/baseline-bm25s"})
