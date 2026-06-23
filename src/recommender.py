"""
src/recommender.py
------------------
Core recommendation engine for the Academic Paper Recommendation System.

Ranking strategy (validated)
----------------------------
    final_score = cosine_similarity * (1 + ALPHA * same_category_flag)

This is exactly the configuration validated in the alpha-sweep experiment
(src/experiments_with_alpha.py): a *multiplicative* category boost with
ALPHA = 0.03. That run gave the best validated NDCG (~+37% over the
un-boosted cosine baseline) while still letting a strongly-similar
cross-category paper surface. ALPHA = 0.03 is the principled choice — it
reaches ~98% of the maximum gain before the boost saturates (~0.05+), so it
is not sitting at the saturation ceiling.

Two query modes (spec Section 9)
--------------------------------
    "title"    : fuzzy-match the query against the dataset titles, then reuse
                 the matched paper's PRECOMPUTED embedding. The matched paper's
                 category drives the category boost.
    "keywords" : embed the free-text query on the fly with the selected model.
                 For SPECTER2 this uses the ADHOC_QUERY adapter (the tool meant
                 for short text queries), compared against the proximity corpus
                 embeddings — SPECTER2's documented query->document setup.
                 There is no query category, so NO boost is applied — ranking
                 is pure cosine similarity. This is honest: the boost is
                 undefined without a query category.

Category, KeyBERT tags, and citation_count are RETURNED as metadata so the
frontend can display and filter on them (spec Section 7), but they are NOT
part of the ranking score. Only the validated cosine + category-boost formula
ranks papers, so what the app does matches what the experiments measured.

Usage
-----
    from src.recommender import PaperRecommender

    rec = PaperRecommender(model="specter2")
    results = rec.recommend("attention is all you need", mode="title", top_k=10)

Install requirement:
    pip install rapidfuzz
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EMBEDDINGS_DIR = Path("models/embeddings")
DATA_FILE      = Path("data/processed/papers_keybert_final.csv")

MODEL_FILES = {
    "specter2": {
        "embeddings": EMBEDDINGS_DIR / "specter2_embeddings.npy",
        "arxiv_ids":  EMBEDDINGS_DIR / "specter2_arxiv_ids.npy",
    },
    "scibert": {
        "embeddings": EMBEDDINGS_DIR / "scibert_embeddings.npy",
        "arxiv_ids":  EMBEDDINGS_DIR / "scibert_arxiv_ids.npy",
    },
}


# ---------------------------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------------------------

# Multiplicative category-boost factor.
# Validated in experiments_with_alpha.py — see module docstring.
# Set to 0.0 to disable the boost and rank on pure cosine similarity.
CATEGORY_BOOST_ALPHA = 0.03

# Fuzzy title-match threshold (0-100). Below this we treat it as "no match"
# (the app then says it doesn't have the paper and falls back to a free-text
# search). Empirically with token_sort_ratio: a genuinely different title that
# shares ~3/5 words (e.g. "Attention Is All You Need" vs "Attention Is Where You
# Attack") scores ~78, while a real title typed with minor variation scores ~86+.
# 80 sits in that gap. The margin is small — title-only matching can't perfectly
# separate "same paper, typed loosely" from "different paper, similar words".
FUZZY_THRESHOLD = 80

# Scorer for title matching. token_sort_ratio is order-insensitive but
# length-sensitive. We deliberately AVOID WRatio/partial_ratio here: those reward
# substring matches, so a short query that is a prefix of a much longer (and
# different) title scored ~90 and produced a false match. token_sort_ratio scores
# that same case ~27 (correctly rejected) while still scoring true titles 85-100.
FUZZY_SCORER = fuzz.token_sort_ratio

# Default number of recommendations to return.
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Lazy-loaded encoders (only needed for keyword mode)
# ---------------------------------------------------------------------------

def _load_specter2_model():
    """
    Loads SPECTER2 (base model + ADHOC_QUERY adapter) for embedding short
    free-text keyword queries.

    Why the adhoc_query adapter (not proximity)?
        SPECTER2's documented setup for "short text query -> retrieve papers"
        is to encode the QUERY with the adhoc_query adapter and the CANDIDATE
        papers with the proximity adapter. Our corpus was already embedded with
        the proximity adapter (correct for the candidate side), so here — on the
        query side — we use adhoc_query. The two adapters are trained to be
        compatible, so an adhoc_query query embedding is directly comparable to
        the proximity corpus embeddings.

        (Title mode does NOT use this model at all — it reuses a paper's
        precomputed proximity embedding, i.e. paper-to-paper, which is correct.)

    Returns:
        (tokenizer, model) tuple with the adhoc_query adapter active.
    """
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    logger.info("Loading SPECTER2 (adhoc_query adapter) for keyword embedding...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model     = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(
        "allenai/specter2_adhoc_query", source="hf",
        load_as="adhoc_query", set_active=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("SPECTER2 (adhoc_query) loaded on %s", device)
    return tokenizer, model


def _load_scibert_model():
    """
    Loads SciBERT for on-the-fly keyword embedding.

    Returns:
        (tokenizer, model) tuple.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading SciBERT for keyword embedding...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model     = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("SciBERT loaded on %s", device)
    return tokenizer, model


# ---------------------------------------------------------------------------
# PaperRecommender
# ---------------------------------------------------------------------------

class PaperRecommender:
    """
    Content-based paper recommender (spec Section 9).

    Ranking = cosine similarity, with a validated multiplicative category boost
    (ALPHA = 0.03) applied only when the query has a known category (title mode).

    Args:
        model: "specter2" (primary, per spec) or "scibert" (comparison).

    Example:
        rec = PaperRecommender(model="specter2")
        results = rec.recommend("attention is all you need", mode="title", top_k=10)
    """

    def __init__(self, model: Literal["specter2", "scibert"] = "specter2"):
        if model not in MODEL_FILES:
            raise ValueError(f"model must be 'specter2' or 'scibert', got '{model}'")

        self.model_name = model
        self._encoder   = None   # lazy-loaded only if keyword mode is used

        logger.info("Initializing PaperRecommender (model=%s)", model)

        # --- Load precomputed embeddings + their arxiv_ids (same row order) ---
        paths = MODEL_FILES[model]
        self.embeddings = np.load(paths["embeddings"]).astype(np.float32)   # (N, 768)
        # arxiv_ids saved as strings in the embedding scripts — keep them as str.
        self.arxiv_ids  = np.array([str(a) for a in np.load(paths["arxiv_ids"])])

        logger.info(
            "Embeddings loaded: %s | %d papers",
            self.embeddings.shape, len(self.arxiv_ids),
        )

        # --- Pre-normalize embeddings once so cosine == dot product ---
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-10, a_max=None)
        self.embeddings_normalized = self.embeddings / norms

        # --- Load paper metadata (arxiv_id as string to avoid float corruption) ---
        logger.info("Loading paper metadata from %s", DATA_FILE)
        self.papers = pd.read_csv(DATA_FILE, dtype=str)

        # citation_count kept numeric for display only (not used in ranking).
        self.papers["citation_count"] = (
            pd.to_numeric(self.papers["citation_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # arxiv_id -> metadata row index, for fast lookups.
        self.id_to_row = {
            str(arxiv_id): idx
            for idx, arxiv_id in enumerate(self.papers["arxiv_id"].astype(str))
        }

        # Category aligned to EMBEDDING row order (used to build the boost mask).
        id_to_category = dict(
            zip(self.papers["arxiv_id"].astype(str), self.papers["category"].astype(str))
        )
        self.emb_categories = np.array(
            [id_to_category.get(aid, "") for aid in self.arxiv_ids]
        )

        # Publication year aligned to EMBEDDING row order (used for year filter).
        id_to_year = dict(
            zip(self.papers["arxiv_id"].astype(str),
                self.papers["publication_year"].astype(str))
        )
        self.emb_years = np.array(
            [id_to_year.get(aid, "") for aid in self.arxiv_ids]
        )

        # Lowercased titles for fuzzy matching (title mode).
        self.titles_lower = self.papers["title"].astype(str).str.lower().tolist()

        logger.info("PaperRecommender ready. %d papers indexed.", len(self.papers))

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def recommend(
        self,
        query: str,
        mode:  Literal["title", "keywords"] = "title",
        top_k: int = DEFAULT_TOP_K,
        category: str | None = None,
        year: str | int | None = None,
    ) -> list[dict]:
        """
        Returns the top-K recommended papers for a query.

        Args:
            query: A paper title (mode="title") or free-text keywords
                   (mode="keywords").
            mode:  "title"    -> fuzzy-match against dataset titles and reuse
                                 the matched paper's precomputed embedding.
                   "keywords" -> embed the query text on the fly.
            top_k: Number of recommendations to return (default 10).
            category: If set, restrict recommendations to this arXiv category
                      (e.g. "cs.CV"). Filtering is applied to the candidate pool
                      BEFORE ranking, so you get the true top-K within the filter.
            year:     If set, restrict recommendations to this publication year.

        Returns:
            List of result dicts (highest final_score first), each with:
                arxiv_id, title, authors, abstract_preview, category,
                publication_year, similarity_score (raw cosine),
                final_score (after category boost), boosted (bool),
                citation_count, tags
            Returns [] if no paper in the corpus matches the active filters.
        """
        logger.info(
            "Query: '%s' | mode=%s | top_k=%d | model=%s",
            query, mode, top_k, self.model_name,
        )

        # --- Step 1: query embedding (+ category in title mode) ---
        if mode == "title":
            query_embedding, matched_id, query_category = self._embed_by_title(query)
        elif mode == "keywords":
            query_embedding = self._embed_keywords(query)
            matched_id      = None
            query_category  = None   # no category -> no boost in keyword mode
        else:
            raise ValueError(f"mode must be 'title' or 'keywords', got '{mode}'")

        # --- Step 2: cosine similarity against all papers ---
        cosine = self._cosine_against_corpus(query_embedding)

        # --- Step 3: validated multiplicative category boost ---
        #   final = cosine * (1 + ALPHA * same_category_flag)
        # Only applied when we know the query's category (title mode).
        if query_category and CATEGORY_BOOST_ALPHA > 0.0:
            same_category = (self.emb_categories == query_category).astype(np.float32)
            final_scores  = cosine * (1.0 + CATEGORY_BOOST_ALPHA * same_category)
            boost_applied = True
        else:
            final_scores  = cosine.copy()
            boost_applied = False

        # --- Step 4: exclude the query paper itself (title mode) ---
        if matched_id is not None:
            self_pos = np.where(self.arxiv_ids == matched_id)[0]
            if len(self_pos) > 0:
                final_scores[self_pos[0]] = -np.inf

        # --- Step 4b: restrict the candidate pool by year / category ---
        # Applied BEFORE top-K selection, so results are the true top-K WITHIN
        # the filter (non-matching papers are pushed to -inf and dropped below).
        if category or year is not None:
            keep = np.ones(len(final_scores), dtype=bool)
            if category:
                keep &= (self.emb_categories == str(category))
            if year is not None:
                keep &= (self.emb_years == str(year))
            if not keep.any():
                return []  # nothing in the corpus matches these filters
            final_scores = np.where(keep, final_scores, -np.inf)

        # --- Step 5: take top-K by final score (finite scores only) ---
        k = min(top_k, len(final_scores))
        top_idx = np.argpartition(final_scores, -k)[-k:]
        top_idx = top_idx[np.argsort(-final_scores[top_idx])]
        # Drop any -inf entries (self-paper, or filtered-out pool smaller than k).
        top_idx = [int(idx) for idx in top_idx if np.isfinite(final_scores[idx])]

        # --- Step 6: build result dicts with metadata ---
        return [
            self._build_result(int(idx), cosine, final_scores, boost_applied)
            for idx in top_idx
        ]

    def match_title(self, query: str) -> dict | None:
        """
        Returns metadata for the paper whose title best matches `query` — i.e.
        the paper that title-mode recommendations are based on — or None if no
        title scores above FUZZY_THRESHOLD.

        Useful for the UI to confirm WHICH paper a (fuzzy) title query matched,
        shown separately from the recommendations themselves.
        """
        match = fuzz_process.extractOne(
            query.lower(), self.titles_lower, scorer=FUZZY_SCORER,
        )
        if match is None or match[1] < FUZZY_THRESHOLD:
            return None

        row = self.papers.iloc[match[2]]
        return {
            "arxiv_id":         str(row["arxiv_id"]),
            "title":            str(row["title"]),
            "authors":          str(row["authors"]),
            "category":         str(row["category"]),
            "publication_year": str(row["publication_year"]),
            "match_score":      round(float(match[1]), 1),   # 0-100 fuzzy score
        }

    # -----------------------------------------------------------------------
    # Step 1: query embedding
    # -----------------------------------------------------------------------

    def _embed_by_title(self, query: str) -> tuple[np.ndarray, str, str]:
        """
        Fuzzy-matches the query against dataset titles and returns the matched
        paper's precomputed embedding, its arxiv_id, and its category.

        Returns:
            (embedding (768,), matched_arxiv_id, matched_category)

        Raises:
            ValueError if no title scores above FUZZY_THRESHOLD, or if the
            matched paper has no embedding (metadata/embedding out of sync).
        """
        logger.info("Fuzzy matching title: '%s'", query)

        match = fuzz_process.extractOne(
            query.lower(),
            self.titles_lower,
            scorer=FUZZY_SCORER,   # order-insensitive, length-sensitive
        )

        if match is None or match[1] < FUZZY_THRESHOLD:
            best = match[1] if match else 0
            raise ValueError(
                f"No paper found matching '{query}' "
                f"(best score {best} < threshold {FUZZY_THRESHOLD}). "
                f"Try different wording or use mode='keywords'."
            )

        match_idx   = match[2]
        matched_row = self.papers.iloc[match_idx]
        arxiv_id    = str(matched_row["arxiv_id"])
        category    = str(matched_row["category"])

        logger.info("Matched: '%s' (score %.1f)", matched_row["title"], match[1])

        emb_pos = np.where(self.arxiv_ids == arxiv_id)[0]
        if len(emb_pos) == 0:
            raise ValueError(
                f"Matched paper '{matched_row['title']}' (arxiv_id {arxiv_id}) "
                f"has no embedding — metadata and embedding files are out of sync."
            )

        return self.embeddings[emb_pos[0]], arxiv_id, category

    def _embed_keywords(self, keywords: str) -> np.ndarray:
        """
        Generates an embedding for free-text keywords using the selected model.
        The model is loaded lazily — only on the first keyword query.

        Args:
            keywords: Free-text research keywords or short description.

        Returns:
            Embedding array of shape (768,).
        """
        import torch

        if self._encoder is None:
            self._encoder = (
                _load_specter2_model() if self.model_name == "specter2"
                else _load_scibert_model()
            )

        tokenizer, model = self._encoder
        device = next(model.parameters()).device

        # The adhoc_query adapter (SPECTER2) and SciBERT both take the raw short
        # query text as-is — this is a "short text query", not a title+abstract,
        # so we do NOT reshape it into a paper-like input.
        input_text = keywords

        inputs = tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        if self.model_name == "specter2":
            # CLS-token pooling (matches embeddings_specter2.py)
            embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        else:
            # Mean pooling (matches embeddings_scibert.py)
            mask      = inputs["attention_mask"][0].unsqueeze(-1).float().cpu()
            hidden    = outputs.last_hidden_state[0].cpu()
            embedding = (hidden * mask).sum(0) / mask.sum().clamp(min=1e-9)
            embedding = embedding.numpy()

        return embedding.astype(np.float32)

    # -----------------------------------------------------------------------
    # Step 2: cosine similarity
    # -----------------------------------------------------------------------

    def _cosine_against_corpus(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between the query embedding and all corpus embeddings.
        Corpus embeddings are pre-normalized, so this is a single dot product.

        Returns:
            Array of shape (N,) with cosine similarities in [-1, 1].
        """
        norm = np.linalg.norm(query_embedding)
        if norm < 1e-10:
            raise ValueError("Query embedding is a zero vector — cannot rank.")
        query_norm = query_embedding / norm
        return self.embeddings_normalized @ query_norm

    # -----------------------------------------------------------------------
    # Step 6: result construction
    # -----------------------------------------------------------------------

    def _build_result(
        self,
        emb_idx:       int,
        cosine:        np.ndarray,
        final_scores:  np.ndarray,
        boost_applied: bool,
    ) -> dict:
        """
        Builds a single result dict for the paper at embedding row `emb_idx`.
        """
        arxiv_id = str(self.arxiv_ids[emb_idx])
        row      = self.papers.iloc[self.id_to_row[arxiv_id]]

        return {
            "arxiv_id":         arxiv_id,
            "title":            str(row["title"]),
            "authors":          str(row["authors"]),
            "abstract_preview": str(row["abstract"])[:300],
            "category":         str(row["category"]),
            "publication_year": str(row["publication_year"]),
            "similarity_score": round(float(cosine[emb_idx]), 4),       # raw cosine
            "final_score":      round(float(final_scores[emb_idx]), 4),  # after boost
            "boosted":          boost_applied,
            "citation_count":   int(row["citation_count"]),
            "tags":             str(row.get("keybert_tags_v2", "")),
        }