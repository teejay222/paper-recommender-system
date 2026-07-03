"""Recommendation engine used by the Streamlit app."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Paths

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


# Ranking configuration

# Category boost used after cosine scoring.
# Checked in experiments_with_alpha.py.
# Use 0.0 for plain cosine ranking.
CATEGORY_BOOST_ALPHA = 0.03

# Minimum fuzzy title score.
FUZZY_THRESHOLD = 80

# token_sort_ratio avoids prefix-only false matches.
FUZZY_SCORER = fuzz.token_sort_ratio

# Default recommendation count.
DEFAULT_TOP_K = 10


# Encoders for keyword search

def _load_specter2_model():
    """Load the SPECTER2 query model."""
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
    """Load the SciBERT query model."""
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


# PaperRecommender

class PaperRecommender:
    """Loads corpus data and ranks papers."""

    def __init__(self, model: Literal["specter2", "scibert"] = "specter2"):
        if model not in MODEL_FILES:
            raise ValueError(f"model must be 'specter2' or 'scibert', got '{model}'")

        self.model_name = model
        self._encoder   = None   # loaded on demand

        logger.info("Initializing PaperRecommender (model=%s)", model)

        # Load embeddings and ids in the same order.
        paths = MODEL_FILES[model]
        self.embeddings = np.load(paths["embeddings"]).astype(np.float32)   # (N, 768)
        # Keep ids as strings.
        self.arxiv_ids  = np.array([str(a) for a in np.load(paths["arxiv_ids"])])

        logger.info(
            "Embeddings loaded: %s | %d papers",
            self.embeddings.shape, len(self.arxiv_ids),
        )

        # Normalize once; cosine becomes a dot product.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-10, a_max=None)
        self.embeddings_normalized = self.embeddings / norms

        # Load metadata.
        logger.info("Loading paper metadata from %s", DATA_FILE)
        self.papers = pd.read_csv(DATA_FILE, dtype=str)

        # citation_count is display-only.
        self.papers["citation_count"] = (
            pd.to_numeric(self.papers["citation_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

        # Fast lookup by arxiv_id.
        self.id_to_row = {
            str(arxiv_id): idx
            for idx, arxiv_id in enumerate(self.papers["arxiv_id"].astype(str))
        }

        # Category in embedding row order.
        id_to_category = dict(
            zip(self.papers["arxiv_id"].astype(str), self.papers["category"].astype(str))
        )
        self.emb_categories = np.array(
            [id_to_category.get(aid, "") for aid in self.arxiv_ids]
        )

        # Year in embedding row order.
        id_to_year = dict(
            zip(self.papers["arxiv_id"].astype(str),
                self.papers["publication_year"].astype(str))
        )
        self.emb_years = np.array(
            [id_to_year.get(aid, "") for aid in self.arxiv_ids]
        )

        # Lowercase titles for matching.
        self.titles_lower = self.papers["title"].astype(str).str.lower().tolist()

        logger.info("PaperRecommender ready. %d papers indexed.", len(self.papers))

    # Public methods

    def recommend(
        self,
        query: str,
        mode:  Literal["title", "keywords"] = "title",
        top_k: int = DEFAULT_TOP_K,
        category: str | None = None,
        year: str | int | None = None,
    ) -> list[dict]:
        """Recommend."""
        logger.info(
            "Query: '%s' | mode=%s | top_k=%d | model=%s",
            query, mode, top_k, self.model_name,
        )

        # Build the query embedding.
        if mode == "title":
            query_embedding, matched_id, query_category = self._embed_by_title(query)
        elif mode == "keywords":
            query_embedding = self._embed_keywords(query)
            matched_id      = None
            query_category  = None   # Keyword mode has no category boost.
        else:
            raise ValueError(f"mode must be 'title' or 'keywords', got '{mode}'")

        # Score against all papers.
        cosine = self._cosine_against_corpus(query_embedding)

        # Apply category boost.
        # boosted score = cosine * category weight
        # Only title mode has the query category.
        if query_category and CATEGORY_BOOST_ALPHA > 0.0:
            same_category = (self.emb_categories == query_category).astype(np.float32)
            final_scores  = cosine * (1.0 + CATEGORY_BOOST_ALPHA * same_category)
            boost_applied = True
        else:
            final_scores  = cosine.copy()
            boost_applied = False

        # Remove the query paper.
        if matched_id is not None:
            self_pos = np.where(self.arxiv_ids == matched_id)[0]
            if len(self_pos) > 0:
                final_scores[self_pos[0]] = -np.inf

        # Step 4b: restrict the candidate pool by year / category
        if category or year is not None:
            keep = np.ones(len(final_scores), dtype=bool)
            if category:
                keep &= (self.emb_categories == str(category))
            if year is not None:
                keep &= (self.emb_years == str(year))
            if not keep.any():
                return []  # No paper matches these filters.
            final_scores = np.where(keep, final_scores, -np.inf)

        # Take the top finite scores.
        k = min(top_k, len(final_scores))
        top_idx = np.argpartition(final_scores, -k)[-k:]
        top_idx = top_idx[np.argsort(-final_scores[top_idx])]
        # Drop filtered rows.
        top_idx = [int(idx) for idx in top_idx if np.isfinite(final_scores[idx])]

        # Build result rows.
        return [
            self._build_result(int(idx), cosine, final_scores, boost_applied)
            for idx in top_idx
        ]

    def match_title(self, query: str) -> dict | None:
        """Match title."""
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

    # query embedding

    def _embed_by_title(self, query: str) -> tuple[np.ndarray, str, str]:
        """Embed by title."""
        logger.info("Fuzzy matching title: '%s'", query)

        match = fuzz_process.extractOne(
            query.lower(),
            self.titles_lower,
            scorer=FUZZY_SCORER,   # order-insensitive score
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
        """Embed keywords."""
        import torch

        if self._encoder is None:
            self._encoder = (
                _load_specter2_model() if self.model_name == "specter2"
                else _load_scibert_model()
            )

        tokenizer, model = self._encoder
        device = next(model.parameters()).device

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
            # CLS pooling.
            embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        else:
            # Mean pooling.
            mask      = inputs["attention_mask"][0].unsqueeze(-1).float().cpu()
            hidden    = outputs.last_hidden_state[0].cpu()
            embedding = (hidden * mask).sum(0) / mask.sum().clamp(min=1e-9)
            embedding = embedding.numpy()

        return embedding.astype(np.float32)

    # Cosine scores

    def _cosine_against_corpus(self, query_embedding: np.ndarray) -> np.ndarray:
        """Cosine against corpus."""
        norm = np.linalg.norm(query_embedding)
        if norm < 1e-10:
            raise ValueError("Query embedding is a zero vector — cannot rank.")
        query_norm = query_embedding / norm
        return self.embeddings_normalized @ query_norm

    # Build results

    def _build_result(
        self,
        emb_idx:       int,
        cosine:        np.ndarray,
        final_scores:  np.ndarray,
        boost_applied: bool,
    ) -> dict:
        """Build result."""
        arxiv_id = str(self.arxiv_ids[emb_idx])
        row      = self.papers.iloc[self.id_to_row[arxiv_id]]

        return {
            "arxiv_id":         arxiv_id,
            "title":            str(row["title"]),
            "authors":          str(row["authors"]),
            "abstract_preview": str(row["abstract"])[:300],
            "category":         str(row["category"]),
            "publication_year": str(row["publication_year"]),
            "similarity_score": round(float(cosine[emb_idx]), 4),       # raw score
            "final_score":      round(float(final_scores[emb_idx]), 4),  # boosted score
            "boosted":          boost_applied,
            "citation_count":   int(row["citation_count"]),
            "tags":             str(row.get("keybert_tags_v2", "")),
        }