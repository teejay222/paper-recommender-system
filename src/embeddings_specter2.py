"""
src/embeddings_specter2.py
--------------------------
Generates SPECTER2 embeddings for all papers.
Designed to run on a CUDA GPU (university cluster).
Falls back to CPU automatically if CUDA is not available.

Model:
  Base  : allenai/specter2_base
  Adapter: allenai/specter2 (proximity/retrieval adapter)
  Output: 768-dimensional float32 vectors

Input:
  data/processed/papers_keybert_v2.csv

Output:
  models/embeddings/specter2_embeddings.npy  — shape (N, 768)
  models/embeddings/specter2_arxiv_ids.npy   — shape (N,) arxiv_ids in same row order

Why two output files:
  The .npy row index must map back to a paper.
  specter2_arxiv_ids.npy[i] is the arxiv_id for specter2_embeddings.npy[i].
  These two files must always stay in sync.

Install requirement (run once before this script):
  pip install -U adapters

Usage:
  python src/embeddings_specter2.py
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE   = Path("data/processed/papers_keybert_v2.csv")
OUTPUT_DIR   = Path("models/embeddings")
EMBEDDINGS_FILE = OUTPUT_DIR / "specter2_embeddings.npy"
ARXIV_IDS_FILE  = OUTPUT_DIR / "specter2_arxiv_ids.npy"

# Model identifiers
BASE_MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME    = "allenai/specter2"        # proximity/retrieval adapter

# Batch size:
#   32  — safe for 8GB VRAM
#   64  — safe for 16GB VRAM
#   16  — safe for CPU
# Change this if you get CUDA out-of-memory errors
BATCH_SIZE = 32

# Max token length — SPECTER2 was trained with 512
MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Returns CUDA device if available, otherwise CPU.
    Logs which device is being used so cluster logs are clear.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("CUDA available — using GPU: %s (%.1f GB VRAM)", gpu_name, vram)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — falling back to CPU (will be slow)")
    return device


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def load_papers(filepath: Path) -> pd.DataFrame:
    """
    Loads the CSV and validates required columns exist.

    Only title and abstract are used for embeddings.
    All other columns (tags, citation counts etc.) are ignored here.

    Args:
        filepath: Path to papers_keybert_v2.csv

    Returns:
        DataFrame with at minimum arxiv_id, title, abstract columns.
    """
    df = pd.read_csv(filepath, dtype=str)
    logger.info("Loaded %d papers from %s", len(df), filepath)

    required = ["arxiv_id", "title", "abstract"]
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Fill any NaN in title or abstract with empty string
    # so tokenizer doesn't crash on rare edge cases
    df["title"]    = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    return df


def build_input_texts(df: pd.DataFrame, sep_token: str) -> list[str]:
    """
    Concatenates title and abstract using the model's separator token.

    SPECTER2 was trained on inputs formatted as:
        title [SEP] abstract
    where [SEP] is the tokenizer's actual separator token.
    Using the correct separator is important — it is what the model
    was trained to expect as the boundary between title and abstract.

    Args:
        df:        DataFrame with title and abstract columns
        sep_token: The tokenizer's separator token string (e.g. "[SEP]")

    Returns:
        List of formatted input strings, one per paper.
    """
    return [
        f"{row['title']}{sep_token}{row['abstract']}"
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts:     list[str],
    tokenizer: AutoTokenizer,
    model:     AutoAdapterModel,
    device:    torch.device,
) -> np.ndarray:
    """
    Generates SPECTER2 embeddings for all input texts using batched inference.

    Pooling strategy: CLS token (index 0 of last hidden state).
    This is the correct pooling method for SPECTER2 — it was trained
    using CLS token representations.

    Args:
        texts:     List of "title [SEP] abstract" strings
        tokenizer: SPECTER2 tokenizer
        model:     SPECTER2 model with proximity adapter loaded
        device:    torch.device (cuda or cpu)

    Returns:
        NumPy array of shape (len(texts), 768) in float32.
    """
    total      = len(texts)
    all_embeddings = []

    model.eval()   # disable dropout — we are doing inference not training

    start_time = time.time()

    with torch.no_grad():
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end   = min(batch_start + BATCH_SIZE, total)
            batch_texts = texts[batch_start:batch_end]

            # Tokenize the batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            # Move inputs to GPU/CPU
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)

            # CLS token pooling: take the first token of the last hidden state
            # Shape: (batch_size, 768)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

            # Move back to CPU and convert to numpy before storing
            # Keeping tensors on GPU across all batches would exhaust VRAM
            all_embeddings.append(batch_embeddings.cpu().numpy())

            # Progress logging every 1000 papers
            if batch_end % 1000 == 0 or batch_end == total:
                elapsed  = time.time() - start_time
                rate     = batch_end / elapsed
                remaining = (total - batch_end) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d / %d papers | %.1f papers/sec | ETA: %.0f min",
                    batch_end, total, rate, remaining / 60
                )

    # Stack all batch results into one array
    embeddings = np.vstack(all_embeddings).astype(np.float32)

    total_time = time.time() - start_time
    logger.info(
        "Embedding generation complete. Time: %.0f min | Shape: %s",
        total_time / 60, embeddings.shape
    )

    return embeddings


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_embeddings(embeddings: np.ndarray, arxiv_ids: np.ndarray) -> None:
    """
    Runs sanity checks on the generated embeddings before saving.

    Checks:
      - Shape is (N, 768)
      - No NaN values
      - No all-zero vectors (would indicate failed forward pass)
      - arxiv_ids length matches embedding count
      - Spot-checks cosine similarity between first 3 papers

    Args:
        embeddings: Float32 array of shape (N, 768)
        arxiv_ids:  String array of shape (N,)
    """
    logger.info("Running verification checks...")

    # Shape check
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
    assert embeddings.shape[1] == 768, f"Expected 768 dims, got {embeddings.shape[1]}"
    logger.info("  Shape check passed: %s", embeddings.shape)

    # NaN check
    nan_count = np.isnan(embeddings).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in embeddings"
    logger.info("  NaN check passed: 0 NaN values")

    # Zero vector check
    zero_vectors = np.all(embeddings == 0, axis=1).sum()
    assert zero_vectors == 0, f"Found {zero_vectors} all-zero embedding vectors"
    logger.info("  Zero vector check passed: 0 zero vectors")

    # Length sync check
    assert len(embeddings) == len(arxiv_ids), (
        f"Embedding count ({len(embeddings)}) != arxiv_id count ({len(arxiv_ids)})"
    )
    logger.info("  ID sync check passed: %d embeddings match %d arxiv_ids",
                len(embeddings), len(arxiv_ids))

    # Spot-check cosine similarity between first 3 papers
    # Normalized dot product = cosine similarity
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_01 = cosine_sim(embeddings[0], embeddings[1])
    sim_02 = cosine_sim(embeddings[0], embeddings[2])
    sim_self = cosine_sim(embeddings[0], embeddings[0])

    logger.info("  Cosine similarity spot-check:")
    logger.info("    paper[0] vs paper[0] (should be 1.0): %.4f", sim_self)
    logger.info("    paper[0] vs paper[1]: %.4f", sim_01)
    logger.info("    paper[0] vs paper[2]: %.4f", sim_02)

    assert abs(sim_self - 1.0) < 1e-5, "Self-similarity is not 1.0 — something is wrong"
    logger.info("  All verification checks passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_specter2_embeddings() -> None:
    """
    Full pipeline:
      1. Load papers
      2. Load SPECTER2 model + proximity adapter
      3. Generate embeddings in batches
      4. Verify embeddings
      5. Save embeddings and arxiv_ids to models/embeddings/
    """
    logger.info("=" * 60)
    logger.info("SPECTER2 Embedding Generation")
    logger.info("Input : %s", INPUT_FILE)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("=" * 60)

    # --- Device ---
    device = get_device()

    # --- Load papers ---
    df = load_papers(INPUT_FILE)

    # --- Load model and adapter ---
    logger.info("Loading tokenizer: %s", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    logger.info("Loading base model: %s", BASE_MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(BASE_MODEL_NAME)

    logger.info("Loading proximity adapter: %s", ADAPTER_NAME)
    model.load_adapter(
        ADAPTER_NAME,
        source="hf",
        load_as="proximity",
        set_active=True,
    )

    model.to(device)
    logger.info("Model loaded and moved to %s", device)

    # --- Build input texts ---
    logger.info("Building title [SEP] abstract inputs...")
    texts = build_input_texts(df, tokenizer.sep_token)
    logger.info("Input texts ready. Example: %s", texts[0][:120])

    # --- Generate embeddings ---
    logger.info("Starting embedding generation for %d papers...", len(texts))
    embeddings = generate_embeddings(texts, tokenizer, model, device)

    # --- Prepare arxiv_ids array ---
    arxiv_ids = df["arxiv_id"].values.astype(str)

    # --- Verify ---
    verify_embeddings(embeddings, arxiv_ids)

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)
    logger.info("Saved embeddings -> %s  (shape: %s, size: %.1f MB)",
                EMBEDDINGS_FILE, embeddings.shape,
                embeddings.nbytes / 1024**2)

    np.save(ARXIV_IDS_FILE, arxiv_ids)
    logger.info("Saved arxiv_ids  -> %s  (%d IDs)",
                ARXIV_IDS_FILE, len(arxiv_ids))

    logger.info("=" * 60)
    logger.info("SPECTER2 embedding generation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_specter2_embeddings()