"""Create SciBERT embeddings for the paper corpus."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Config

INPUT_FILE      = Path("data/processed/papers_keybert_v2.csv")
OUTPUT_DIR      = Path("models/embeddings")
EMBEDDINGS_FILE = OUTPUT_DIR / "scibert_embeddings.npy"
ARXIV_IDS_FILE  = OUTPUT_DIR / "scibert_arxiv_ids.npy"

MODEL_NAME = "allenai/scibert_scivocab_uncased"

# Same batch size guidance as SPECTER2:
# 32 — safe for 8GB VRAM
# 64 — safe for 16GB VRAM
# 16 — safe for CPU
BATCH_SIZE = 32

MAX_LENGTH = 512


# Device

def get_device() -> torch.device:
    """Pick CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("CUDA available — using GPU: %s (%.1f GB VRAM)", gpu_name, vram)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — falling back to CPU (will be slow)")
    return device


# Inputs
# (identical to SPECTER2 script — same input format)

def load_papers(filepath: Path) -> pd.DataFrame:
    """Load the paper CSV and check required columns."""
    df = pd.read_csv(filepath, dtype=str)
    logger.info("Loaded %d papers from %s", len(df), filepath)

    required = ["arxiv_id", "title", "abstract"]
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df["title"]    = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    return df


def build_input_texts(df: pd.DataFrame, sep_token: str) -> list[str]:
    """Build model input strings from title and abstract."""
    return [
        f"{row['title']}{sep_token}{row['abstract']}"
        for _, row in df.iterrows()
    ]


# Mean pooling

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool."""
    # Expand the attention mask.
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Mask padding tokens.
    sum_hidden = torch.sum(last_hidden_state * mask_expanded, dim=1)

    # Count non-padding tokens.
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    return sum_hidden / sum_mask


# Embeddings

def generate_embeddings(
    texts:     list[str],
    tokenizer: AutoTokenizer,
    model:     AutoModel,
    device:    torch.device,
) -> np.ndarray:
    """Run batched embedding inference."""
    total          = len(texts)
    all_embeddings = []

    model.eval()

    start_time = time.time()

    with torch.no_grad():
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end   = min(batch_start + BATCH_SIZE, total)
            batch_texts = texts[batch_start:batch_end]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # Mean pooling — different from SPECTER2's CLS pooling
            batch_embeddings = mean_pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            all_embeddings.append(batch_embeddings.cpu().numpy())

            if batch_end % 1000 == 0 or batch_end == total:
                elapsed   = time.time() - start_time
                rate      = batch_end / elapsed
                remaining = (total - batch_end) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d / %d papers | %.1f papers/sec | ETA: %.0f min",
                    batch_end, total, rate, remaining / 60
                )

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    total_time = time.time() - start_time
    logger.info(
        "Embedding generation complete. Time: %.0f min | Shape: %s",
        total_time / 60, embeddings.shape
    )

    return embeddings


# Checks
# (identical logic to SPECTER2 script)

def verify_embeddings(embeddings: np.ndarray, arxiv_ids: np.ndarray) -> None:
    """Check the saved embedding arrays before writing them."""
    logger.info("Running verification checks...")

    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
    assert embeddings.shape[1] == 768, f"Expected 768 dims, got {embeddings.shape[1]}"
    logger.info("  Shape check passed: %s", embeddings.shape)

    nan_count = np.isnan(embeddings).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in embeddings"
    logger.info("  NaN check passed")

    zero_vectors = np.all(embeddings == 0, axis=1).sum()
    assert zero_vectors == 0, f"Found {zero_vectors} all-zero vectors"
    logger.info("  Zero vector check passed")

    assert len(embeddings) == len(arxiv_ids), (
        f"Embedding count ({len(embeddings)}) != arxiv_id count ({len(arxiv_ids)})"
    )
    logger.info("  ID sync check passed")

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_self = cosine_sim(embeddings[0], embeddings[0])
    sim_01   = cosine_sim(embeddings[0], embeddings[1])
    sim_02   = cosine_sim(embeddings[0], embeddings[2])

    logger.info("  Cosine similarity spot-check:")
    logger.info("    paper[0] vs paper[0] (should be 1.0): %.4f", sim_self)
    logger.info("    paper[0] vs paper[1]: %.4f", sim_01)
    logger.info("    paper[0] vs paper[2]: %.4f", sim_02)

    assert abs(sim_self - 1.0) < 1e-5, "Self-similarity is not 1.0"
    logger.info("  All verification checks passed.")


# Main

def generate_scibert_embeddings() -> None:
    """Generate scibert embeddings."""
    logger.info("=" * 60)
    logger.info("SciBERT Embedding Generation")
    logger.info("Model : %s", MODEL_NAME)
    logger.info("Input : %s", INPUT_FILE)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("=" * 60)

    device = get_device()

    df = load_papers(INPUT_FILE)

    logger.info("Loading SciBERT tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    logger.info("Model loaded and moved to %s", device)

    logger.info("Building title [SEP] abstract inputs...")
    texts = build_input_texts(df, tokenizer.sep_token)
    logger.info("Input texts ready. Example: %s", texts[0][:120])

    logger.info("Starting embedding generation for %d papers...", len(texts))
    embeddings = generate_embeddings(texts, tokenizer, model, device)

    arxiv_ids = df["arxiv_id"].values.astype(str)

    verify_embeddings(embeddings, arxiv_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)
    logger.info("Saved embeddings -> %s  (shape: %s, size: %.1f MB)",
                EMBEDDINGS_FILE, embeddings.shape,
                embeddings.nbytes / 1024**2)

    np.save(ARXIV_IDS_FILE, arxiv_ids)
    logger.info("Saved arxiv_ids  -> %s  (%d IDs)",
                ARXIV_IDS_FILE, len(arxiv_ids))

    logger.info("=" * 60)
    logger.info("SciBERT embedding generation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_scibert_embeddings()