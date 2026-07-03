"""Create SPECTER2 embeddings for the paper corpus."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Config

INPUT_FILE   = Path("data/processed/papers_keybert_v2.csv")
OUTPUT_DIR   = Path("models/embeddings")
EMBEDDINGS_FILE = OUTPUT_DIR / "specter2_embeddings.npy"
ARXIV_IDS_FILE  = OUTPUT_DIR / "specter2_arxiv_ids.npy"

# Model ids
BASE_MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME    = "allenai/specter2"        # proximity/retrieval adapter

# Batch size:
# 32 — safe for 8GB VRAM
# 64 — safe for 16GB VRAM
# 16 — safe for CPU
# Change this if you get CUDA out-of-memory errors
BATCH_SIZE = 32

# Max token length — SPECTER2 was trained with 512
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

def load_papers(filepath: Path) -> pd.DataFrame:
    """Load the paper CSV and check required columns."""
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
    """Build model input strings from title and abstract."""
    return [
        f"{row['title']}{sep_token}{row['abstract']}"
        for _, row in df.iterrows()
    ]


# Embeddings

def generate_embeddings(
    texts:     list[str],
    tokenizer: AutoTokenizer,
    model:     AutoAdapterModel,
    device:    torch.device,
) -> np.ndarray:
    """Run batched embedding inference."""
    total      = len(texts)
    all_embeddings = []

    model.eval()   # disable dropout — we are doing inference not training

    start_time = time.time()

    with torch.no_grad():
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end   = min(batch_start + BATCH_SIZE, total)
            batch_texts = texts[batch_start:batch_end]

            # Tokenize batch.
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            # Move tensors to the device.
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run the model.
            outputs = model(**inputs)

            # CLS pooling.
            # Shape: (batch_size, 768)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

            # Move results back to CPU.
            # Do not keep every batch on the GPU.
            all_embeddings.append(batch_embeddings.cpu().numpy())

            # Log progress every 1000 papers.
            if batch_end % 1000 == 0 or batch_end == total:
                elapsed  = time.time() - start_time
                rate     = batch_end / elapsed
                remaining = (total - batch_end) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d / %d papers | %.1f papers/sec | ETA: %.0f min",
                    batch_end, total, rate, remaining / 60
                )

    # Stack batch outputs.
    embeddings = np.vstack(all_embeddings).astype(np.float32)

    total_time = time.time() - start_time
    logger.info(
        "Embedding generation complete. Time: %.0f min | Shape: %s",
        total_time / 60, embeddings.shape
    )

    return embeddings


# Checks

def verify_embeddings(embeddings: np.ndarray, arxiv_ids: np.ndarray) -> None:
    """Check the saved embedding arrays before writing them."""
    logger.info("Running verification checks...")

    # Shape
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
    assert embeddings.shape[1] == 768, f"Expected 768 dims, got {embeddings.shape[1]}"
    logger.info("  Shape check passed: %s", embeddings.shape)

    # NaNs
    nan_count = np.isnan(embeddings).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in embeddings"
    logger.info("  NaN check passed: 0 NaN values")

    # Zero vectors
    zero_vectors = np.all(embeddings == 0, axis=1).sum()
    assert zero_vectors == 0, f"Found {zero_vectors} all-zero embedding vectors"
    logger.info("  Zero vector check passed: 0 zero vectors")

    # Length check
    assert len(embeddings) == len(arxiv_ids), (
        f"Embedding count ({len(embeddings)}) != arxiv_id count ({len(arxiv_ids)})"
    )
    logger.info("  ID sync check passed: %d embeddings match %d arxiv_ids",
                len(embeddings), len(arxiv_ids))

    # Small cosine check.
    # Normalized dot product is cosine.
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


# Main

def generate_specter2_embeddings() -> None:
    """Generate specter2 embeddings."""
    logger.info("=" * 60)
    logger.info("SPECTER2 Embedding Generation")
    logger.info("Input : %s", INPUT_FILE)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Batch size: %d", BATCH_SIZE)
    logger.info("=" * 60)

    # Device
    device = get_device()

    # Load papers
    df = load_papers(INPUT_FILE)

    # Load model and adapter
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

    # Build input texts
    logger.info("Building title [SEP] abstract inputs...")
    texts = build_input_texts(df, tokenizer.sep_token)
    logger.info("Input texts ready. Example: %s", texts[0][:120])

    # Generate embeddings
    logger.info("Starting embedding generation for %d papers...", len(texts))
    embeddings = generate_embeddings(texts, tokenizer, model, device)

    # Prepare arxiv_ids array
    arxiv_ids = df["arxiv_id"].values.astype(str)

    # Verify
    verify_embeddings(embeddings, arxiv_ids)

    # Save
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