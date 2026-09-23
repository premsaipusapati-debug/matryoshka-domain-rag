import logging
from typing import List, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_device() -> str:
    """Selects CUDA if an NVIDIA GPU is available, otherwise CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Execution device selected: {device.upper()}")
    return device

def load_embedding_model(
    model_name_or_path: str = "BAAI/bge-base-en-v1.5",
    max_seq_length: int = 512,
    device: Optional[str] = None
) -> SentenceTransformer:
    """
    Initializes a SentenceTransformer model and configures its maximum sequence length.
    """
    if device is None:
        device = get_device()
    logger.info(f"Loading SentenceTransformer from '{model_name_or_path}' on {device}....")
    model = SentenceTransformer(model_name_or_path, device=device)
    model.max_seq_length = max_seq_length
    return model

def truncate_embeddings(
    embeddings: np.ndarray,
    target_dim: Optional[int] = None
) -> np.ndarray:
    """
    Performs Matryoshka dimensioality truncation and re-applies L2 normalization.

    CRITICAL MATHEMATICAL STEP:
    When a vector is truncated (e.g., from 768d to 256d), its L2 norm is no longer 1.0.
    Re-normalizing ensures that inner product calculations strictly represent cosine simmilarity:
    ConsineSimilarity(u, v) = u_norm . v_norm
    """
    if target_dim is None or target_dim >= embeddings.shape[1]:
        return embeddings

    # 1. Array Slicing: take the first `target_dim` numbers
    sliced = embeddings[:, :target_dim]

    # 2. L2 Re-normalization: divide each vector by its Euclidean norm
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    # Prevent division by zero with small epsilon
    normalized = sliced / np.maximum(norms, 1e-12)

    return normalized.astype(np.float32)

def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
    target_dim: Optional[int] = None,
    show_progress_bar: bool = True
) -> np.ndarray:
    """
    Encodes a list of texts into vectors with optional dimension truncation.
    """
    raw_embeddings = model.encode(
        texts,
        batch_size = batch_size,
        show_progress_bar = show_progress_bar,
        normalize_embeddings = True,
        convert_to_numpy = True
    )

    if target_dim is not None:
        return truncate_embeddings(raw_embeddings, target_dim = target_dim)

    return raw_embeddings
    

#                 SciFact
#                    │
#                    ▼
#        ┌──────────────────────┐
#        │ Training Data        │
#        │                      │
#        │ Query → Positive Doc │
#        └──────────┬───────────┘
#                   │
#                   ▼
#          Sentence Transformer
#                   │
#                   ▼
#             Embeddings
#                   │
#          ┌────────┴────────┐
#          │                 │
#          ▼                 ▼
#       Query            Document
#      vector             vector
#          │                 │
#          └────────┬────────┘
#                   ▼
#            Similarity
#                   │
#                   ▼
#             Retrieval