import time
import math
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src.model import encode_texts, truncate_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Constructs an exact Inner Product (FlatIP) FAISS index.
    Because vectors are L2-normalized, inner product equals cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    #Ensure float32 contiguous array for C++ FAISS engine
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index

def compute_metrics(
    retrieved_doc_ids: List[List[str]],
    qrels: Dict[str,List[str]],
    query_ids: List[str],
    top_k: int = 10
) -> Dict[str, float]:
    """
    Computes Recall@K, MRR@K, and nDCG@K
    """
    recall_hits = 0
    rr_total = 0.0
    ndcg_total = 0.0

    valid_queries = 0

    for i, qid in enumerate(query_ids):
        relevant = set(qrels.get(qid,[]))
        if not relevant:
            continue

        valid_queries += 1
        retrieved = retrieved_doc_ids[i][:top_k]

        # 1. Recall@K: Did at least one relevant document apper in Top-K?
        hit = any(doc in relevant for doc in retrieved)
        if hit:
            recall_hits +=1

        # 2. MRR@K: Reciprocal rank of the FIRST relevant document
        rr = 0.0
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                rr = 1.0 / rank
                break
        rr_total += rr

        # 3. nDCG@K (Binary relevance)
        dcg = 0.0

        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG (best possible rank)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), top_k) + 1))
        ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

    return {
        f"Recall@{top_k}": round(recall_hits / valid_queries, 4),
        f"MRR@{top_k}": round(rr_total / valid_queries, 4),
        f"nDCG@{top_k}": round(ndcg_total / valid_queries, 4),
    }

def evaluate_retrieval(
    corpus_dict: Dict[str, str],
    queries_dict: Dict[str, str],
    qrels_dict: Dict[str, List[str]],
    model: SentenceTransformer,
    target_dim: Optional[int] = None,
    top_k: int = 10,
    batch_size: int =64
) -> Dict[str, float]:
    """
    Evaluates retrieval for a single dimension slice.
    """
    corpus_ids = list(corpus_dict.keys())
    corpus_texts = [corpus_dict[cid] for cid in corpus_ids]

    query_ids = list(queries_dict.keys())
    query_texts = [queries_dict[qid] for qid in query_ids]
    
    logger.info(f"Encoding {len(corpus_texts)} corpus texts at dim={target_dim or 'native'}...")
    corpus_embeddings = encode_texts(model, corpus_texts, batch_size=batch_size, target_dim=target_dim)

    logger.info(f"Encoding {len(query_texts)} queries at dim={target_dim or 'native'}...")
    query_embeddings = encode_texts(model, query_texts, batch_size=batch_size, target_dim=target_dim)

    # Build FAISS index and measure search time
    index = build_faiss_index(corpus_embeddings)

    start_time = time.perf_counter()
    _, indices = index.search(np.ascontiguousarray(query_embeddings, dtype=np.float32), top_k)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency_ms = round(elapsed_ms / len(query_ids),3)

    #Map FAISS integer indices back to document string IDs
    retrieved_doc_ids = [[corpus_ids[idx] for idx in row if idx != -1] for row in indices]

    metrics = compute_metrics(retrieved_doc_ids, qrels_dict, query_ids, top_k=top_k)
    metrics["Avg_Latency_ms"] = avg_latency_ms

    # Calculate storage per 1M docs in Megabytes (FP32 = 4 bytes)
    eval_dim = target_dim if target_dim else corpus_embeddings.shape[1]
    storage_mb_per_1m = round((eval_dim * 4 * 1_000_000) / (1024 * 1024), 1)
    metrics["Storage_per_1M_MB"] = storage_mb_per_1m
    metrics["Dimension"] = eval_dim

    return metrics


def run_dimensional_benchmarks(
    corpus_dict: Dict[str, str],
    queries_dict: Dict[str, str],
    qrels_dict: Dict[str, List[str]],
    model: SentenceTransformer,
    dimensions: List[int] = [768, 512, 256, 128, 64],
    top_k: int = 10
) -> pd.DataFrame:
    """
    Runs systematic across all target dimensions and compiles a results DataFrame.
    """

    records = []
    baseline_storage = None

    for dim in dimensions:
        logger.info(f"---Evaluating Dimension; {dim} ---")
        metrics = evaluate_retrieval(corpus_dict, queries_dict, qrels_dict, model, target_dim=dim, top_k=top_k)

        if baseline_storage is None:
            baseline_storage = metrics["Storage_per_1M_MB"]
            metrics["Storage_Savings_%"] = 0.0
        else:
            savings = (1.0 - (metrics["Storage_per_1M_MB"] / baseline_storage)) * 100
            metrics["Storage_Savings_%"] = round(savings, 1)

        records.append(metrics)

    df = pd.DataFrame(records)
    # Order columns cleanly
    cols = ["Dimension", f"Recall@{top_k}", f"MRR@{top_k}", f"nDCG@{top_k}", "Storage_per_1M_MB", "Storage_Savings_%", "Avg_Latency_ms"]
    df = df[cols]
    return df
        

#                 TEST DATA
#                    │
#          ┌─────────┴─────────┐
#          ↓                   ↓
#       Queries              Corpus
#      Q1, Q2...             D1, D2...
#          │                   │
#          └─────────┬─────────┘
#                    ↓
#             Trained Model
#                    │
#          ┌─────────┴─────────┐
#          ↓                   ↓
#     Query embeddings    Document embeddings
#          │                   │
#          └─────────┬─────────┘
#                    ↓
#              FAISS Index
#                    ↓
#          Similarity Search
#                    ↓
#              Top-K docs
#                    │
#                    ↓
#                 Qrels
#                    │
#                    ↓
#       Recall / MRR / nDCG
    



