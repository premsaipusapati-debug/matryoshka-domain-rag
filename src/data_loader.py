import logging
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_scifact_raw(dataset_name: str = "mteb/scifact") -> Dict:
    """
    Downloads the BEIR SciFact dataset from Hugging Face.
    Contains three components: corpus, queries, and qrels (train/test).
    """
    logger.info(f"Loading dataset: {dataset_name}...")
    corpus_ds = load_dataset(dataset_name, "corpus")["corpus"]
    queries_ds = load_dataset(dataset_name, "queries")["queries"]
    qrels_train_ds = load_dataset(dataset_name, "default")["train"]
    qrels_test_ds = load_dataset(dataset_name, "default")["test"]

    return {
        "corpus": corpus_ds,
        "queries": queries_ds,
        "qrels_train": qrels_train_ds,
        "qrels_test": qrels_test_ds,
    }


def parse_corpus(corpus_ds) -> Dict[str, str]:
    """
    Transforms corpus records into a lookup dictionary: {doc_id: formatted_text}.
    Combines paper title and body text for richer semantic representation.
    """
    corpus = {}
    for row in corpus_ds:
        doc_id = str(row["_id"])
        title = row.get("title", "").strip()
        text = row.get("text", "").strip()
        full_text = f"{title} {text}".strip() if title else text
        corpus[doc_id] = full_text
    logger.info(f"Parsed {len(corpus)} corpus documents.")
    return corpus


def parse_queries(queries_ds) -> Dict[str, str]:
    """
    Transforms queries into a lookup dictionary: {query_id: query_text}.
    """
    queries = {str(row["_id"]): row["text"].strip() for row in queries_ds}
    logger.info(f"Parsed {len(queries)} queries.")
    return queries


def create_training_data(
    raw_data: Dict,
    max_samples: Optional[int] = None
) -> List[InputExample]:
    """
    Constructs positive pairs (query, positive_doc) for MultipleNegativesRankingLoss.
    In-batch negatives are automatically drawn from other positive documents in the batch.
    """
    corpus = parse_corpus(raw_data["corpus"])
    queries = parse_queries(raw_data["queries"])
    qrels_train = raw_data["qrels_train"]

    train_examples: List[InputExample] = []
    seen_pairs = set()

    for row in qrels_train:
        qid = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        score = int(row.get("score", 1))

        # Only retain positive relations (score >= 1)
        if score > 0 and qid in queries and doc_id in corpus:
            pair_key = (qid, doc_id)
            if pair_key not in seen_pairs:
                query_text = queries[qid]
                doc_text = corpus[doc_id]
                train_examples.append(InputExample(texts=[query_text, doc_text]))
                seen_pairs.add(pair_key)

        if max_samples and len(train_examples) >= max_samples:
            break

    logger.info(f"Generated {len(train_examples)} unique training pairs for MNRL.")
    return train_examples


def create_training_dataloader(
    train_examples: List[InputExample],
    batch_size: int = 32
) -> DataLoader:
    """
    Wraps InputExamples into a PyTorch DataLoader.
    Crucial: shuffle=True guarantees dynamic diversity of in-batch negatives per epoch.
    """
    return DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True  # Drops incomplete final batch to maintain consistent in-batch negative count
    )


def get_eval_data(
    raw_data: Dict,
    split: str = "test",
    max_samples: Optional[int] = None
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Extracts evaluation data structures:
    - corpus: {doc_id: text}
    - queries: {query_id: text}
    - qrels: {query_id: [relevant_doc_ids]}
    """
    corpus = parse_corpus(raw_data["corpus"])
    queries_all = parse_queries(raw_data["queries"])
    qrels_ds = raw_data["qrels_test"] if split == "test" else raw_data["qrels_train"]

    queries = {}
    qrels = {}

    for row in qrels_ds:
        qid = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        score = int(row.get("score", 1))

        if score > 0 and qid in queries_all and doc_id in corpus:
            if qid not in queries:
                if max_samples and len(queries) >= max_samples:
                    continue
                queries[qid] = queries_all[qid]
                qrels[qid] = []
            
            if qid in qrels:
                qrels[qid].append(doc_id)

    logger.info(f"Prepared eval split '{split}': {len(queries)} queries across {len(corpus)} documents.")
    return corpus, queries, qrels
