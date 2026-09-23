import logging
from typing import Dict, List, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from src.model import encode_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MatryoshkaRAGPipeline:
    """
    End-to-End Minimal Viable Product RAG Pipeline using 256d Matryoshka Embeddings.
    """
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        corpus_dict: Dict[str, str],
        inference_dim: int = 256,
        generator_model_name: str = "google/flan-t5-base",
        device: Optional[str] = None
    ):
        self.embedding_model = embedding_model
        self.corpus_dict = corpus_dict
        self.corpus_ids = list(corpus_dict.keys())
        self.inference_dim = inference_dim

        logger.info(f"Indexing {len(self.corpus_ids)} documents at truncated dim={inference_dim}...")
        corpus_texts = [self.corpus_dict[cid] for cid in self.corpus_ids]
        
        # Pre-encode and truncate corpus embeddings to 256d
        self.corpus_embeddings = encode_texts(
            self.embedding_model,
            corpus_texts,
            target_dim=self.inference_dim,
            show_progress_bar=True
        )

        # Build FAISS index for high-speed retrieval
        self.index = faiss.IndexFlatIP(self.inference_dim)
        self.index.add(np.ascontiguousarray(self.corpus_embeddings, dtype=np.float32))

        # Setup generative LLM
        logger.info(f"Loading generative model: {generator_model_name}...")
        dev_idx = 0 if device == "cuda" else -1
        self.generator = pipeline(
            "text2text-generation",
            model=generator_model_name,
            device=dev_idx
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encodes query, truncates to 256d, and retrieves Top-K passages from FAISS.
        """
        query_vec = encode_texts(
            self.embedding_model,
            [query],
            target_dim=self.inference_dim,
            show_progress_bar=False
        )

        scores, indices = self.index.search(
            np.ascontiguousarray(query_vec, dtype=np.float32),
            top_k
        )

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx != -1:
                doc_id = self.corpus_ids[idx]
                results.append({
                    "rank": rank + 1,
                    "doc_id": doc_id,
                    "score": round(float(scores[0][rank]), 4),
                    "text": self.corpus_dict[doc_id]
                })
        return results

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Injects retrieved context into prompt and generates response with LLM.
        """
        context_str = "\n\n".join([f"Passage {d['rank']}: {d['text']}" for d in retrieved_docs])
        
        prompt = (
            f"Context information:\n{context_str}\n\n"
            f"Based solely on the scientific context above, answer the question accurately:\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        response = self.generator(prompt, max_new_tokens=200, do_sample=False)
        return response[0]["generated_text"].strip()

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes end-to-end RAG: retrieve candidates + synthesize answer.
        """
        candidates = self.retrieve(query_text, top_k=top_k)
        answer = self.generate_answer(query_text, candidates)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_context": candidates,
            "inference_dim": self.inference_dim
        }
