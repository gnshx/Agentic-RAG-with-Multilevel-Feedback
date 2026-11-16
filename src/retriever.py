import re
import math
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from core.embeddings import LocalEmbeddingEngine

class HybridRetriever:
    """
    Hybrid Retriever combining BM25 lexical ranking and dense vector representations
    using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None, embedding_engine: Optional[LocalEmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or LocalEmbeddingEngine(dimension=128)
        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.dense_vectors: List[np.ndarray] = []

        if corpus:
            self.index_corpus(corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index_corpus(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc.get("text", "") + " " + doc.get("title", "")) for doc in corpus]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self.dense_vectors = [
                self.embedding_engine.embed_text(doc.get("text", "") + " " + doc.get("title", ""))
                for doc in corpus
            ]
        else:
            self.bm25 = None
            self.dense_vectors = []

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.corpus:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return self.corpus[:top_k]
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            doc = dict(self.corpus[idx])
            doc["bm25_score"] = float(scores[idx])
            doc["bm25_rank"] = rank + 1
            results.append(doc)
        return results

    def search_dense(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.dense_vectors or not self.corpus:
            return []
        q_vec = self.embedding_engine.embed_text(query)
        sims = [self.embedding_engine.cosine_similarity(q_vec, d_vec) for d_vec in self.dense_vectors]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            doc = dict(self.corpus[idx])
            doc["dense_score"] = float(sims[idx])
            doc["dense_rank"] = rank + 1
            results.append(doc)
        return results

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Retrieval using Reciprocal Rank Fusion:
        RRF_score(d) = sum( 1 / (rrf_k + rank_m(d)) )
        """
        if not self.corpus:
            return []

        top_pool = max(k * 2, 10)
        bm25_results = self.search_bm25(query, top_k=top_pool)
        dense_results = self.search_dense(query, top_k=top_pool)

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Accumulate BM25 ranks
        for item in bm25_results:
            did = item["id"]
            doc_map[did] = item
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + item["bm25_rank"]))

        # Accumulate Dense ranks
        for item in dense_results:
            did = item["id"]
            if did not in doc_map:
                doc_map[did] = item
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + item["dense_rank"]))

        # Sort by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        final_docs: List[Dict[str, Any]] = []

        for did in sorted_ids[:k]:
            res = dict(doc_map[did])
            res["score"] = round(rrf_scores[did], 6)
            final_docs.append(res)

        return final_docs

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Lightweight cross-match reranker for Level 2 processing.
        Weights title matches, exact phrase hits, and semantic embedding alignment.
        """
        if not docs:
            return []
        q_tokens = set(self._tokenize(query))
        q_vec = self.embedding_engine.embed_text(query)
        reranked = []

        for d in docs:
            doc_tokens = set(self._tokenize(d.get("text", "") + " " + d.get("title", "")))
            overlap = len(q_tokens.intersection(doc_tokens)) / max(len(q_tokens), 1)
            d_vec = self.embedding_engine.embed_text(d.get("text", ""))
            sem_sim = self.embedding_engine.cosine_similarity(q_vec, d_vec)

            # Combined rerank score: 60% semantic + 40% keyword overlap
            rerank_score = 0.6 * sem_sim + 0.4 * overlap
            d_copy = dict(d)
            d_copy["rerank_score"] = round(rerank_score, 4)
            reranked.append((rerank_score, d_copy))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in reranked[:top_k]]

