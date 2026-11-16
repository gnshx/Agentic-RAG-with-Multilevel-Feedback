import numpy as np
from typing import List, Dict, Optional, Tuple
from core.state import DocumentChunk
from core.embeddings import LocalEmbeddingEngine

class LocalVectorStore:
    """
    Local Vector Store supporting:
    - Real-time in-memory vector indexing and search
    - Dynamic Weight Multipliers modified by Level 3 feedback loops
    - Re-ranking based on combined (similarity * (1 + dynamic_weight))
    """
    def __init__(self, embedding_engine: Optional[LocalEmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or LocalEmbeddingEngine()
        self.documents: Dict[str, DocumentChunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}

    def add_documents(self, docs: List[DocumentChunk]):
        for doc in docs:
            doc_id = doc["id"]
            self.documents[doc_id] = doc
            self.vectors[doc_id] = self.embedding_engine.embed_text(doc["content"])

    def update_weight(self, doc_id: str, delta: float) -> bool:
        """
        Adjust dynamic weight for a document chunk based on Level 3 feedback.
        Positive feedback raises the weight; negative feedback penalizes it.
        """
        if doc_id in self.documents:
            current = self.documents[doc_id].get("dynamic_weight", 0.0)
            # Bound dynamic weight between -0.8 and +1.5 to prevent extreme drift
            new_weight = max(-0.8, min(1.5, current + delta))
            self.documents[doc_id]["dynamic_weight"] = round(new_weight, 4)
            return True
        return False

    def search(self, query: str, top_k: int = 4) -> List[DocumentChunk]:
        if not self.documents:
            return []

        query_vec = self.embedding_engine.embed_text(query)
        scored_docs: List[Tuple[float, DocumentChunk]] = []

        for doc_id, doc in self.documents.items():
            doc_vec = self.vectors[doc_id]
            base_sim = self.embedding_engine.cosine_similarity(query_vec, doc_vec)
            dynamic_weight = doc.get("dynamic_weight", 0.0)

            # Combined score: base cosine similarity boosted or penalized by dynamic feedback weight
            combined_score = base_sim * (1.0 + dynamic_weight)
            combined_score = max(0.0, min(1.0, combined_score))

            doc_copy: DocumentChunk = {
                "id": str(doc["id"]),
                "content": str(doc["content"]),
                "source": str(doc["source"]),
                "score": float(round(combined_score, 4)),
                "dynamic_weight": float(dynamic_weight),
                "metadata": dict(doc.get("metadata", {}), base_similarity=float(round(base_sim, 4)))
            }
            scored_docs.append((float(combined_score), doc_copy))

        # Sort descending by final combined score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
