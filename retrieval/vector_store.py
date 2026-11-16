import sqlite3
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from core.state import DocumentChunk
from core.embeddings import LocalEmbeddingEngine

class LocalVectorStore:
    """
    Local Vector Store supporting:
    - Real-time in-memory vector indexing and search
    - Dynamic Weight Multipliers modified by Level 3 feedback loops
    - SQLite persistence of document weights across process restarts
    - Re-ranking based on combined (similarity * (1 + dynamic_weight))
    """
    def __init__(
        self,
        embedding_engine: Optional[LocalEmbeddingEngine] = None,
        db_path: Optional[str] = None
    ):
        self.embedding_engine = embedding_engine or LocalEmbeddingEngine()
        self.db_path = db_path
        self.documents: Dict[str, DocumentChunk] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        if self.db_path:
            self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        if not self.db_path:
            return
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_weights (
                    doc_id TEXT PRIMARY KEY,
                    dynamic_weight REAL,
                    updated_at TIMESTAMP
                )
            """)
            conn.commit()

    def _persist_weight(self, doc_id: str, dynamic_weight: float):
        if not self.db_path:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO document_weights (doc_id, dynamic_weight, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    dynamic_weight = excluded.dynamic_weight,
                    updated_at = excluded.updated_at
            """, (doc_id, dynamic_weight, now_iso))
            conn.commit()

    def _get_persisted_weights(self, doc_ids: List[str]) -> Dict[str, float]:
        if not self.db_path or not doc_ids:
            return {}
        weights = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in doc_ids)
            cursor.execute(
                f"SELECT doc_id, dynamic_weight FROM document_weights WHERE doc_id IN ({placeholders})",
                doc_ids
            )
            for row in cursor.fetchall():
                weights[row["doc_id"]] = float(row["dynamic_weight"])
        return weights

    def add_documents(self, docs: List[DocumentChunk]):
        persisted_weights = self._get_persisted_weights([doc["id"] for doc in docs]) if self.db_path else {}
        for doc in docs:
            doc_id = doc["id"]
            if doc_id in persisted_weights:
                doc["dynamic_weight"] = persisted_weights[doc_id]
            self.documents[doc_id] = doc
            self.vectors[doc_id] = self.embedding_engine.embed_text(doc["content"])

    def update_weight(self, doc_id: str, delta: float) -> bool:
        """
        Adjust dynamic weight for a document chunk based on Level 3 feedback.
        Positive feedback raises the weight; negative feedback penalizes it.
        Bounds dynamic weight between -0.8 and +1.5 to prevent extreme drift.
        Persists to SQLite if db_path is configured.
        """
        if doc_id in self.documents:
            current = self.documents[doc_id].get("dynamic_weight", 0.0)
            new_weight = round(max(-0.8, min(1.5, current + delta)), 4)
            self.documents[doc_id]["dynamic_weight"] = new_weight
            if self.db_path:
                self._persist_weight(doc_id, new_weight)
            return True
        elif self.db_path:
            persisted = self._get_persisted_weights([doc_id])
            if doc_id in persisted:
                current = persisted[doc_id]
                new_weight = round(max(-0.8, min(1.5, current + delta)), 4)
                self._persist_weight(doc_id, new_weight)
                return True
        return False

    def get_weight(self, doc_id: str) -> float:
        if doc_id in self.documents:
            return float(self.documents[doc_id].get("dynamic_weight", 0.0))
        if self.db_path:
            weights = self._get_persisted_weights([doc_id])
            return weights.get(doc_id, 0.0)
        return 0.0

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
