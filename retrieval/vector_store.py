import json
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

    def reconcile_id(self, old_id: str, new_id: str) -> bool:
        """
        Atomically reconcile a document ID from old_id to new_id across:
        1. SQLite persistence tables (document_weights, inferences if present).
        2. In-memory document storage (self.documents) and vectors (self.vectors).
        Ensures in-memory documents/vectors and persisted weights/telemetry
        references remain consistent. Documents and vectors are rehydrated
        when the store is initialized.
        Raises RuntimeError on persistence failure without leaving partial state.
        """
        if old_id == new_id:
            return True

        if old_id not in self.documents and not self.db_path:
            return False

        # Collision prevention: target ID must not already exist in documents or vectors
        if new_id in self.documents:
            raise ValueError(
                f"Cannot reconcile document ID: target ID '{new_id}' already exists in vector store documents."
            )
        if new_id in self.vectors:
            raise ValueError(
                f"Cannot reconcile document ID: target ID '{new_id}' already exists in vector store vectors."
            )

        old_doc = self.documents.get(old_id)
        old_vec = self.vectors.get(old_id)

        conn = self._get_connection() if self.db_path else None
        try:
            if conn:
                # Begin immediate write-locked transaction to prevent concurrent insertion races
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.cursor()

                # Verify target ID doesn't already exist in persisted weights within this same transaction
                cursor.execute("SELECT 1 FROM document_weights WHERE doc_id = ?", (new_id,))
                if cursor.fetchone() is not None:
                    raise ValueError(
                        f"Cannot reconcile document ID: target ID '{new_id}' already exists in persisted document weights."
                    )

                now_iso = datetime.now(timezone.utc).isoformat()

                # 1. document_weights migration (strict INSERT without ON CONFLICT to prevent overwriting)
                cursor.execute(
                    "SELECT dynamic_weight FROM document_weights WHERE doc_id = ?",
                    (old_id,)
                )
                row = cursor.fetchone()
                if row is not None:
                    migrated_weight = float(row["dynamic_weight"])
                    cursor.execute("""
                        INSERT INTO document_weights (doc_id, dynamic_weight, updated_at)
                        VALUES (?, ?, ?)
                    """, (new_id, migrated_weight, now_iso))
                    cursor.execute("DELETE FROM document_weights WHERE doc_id = ?", (old_id,))
                elif old_id in self.documents:
                    mem_weight = float(self.documents[old_id].get("dynamic_weight", 0.0))
                    if mem_weight != 0.0:
                        cursor.execute("""
                            INSERT INTO document_weights (doc_id, dynamic_weight, updated_at)
                            VALUES (?, ?, ?)
                        """, (new_id, mem_weight, now_iso))

                # 2. inferences table migration (if inferences table exists in same db)
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='inferences'"
                )
                if cursor.fetchone() is not None:
                    cursor.execute(
                        "SELECT id, doc_ids FROM inferences WHERE doc_ids LIKE ?",
                        (f'%"{old_id}"%',)
                    )
                    inf_rows = cursor.fetchall()
                    for inf in inf_rows:
                        raw_doc_ids = inf["doc_ids"]
                        try:
                            d_ids = json.loads(raw_doc_ids) if raw_doc_ids else []
                        except Exception as parse_err:
                            raise ValueError(
                                f"Malformed telemetry JSON in inferences record '{inf['id']}': {parse_err}"
                            ) from parse_err
                        if not isinstance(d_ids, list):
                            raise ValueError(
                                f"Invalid doc_ids format in inferences record '{inf['id']}': expected list, got {type(d_ids).__name__}"
                            )
                        if old_id in d_ids:
                            updated_ids = [new_id if x == old_id else x for x in d_ids]
                            cursor.execute(
                                "UPDATE inferences SET doc_ids = ? WHERE id = ?",
                                (json.dumps(updated_ids), inf["id"])
                            )

            # In-memory migration (runs BEFORE committing SQLite transaction)
            if old_id in self.documents:
                doc = self.documents.pop(old_id)
                doc["id"] = new_id
                self.documents[new_id] = doc

            if old_id in self.vectors:
                self.vectors[new_id] = self.vectors.pop(old_id)

            # Atomically commit SQLite changes ONLY after in-memory migration succeeds
            if conn:
                conn.commit()

        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            # Rollback in-memory changes if modified
            if old_doc is not None:
                old_doc["id"] = old_id
                self.documents[old_id] = old_doc
                self.documents.pop(new_id, None)
            if old_vec is not None:
                self.vectors[old_id] = old_vec
                self.vectors.pop(new_id, None)

            if isinstance(e, ValueError) and "already exists in persisted document weights" in str(e):
                raise
            raise RuntimeError(
                f"Failed to reconcile ID from '{old_id}' to '{new_id}': {e}"
            ) from e
        finally:
            if conn:
                conn.close()

        return True

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
