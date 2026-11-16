import zlib
import re
import math
import numpy as np
from typing import List

class LocalEmbeddingEngine:
    """
    Lightweight, fast, local embedding engine using deterministic subword-aware
    hashed dense projections (CRC32). Guarantees 100% reproducible embeddings across
    independent Python processes and operating system restarts.
    """
    ENGINE_VERSION = "crc32-v1"

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Word tokens + character 3-grams for robust partial and typo matching
        words = re.findall(r"\b\w+\b", text)
        ngrams = []
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    ngrams.append(word[i:i+3])
        return words + ngrams

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            # Deterministic, process-stable hash projection into embedding dimensions
            h = zlib.crc32(token.encode("utf-8"))
            idx = h % self.dimension
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def embed_documents(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed_text(t) for t in texts]

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        dot = float(np.dot(vec_a, vec_b))
        sim = dot / (norm_a * norm_b)
        # Clamp to [0.0, 1.0] for safety
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))

